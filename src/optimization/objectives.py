"""
objectives.py
-------------
Defines the three objective functions used by the NSGA-II optimiser and
the weighted ranker.

Objectives (all expressed as minimisation for pymoo compatibility)
------------------------------------------------------------------
f1 = −profit        = −(predicted_yield × predicted_price_p50 − cost_inr_ha)
f2 = water_deficit  = max(0, water_req_mm − (seasonal_rainfall_mm + irrigation_mm))
f3 = risk_score     = normalised combined CV + dry-year failure probability

The function ``compute_objectives`` is the single entry-point used by both
the pymoo problem class (in nsga2_optimizer.py) and the dashboard.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.models.yield_model import predict_yield
from src.models.price_model import predict_price
from src.risk.risk_scorer import get_risk_score
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_objectives(
    crop: str,
    season: str,
    year: int,
    irrigation_mm: float,
    rainfall_scenario_factor: float,
    features_df: pd.DataFrame,
    risk_df: pd.DataFrame,
    yield_bundle: dict[str, Any],
    price_bundle: dict[str, Any],
) -> tuple[float, float, float]:
    """Compute the three objective values for a single (crop, season, year).

    The caller supplies pre-loaded model bundles and DataFrames so this
    function remains stateless and fast to call in a loop.

    Parameters
    ----------
    crop:                     Crop name.
    season:                   Season name (kharif | rabi | summer).
    year:                     Reference year for model inference.
    irrigation_mm:            User-supplied irrigation water available (mm).
    rainfall_scenario_factor: Multiplier on historical rainfall
                              (1.0 = normal, 0.7 = 30 % deficit).
    features_df:              Merged feature table (features.parquet).
    risk_df:                  Risk scores table (risk_scores.parquet).
    yield_bundle:             Loaded yield model bundle.
    price_bundle:             Loaded price model bundle.

    Returns
    -------
    tuple (neg_profit, water_deficit, risk_score)
        neg_profit    : −(yield × price_p50 − cost), in ₹/ha (minimise)
        water_deficit : max(0, water_req − effective_rainfall − irrigation), mm
        risk_score    : normalised risk in [0, 1]
    """
    # ---- Retrieve reference weather row for (year, season, crop) ----------
    row = features_df[
        (features_df["crop"] == crop) &
        (features_df["season"] == season) &
        (features_df["year"] == year)
    ]

    if row.empty:
        # Fallback: use seasonal average across all years
        row = features_df[
            (features_df["crop"] == crop) &
            (features_df["season"] == season)
        ]
        if row.empty:
            logger.warning(
                "No feature data for crop=%s season=%s — using global defaults.",
                crop, season,
            )
            return (0.0, 0.0, 0.5)
        row = row.mean(numeric_only=True).to_frame().T

    row = row.iloc[0]

    # Apply rainfall scenario
    base_rainfall   = float(row.get("seasonal_rainfall_mm", 500))
    adj_rainfall    = base_rainfall * rainfall_scenario_factor
    mean_tmax       = float(row.get("mean_tmax", 30))
    mean_tmin       = float(row.get("mean_tmin", 22))
    heat_stress     = float(row.get("heat_stress_days", 10))
    dry_spell       = float(row.get("dry_spell_days", 15))
    water_req_mm    = float(row.get("water_req_mm", 800))
    cost_inr_ha     = float(row.get("cost_inr_ha", 30000))

    # ---- Yield prediction -------------------------------------------------
    pred_yield = predict_yield(
        bundle=yield_bundle,
        crop=crop,
        season=season,
        year=year,
        seasonal_rainfall_mm=adj_rainfall,
        mean_tmax=mean_tmax,
        mean_tmin=mean_tmin,
        heat_stress_days=heat_stress,
        dry_spell_days=dry_spell,
    )

    # ---- Price prediction (P50) -------------------------------------------
    price_interval  = predict_price(price_bundle, crop, season, year)
    price_p50_kg    = price_interval.p50  # ₹/kg

    # Convert yield (kg/ha) × price (₹/kg) → revenue (₹/ha)
    # Convert cost from ₹/ha directly
    profit = pred_yield * price_p50_kg - cost_inr_ha

    # ---- Water deficit -----------------------------------------------------
    effective_water  = adj_rainfall + irrigation_mm
    water_deficit    = max(0.0, water_req_mm - effective_water)

    # ---- Risk score --------------------------------------------------------
    risk_score = get_risk_score(risk_df, crop, season)

    return float(-profit), float(water_deficit), float(risk_score)


def compute_objectives_batch(
    crops: list[str],
    season: str,
    year: int,
    irrigation_mm: float,
    rainfall_scenario_factor: float,
    features_df: pd.DataFrame,
    risk_df: pd.DataFrame,
    yield_bundle: dict[str, Any],
    price_bundle: dict[str, Any],
) -> pd.DataFrame:
    """Compute objectives for a list of crops and return a tidy DataFrame.

    Parameters
    ----------
    crops:                    List of crop names to evaluate.
    season:                   Season name.
    year:                     Reference year.
    irrigation_mm:            Irrigation water available (mm).
    rainfall_scenario_factor: Rainfall multiplier.
    features_df:              Feature table.
    risk_df:                  Risk scores table.
    yield_bundle:             Yield model bundle.
    price_bundle:             Price model bundle.

    Returns
    -------
    pd.DataFrame with columns:
        crop, season, neg_profit, water_deficit, risk_score,
        profit_inr_ha (= −neg_profit)
    """
    records = []
    for crop in crops:
        neg_profit, water_def, risk = compute_objectives(
            crop=crop,
            season=season,
            year=year,
            irrigation_mm=irrigation_mm,
            rainfall_scenario_factor=rainfall_scenario_factor,
            features_df=features_df,
            risk_df=risk_df,
            yield_bundle=yield_bundle,
            price_bundle=price_bundle,
        )
        records.append({
            "crop":           crop,
            "season":         season,
            "neg_profit":     neg_profit,
            "profit_inr_ha":  -neg_profit,
            "water_deficit":  water_def,
            "risk_score":     risk,
        })
    return pd.DataFrame(records)
