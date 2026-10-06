"""
sensitivity.py
--------------
Grid sensitivity analysis: varies rainfall, price, and objective-weight
presets and records how crop rankings change across scenarios.

Grid
----
* Rainfall multipliers: −30 %, −10 %, 0, +10 %
* Price multipliers:    −20 %, 0, +20 %
* Weight presets:       profit-heavy, balanced, risk-averse

Outputs
-------
* ``outputs/sensitivity_results.csv``   — full grid results
* ``outputs/plots/sensitivity_heatmap.png`` — rank stability heatmap

Usage
-----
    python src/data/sensitivity.py
"""

from __future__ import annotations

import sys
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

_RAINFALL_FACTORS = {
    "rain-30%": 0.70,
    "rain-10%": 0.90,
    "rain+0%":  1.00,
    "rain+10%": 1.10,
}

_PRICE_FACTORS = {
    "price-20%": 0.80,
    "price+0%":  1.00,
    "price+20%": 1.20,
}

_WEIGHT_PRESETS: dict[str, dict[str, float]] = {
    "profit-heavy": {"profit": 0.7, "water": 0.2, "risk": 0.1},
    "balanced":     {"profit": 0.4, "water": 0.3, "risk": 0.3},
    "risk-averse":  {"profit": 0.2, "water": 0.3, "risk": 0.5},
}


def _apply_price_factor(obj_df: pd.DataFrame, factor: float) -> pd.DataFrame:
    """Scale the profit column by a price factor.

    Parameters
    ----------
    obj_df: Objectives DataFrame with profit_inr_ha column.
    factor: Multiplicative price adjustment.

    Returns
    -------
    pd.DataFrame with scaled profit.
    """
    df = obj_df.copy()
    df["profit_inr_ha"] = df["profit_inr_ha"] * factor
    df["neg_profit"]    = -df["profit_inr_ha"]
    return df


def run_sensitivity(
    cfg: dict[str, Any],
    season: str = "kharif",
    year: int   = 2022,
    irrigation_mm: float = 100.0,
) -> pd.DataFrame:
    """Run the full sensitivity grid and return results.

    Parameters
    ----------
    cfg:          Parsed configuration dictionary.
    season:       Season for the base scenario.
    year:         Reference year for model inference.
    irrigation_mm: Irrigation water available (mm).

    Returns
    -------
    pd.DataFrame  Columns: scenario, crop, rank, rainfall_label,
                  price_label, weight_preset.
    """
    from src.models.yield_model import load_yield_model          # noqa: PLC0415
    from src.models.price_model import load_price_model          # noqa: PLC0415
    from src.risk.risk_scorer import compute_risk_scores         # noqa: PLC0415
    from src.optimization.objectives import compute_objectives_batch  # noqa: PLC0415
    from src.optimization.nsga2_optimizer import run_nsga2       # noqa: PLC0415
    from src.optimization.weighted_ranker import rank_crops      # noqa: PLC0415

    features_df  = pd.read_parquet(
        Path(cfg["data"]["processed_dir"]) / "features.parquet"
    )
    yield_bundle = load_yield_model(cfg)
    price_bundle = load_price_model(cfg)
    risk_df      = compute_risk_scores(cfg)
    crops        = list(features_df["crop"].unique())

    records: list[dict[str, Any]] = []
    total = len(_RAINFALL_FACTORS) * len(_PRICE_FACTORS) * len(_WEIGHT_PRESETS)
    done  = 0

    for (rain_label, rain_fac), (price_label, price_fac), (wp_label, weights) in product(
        _RAINFALL_FACTORS.items(),
        _PRICE_FACTORS.items(),
        _WEIGHT_PRESETS.items(),
    ):
        scenario = f"{rain_label}|{price_label}|{wp_label}"
        done += 1
        logger.info("[%d/%d] Scenario: %s", done, total, scenario)

        try:
            obj_df = compute_objectives_batch(
                crops=crops,
                season=season,
                year=year,
                irrigation_mm=irrigation_mm,
                rainfall_scenario_factor=rain_fac,
                features_df=features_df,
                risk_df=risk_df,
                yield_bundle=yield_bundle,
                price_bundle=price_bundle,
            )
            # Apply price factor on top of model output
            obj_df = _apply_price_factor(obj_df, price_fac)

            pareto = run_nsga2(obj_df, cfg)
            ranked = rank_crops(pareto, weights)

            for _, row in ranked.iterrows():
                records.append({
                    "scenario":      scenario,
                    "rainfall_label": rain_label,
                    "price_label":   price_label,
                    "weight_preset": wp_label,
                    "crop":          row["crop"],
                    "rank":          int(row["rank"]),
                })
        except Exception as exc:
            logger.warning("Scenario '%s' failed: %s", scenario, exc)

    result_df = pd.DataFrame(records)

    # Save CSV
    out_csv = Path("outputs/sensitivity_results.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_csv, index=False)
    logger.info("Sensitivity results saved to %s (%d rows)", out_csv, len(result_df))

    # Save heatmap
    if not result_df.empty:
        from src.utils.plotting import sensitivity_heatmap  # noqa: PLC0415
        fig = sensitivity_heatmap(result_df)
        logger.info("Sensitivity heatmap saved.")

    return result_df


if __name__ == "__main__":
    cfg = load_config()
    df  = run_sensitivity(cfg)
    print(df.groupby(["crop"])["rank"].mean().sort_values())
