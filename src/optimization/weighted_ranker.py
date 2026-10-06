"""
weighted_ranker.py
------------------
Applies a user-supplied weighted-sum score to the Pareto front and
returns a ranked DataFrame.

Score formula (lower = better)
-------------------------------
Each objective is normalised to [0, 1] across the Pareto set before weighting.

    score = w_profit × neg_profit_norm
          + w_water  × water_deficit_norm
          + w_risk   × risk_score_norm

where neg_profit_norm = (−profit − min_neg_profit) / range.

Usage
-----
    from src.optimization.weighted_ranker import rank_crops
    ranked = rank_crops(pareto_df, weights={"profit": 0.5, "water": 0.3, "risk": 0.2})
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.logger import get_logger

logger = get_logger(__name__)

_WEIGHT_TOLERANCE = 0.01  # allowed deviation from sum-to-1


# ---------------------------------------------------------------------------
# Normalisation helper
# ---------------------------------------------------------------------------

def _norm(series: pd.Series) -> pd.Series:
    """Min-max normalise a Series to [0, 1].

    Returns zeros if all values are identical.

    Parameters
    ----------
    series: Numeric pandas Series.

    Returns
    -------
    pd.Series normalised to [0, 1].
    """
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rank_crops(
    pareto_df: pd.DataFrame,
    weights: dict[str, float],
) -> pd.DataFrame:
    """Rank Pareto-optimal crops by weighted-sum score.

    Parameters
    ----------
    pareto_df:
        DataFrame with columns at minimum:
        crop, season, profit_inr_ha, water_deficit, risk_score.
    weights:
        Dict with keys ``profit``, ``water``, ``risk``.
        Values must sum to 1.0 (± ``_WEIGHT_TOLERANCE``).

    Returns
    -------
    pd.DataFrame sorted by ascending ``weighted_score`` (rank 1 = best).
    Adds columns: profit_norm, water_deficit_norm, risk_score_norm,
    weighted_score, rank.

    Raises
    ------
    ValueError
        If weights do not sum to 1.0 within tolerance, or if a required
        weight key is missing.
    """
    # Validate weights
    required_keys = {"profit", "water", "risk"}
    missing = required_keys - set(weights.keys())
    if missing:
        raise ValueError(f"Missing weight keys: {missing}")

    total = sum(weights.values())
    if abs(total - 1.0) > _WEIGHT_TOLERANCE:
        raise ValueError(
            f"Weights must sum to 1.0 (got {total:.4f}). "
            "Adjust the sliders so the total is 1."
        )

    w_profit = weights["profit"]
    w_water  = weights["water"]
    w_risk   = weights["risk"]

    df = pareto_df.copy()

    # For profit we normalise −profit so lower is worse profit
    df["neg_profit"] = -df["profit_inr_ha"]

    df["profit_norm"]        = _norm(df["neg_profit"])
    df["water_deficit_norm"] = _norm(df["water_deficit"])
    df["risk_score_norm"]    = _norm(df["risk_score"])

    df["weighted_score"] = (
        w_profit * df["profit_norm"]
        + w_water  * df["water_deficit_norm"]
        + w_risk   * df["risk_score_norm"]
    )

    df = df.sort_values("weighted_score").reset_index(drop=True)
    df["rank"] = df.index + 1

    # Drop intermediate neg_profit column for cleanliness
    df = df.drop(columns=["neg_profit"])

    logger.info(
        "Ranked %d crops. Top: %s (score=%.3f)",
        len(df),
        df.iloc[0]["crop"],
        df.iloc[0]["weighted_score"],
    )
    return df


def water_saved_vs_paddy(
    ranked_df: pd.DataFrame,
) -> dict[str, float]:
    """Compute water deficit savings of the top crop vs paddy baseline.

    Parameters
    ----------
    ranked_df: Output of :func:`rank_crops`.

    Returns
    -------
    dict with keys:
        top_crop         : name of rank-1 crop
        top_water_deficit: water deficit of rank-1 crop (mm)
        paddy_water_deficit: water deficit of paddy (mm), or NaN if absent
        water_saved_mm   : paddy_deficit − top_deficit (mm)
        water_saved_pct  : percentage saving
    """
    top  = ranked_df.iloc[0]
    paddy_row = ranked_df[ranked_df["crop"] == "paddy"]

    if paddy_row.empty:
        return {
            "top_crop":            top["crop"],
            "top_water_deficit":   top["water_deficit"],
            "paddy_water_deficit": float("nan"),
            "water_saved_mm":      float("nan"),
            "water_saved_pct":     float("nan"),
        }

    paddy_def  = float(paddy_row.iloc[0]["water_deficit"])
    top_def    = float(top["water_deficit"])
    saved_mm   = paddy_def - top_def
    saved_pct  = (saved_mm / paddy_def * 100) if paddy_def > 0 else 0.0

    return {
        "top_crop":            top["crop"],
        "top_water_deficit":   top_def,
        "paddy_water_deficit": paddy_def,
        "water_saved_mm":      saved_mm,
        "water_saved_pct":     saved_pct,
    }
