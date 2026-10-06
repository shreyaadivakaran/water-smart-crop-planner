"""
risk_scorer.py
--------------
Computes per-(crop, season) risk scores from historical yield data.

Two components
--------------
1. **Coefficient of Variation (CV)**: std(yield) / mean(yield).
   Measures yield volatility across all years.

2. **Dry-year failure probability (P_fail)**:
   Fraction of years classified as "dry" (seasonal rainfall below the
   configured percentile) in which yield also fell below a threshold
   (default 70 % of the crop's mean yield).

Combined score
--------------
    risk_score_raw = 0.5 × CV_norm + 0.5 × P_fail_norm

Both components and the combined score are normalised to [0, 1] across
all (crop, season) pairs using min-max scaling.

Output
------
Saves ``data/processed/risk_scores.parquet`` with columns:
    crop, season, cv, p_fail, risk_score, cv_norm, p_fail_norm

Usage
-----
    python src/risk/risk_scorer.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.config_loader import load_config
from src.utils.logger import get_logger, get_validation_logger

logger     = get_logger(__name__)
val_logger = get_validation_logger()

_MIN_DRY_YEARS = 2  # fewer than this → P_fail returned as 0.0


# ---------------------------------------------------------------------------
# Core calculations
# ---------------------------------------------------------------------------

def _cv(series: np.ndarray) -> float:
    """Compute coefficient of variation; return 0.0 for constant series.

    Parameters
    ----------
    series: 1-D array of yield values.

    Returns
    -------
    float  std / mean, or 0.0 if mean is zero or series has length < 2.
    """
    if len(series) < 2:
        return 0.0
    mean = np.nanmean(series)
    if mean == 0:
        return 0.0
    return float(np.nanstd(series, ddof=1) / mean)


def _dry_year_failure_prob(
    yield_series: pd.Series,
    rainfall_series: pd.Series,
    dry_percentile: float,
    failure_threshold_frac: float,
) -> float:
    """Compute probability of yield failure in dry years.

    A year is classified as *dry* if its seasonal rainfall is below
    ``dry_percentile`` of the historical rainfall distribution.
    A year is a *failure* if yield < failure_threshold_frac × mean_yield.

    Parameters
    ----------
    yield_series:           Yield values indexed by year.
    rainfall_series:        Seasonal rainfall values indexed by year.
    dry_percentile:         Percentile threshold for dry-year classification.
    failure_threshold_frac: Fraction of mean yield below which a year is a failure.

    Returns
    -------
    float  P(failure | dry year), or 0.0 if fewer than ``_MIN_DRY_YEARS`` dry
    years exist.
    """
    aligned = pd.DataFrame({
        "yield":    yield_series,
        "rainfall": rainfall_series,
    }).dropna()

    if aligned.empty:
        return 0.0

    rain_threshold = np.percentile(aligned["rainfall"], dry_percentile)
    dry_mask       = aligned["rainfall"] < rain_threshold
    dry_years      = aligned[dry_mask]

    if len(dry_years) < _MIN_DRY_YEARS:
        return 0.0

    mean_yield        = np.nanmean(aligned["yield"])
    failure_threshold = failure_threshold_frac * mean_yield
    n_failures        = (dry_years["yield"] < failure_threshold).sum()
    return float(n_failures / len(dry_years))


def _minmax_norm(series: pd.Series) -> pd.Series:
    """Min-max normalise a Series to [0, 1].

    Parameters
    ----------
    series: Numeric Series.

    Returns
    -------
    pd.Series normalised to [0, 1]; returns zeros if all values are equal.
    """
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# Main scorer
# ---------------------------------------------------------------------------

def compute_risk_scores(
    cfg: dict[str, Any],
    features_path: Path | None = None,
) -> pd.DataFrame:
    """Compute CV, dry-year failure probability, and combined risk score.

    Parameters
    ----------
    cfg:           Parsed configuration dictionary.
    features_path: Path to features.parquet; defaults to config value.

    Returns
    -------
    pd.DataFrame with columns:
        crop, season, cv, p_fail, cv_norm, p_fail_norm, risk_score
    """
    if features_path is None:
        features_path = Path(cfg["data"]["processed_dir"]) / "features.parquet"

    df = pd.read_parquet(features_path)

    dry_pct   = cfg["risk"]["dry_year_rainfall_percentile"]
    fail_frac = cfg["risk"]["yield_failure_threshold_fraction"]

    records: list[dict[str, Any]] = []

    for (crop, season), group in df.groupby(["crop", "season"]):
        group = group.set_index("year")

        yield_series   = group["yield_kg_ha"].dropna()
        rain_series    = group["seasonal_rainfall_mm"].dropna()

        cv_val    = _cv(yield_series.to_numpy())
        p_fail    = _dry_year_failure_prob(
            yield_series, rain_series, dry_pct, fail_frac
        )

        if len(yield_series) < 2:
            val_logger.warning(
                "Crop '%s' season '%s': only %d yield observation(s) — "
                "CV and P_fail may be unreliable.",
                crop, season, len(yield_series),
            )

        records.append({
            "crop":    crop,
            "season":  season,
            "cv":      cv_val,
            "p_fail":  p_fail,
        })

    result = pd.DataFrame(records)

    # Normalise
    result["cv_norm"]    = _minmax_norm(result["cv"])
    result["p_fail_norm"] = _minmax_norm(result["p_fail"])
    result["risk_score"] = 0.5 * result["cv_norm"] + 0.5 * result["p_fail_norm"]

    # Final normalisation of combined score to [0, 1]
    result["risk_score"] = _minmax_norm(result["risk_score"])

    # Save
    processed_dir = Path(cfg["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / "risk_scores.parquet"
    result.to_parquet(out_path, index=False, engine="pyarrow")
    logger.info("Risk scores saved to %s (%d rows)", out_path, len(result))

    return result


# ---------------------------------------------------------------------------
# Lookup helper
# ---------------------------------------------------------------------------

def get_risk_score(
    risk_df: pd.DataFrame,
    crop: str,
    season: str,
) -> float:
    """Look up the normalised risk score for a (crop, season) pair.

    Parameters
    ----------
    risk_df: DataFrame from :func:`compute_risk_scores`.
    crop:    Crop name.
    season:  Season name.

    Returns
    -------
    float  Risk score in [0, 1]; 0.5 if the pair is not found.
    """
    row = risk_df[
        (risk_df["crop"] == crop) & (risk_df["season"] == season)
    ]
    if row.empty:
        val_logger.warning(
            "Risk score not found for crop='%s' season='%s'. Returning 0.5.",
            crop, season,
        )
        return 0.5
    return float(row["risk_score"].iloc[0])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg    = load_config()
    result = compute_risk_scores(cfg)
    print(result.sort_values("risk_score", ascending=False).to_string(index=False))
