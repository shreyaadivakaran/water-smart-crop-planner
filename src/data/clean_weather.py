"""
clean_weather.py
----------------
Loads ``data/raw/weather.csv``, handles missing values, and computes
per-(year, season) aggregate features used by the prediction and risk layers.

Output features per (year, season) row
---------------------------------------
seasonal_rainfall_mm   : sum of daily PRECTOTCORR over the season
mean_tmax              : mean of T2M_MAX
mean_tmin              : mean of T2M_MIN
heat_stress_days       : count of days where T2M_MAX > 35 °C
dry_spell_days         : length of the longest consecutive run of days
                         with PRECTOTCORR < 1 mm
mean_rh2m              : mean relative humidity
mean_ws2m              : mean wind speed
mean_solar_rad         : mean shortwave downwelling irradiance
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

_HEAT_STRESS_THRESHOLD_C = 35.0   # °C
_DRY_DAY_THRESHOLD_MM    = 1.0    # mm/day
_MAX_FILL_DAYS           = 3      # forward/backward fill window


def _assign_season(date: pd.Series, season_months: dict[str, list[int]]) -> pd.Series:
    """Map each date to a season label based on config month ranges.

    Parameters
    ----------
    date:          Series of datetime objects.
    season_months: Dict mapping season name → list of months.

    Returns
    -------
    pd.Series of str season labels; NaN if month not in any season.
    """
    month = date.dt.month
    season = pd.Series(index=date.index, dtype="object")
    for name, months in season_months.items():
        season[month.isin(months)] = name
    return season


def _longest_dry_spell(precip: np.ndarray) -> int:
    """Compute the longest consecutive run of dry days (precip < threshold).

    Parameters
    ----------
    precip: Array of daily precipitation values.

    Returns
    -------
    int  Maximum consecutive dry-day run length.
    """
    max_run = cur_run = 0
    for v in precip:
        if np.isnan(v) or v < _DRY_DAY_THRESHOLD_MM:
            cur_run += 1
            max_run = max(max_run, cur_run)
        else:
            cur_run = 0
    return max_run


def load_raw_weather(raw_dir: Path) -> pd.DataFrame:
    """Load raw weather CSV, skipping comment lines.

    Parameters
    ----------
    raw_dir: Directory containing ``weather.csv``.

    Returns
    -------
    pd.DataFrame with a parsed ``date`` datetime column.
    """
    path = raw_dir / "weather.csv"
    if not path.exists():
        raise FileNotFoundError(f"weather.csv not found at {path}")

    df = pd.read_csv(path, comment="#")
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    logger.info("Loaded weather.csv: %d rows", len(df))
    return df


def _fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill then backward-fill numeric columns within a short window.

    Logs the number of cells filled per column.

    Parameters
    ----------
    df: Daily weather DataFrame.

    Returns
    -------
    pd.DataFrame with NaN values reduced.
    """
    numeric_cols = [c for c in df.columns if c != "date"]
    before = df[numeric_cols].isna().sum()

    df = df.copy()
    df[numeric_cols] = (
        df[numeric_cols]
        .ffill(limit=_MAX_FILL_DAYS)
        .bfill(limit=_MAX_FILL_DAYS)
    )

    after = df[numeric_cols].isna().sum()
    filled = before - after
    for col in numeric_cols:
        if filled[col] > 0:
            logger.info("Filled %d NaN(s) in column '%s'.", filled[col], col)
    return df


def _season_aggregates(
    group: pd.DataFrame,
) -> pd.Series:
    """Compute aggregate weather features for one (year, season) group.

    Parameters
    ----------
    group: Subset of daily weather for a single year-season.

    Returns
    -------
    pd.Series of aggregate features.
    """
    precip = group["PRECTOTCORR"].to_numpy()
    tmax   = group["T2M_MAX"].to_numpy()
    tmin   = group["T2M_MIN"].to_numpy()

    return pd.Series({
        "seasonal_rainfall_mm": float(np.nansum(precip)),
        "mean_tmax":            float(np.nanmean(tmax)),
        "mean_tmin":            float(np.nanmean(tmin)),
        "heat_stress_days":     int(np.sum(tmax > _HEAT_STRESS_THRESHOLD_C)),
        "dry_spell_days":       _longest_dry_spell(precip),
        "mean_rh2m":            float(np.nanmean(group["RH2M"].to_numpy())),
        "mean_ws2m":            float(np.nanmean(group["WS2M"].to_numpy())),
        "mean_solar_rad":       float(np.nanmean(
                                    group["ALLSKY_SFC_SW_DWN"].to_numpy()
                                )),
        "n_days":               len(group),
    })


def clean_weather(cfg: dict[str, Any]) -> pd.DataFrame:
    """Full weather cleaning and feature-engineering pipeline.

    Parameters
    ----------
    cfg: Parsed configuration dictionary from :func:`load_config`.

    Returns
    -------
    pd.DataFrame
        Index: (year, season). Columns: aggregate weather features.
    """
    raw_dir      = Path(cfg["data"]["raw_dir"])
    season_months: dict[str, list[int]] = cfg["season_months"]
    start_year   = cfg["study_period"]["start_year"]
    end_year     = cfg["study_period"]["end_year"]

    df = load_raw_weather(raw_dir)
    df = _fill_missing(df)

    # Filter to study period
    df = df[
        (df["date"].dt.year >= start_year) &
        (df["date"].dt.year <= end_year)
    ].copy()

    df["year"]   = df["date"].dt.year
    df["season"] = _assign_season(df["date"], season_months)

    # Drop days that don't fall in any defined season
    unmapped = df["season"].isna().sum()
    if unmapped > 0:
        val_logger.warning(
            "Dropped %d days not mapped to any season.", unmapped
        )
    df = df.dropna(subset=["season"])

    # Compute per-season aggregates
    agg = (
        df.groupby(["year", "season"])
        .apply(_season_aggregates, include_groups=False)
        .reset_index()
    )

    logger.info(
        "Weather aggregation complete: %d (year, season) rows.", len(agg)
    )
    return agg


if __name__ == "__main__":
    cfg = load_config()
    result = clean_weather(cfg)
    print(result.head())
    print("Shape:", result.shape)
