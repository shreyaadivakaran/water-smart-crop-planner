"""
validate.py
-----------
Data validation checks for all five raw datasets.

Checks performed
----------------
* Missing value ratio per column — WARNING if > 5 %.
* Unit plausibility:
  - Rainfall ≥ 0 mm/day
  - Temperatures in [−5, 60] °C
  - Price > 0 ₹/quintal
  - Yield > 0 kg/ha
  - Cost > 0 ₹/ha
* Year coverage: warns if any year in [start_year, end_year] is absent.

The validator does NOT raise exceptions; the pipeline always continues.
All warnings are written to ``logs/data_validation.log`` via the
:func:`get_validation_logger` helper.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.config_loader import load_config
from src.utils.logger import get_validation_logger

val_logger = get_validation_logger()

_MISSING_RATIO_THRESHOLD = 0.05  # 5 %


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _check_missing_ratio(df: pd.DataFrame, name: str) -> None:
    """Warn for any column whose missing-value ratio exceeds the threshold.

    Parameters
    ----------
    df:   DataFrame to check.
    name: Human-readable name of the dataset (for log messages).
    """
    total = len(df)
    if total == 0:
        val_logger.warning("[%s] DataFrame is empty — skipping checks.", name)
        return
    for col in df.columns:
        ratio = df[col].isna().sum() / total
        if ratio > _MISSING_RATIO_THRESHOLD:
            val_logger.warning(
                "[%s] Column '%s' has %.1f%% missing values (threshold %.0f%%).",
                name, col, ratio * 100, _MISSING_RATIO_THRESHOLD * 100,
            )


def _check_year_coverage(
    df: pd.DataFrame,
    year_col: str,
    start_year: int,
    end_year: int,
    name: str,
) -> None:
    """Warn for any year in [start_year, end_year] absent from the dataset.

    Parameters
    ----------
    df:         DataFrame to check.
    year_col:   Name of the year column.
    start_year: First expected year.
    end_year:   Last expected year.
    name:       Dataset name for log messages.
    """
    present = set(df[year_col].dropna().astype(int))
    expected = set(range(start_year, end_year + 1))
    missing = sorted(expected - present)
    if missing:
        val_logger.warning(
            "[%s] Missing years in coverage: %s", name, missing
        )


def _check_positive(
    df: pd.DataFrame,
    col: str,
    name: str,
    unit: str = "",
) -> None:
    """Warn if any value in ``col`` is non-positive.

    Parameters
    ----------
    df:   DataFrame.
    col:  Column name to check.
    name: Dataset name.
    unit: Unit string for the log message.
    """
    if col not in df.columns:
        return
    bad = (df[col] <= 0).sum()
    if bad > 0:
        val_logger.warning(
            "[%s] Column '%s' has %d non-positive value(s)%s.",
            name, col, bad, f" (expected > 0 {unit})" if unit else "",
        )


def _check_range(
    df: pd.DataFrame,
    col: str,
    lo: float,
    hi: float,
    name: str,
    unit: str = "",
) -> None:
    """Warn if any value in ``col`` falls outside [lo, hi].

    Parameters
    ----------
    df:   DataFrame.
    col:  Column name.
    lo:   Lower bound (inclusive).
    hi:   Upper bound (inclusive).
    name: Dataset name.
    unit: Unit string for messages.
    """
    if col not in df.columns:
        return
    out_of_range = ((df[col] < lo) | (df[col] > hi)).sum()
    if out_of_range > 0:
        val_logger.warning(
            "[%s] Column '%s' has %d value(s) outside [%g, %g]%s.",
            name, col, out_of_range, lo, hi,
            f" {unit}" if unit else "",
        )


# ---------------------------------------------------------------------------
# Per-dataset validators
# ---------------------------------------------------------------------------

def validate_weather(df: pd.DataFrame, cfg: dict[str, Any]) -> None:
    """Run plausibility checks on the daily weather DataFrame.

    Parameters
    ----------
    df:  Daily weather DataFrame (post-cleaning).
    cfg: Parsed configuration dictionary.
    """
    name = "weather"
    _check_missing_ratio(df, name)

    # Year coverage (derived from date column)
    if "date" in df.columns:
        years_present = pd.to_datetime(df["date"]).dt.year
        df_tmp = pd.DataFrame({"year": years_present})
        _check_year_coverage(
            df_tmp, "year",
            cfg["study_period"]["start_year"],
            cfg["study_period"]["end_year"],
            name,
        )

    # Plausibility
    _check_range(df, "PRECTOTCORR",       0,   500, name, "mm/day")
    _check_range(df, "T2M_MAX",           -5,   60, name, "°C")
    _check_range(df, "T2M_MIN",           -5,   60, name, "°C")
    _check_range(df, "RH2M",               0,  100, name, "%")
    _check_range(df, "WS2M",               0,   30, name, "m/s")
    _check_range(df, "ALLSKY_SFC_SW_DWN",  0,   40, name, "MJ/m²/day")

    val_logger.info("[%s] Validation complete.", name)


def validate_yield(df: pd.DataFrame, cfg: dict[str, Any]) -> None:
    """Run plausibility checks on the yield DataFrame.

    Parameters
    ----------
    df:  Cleaned yield DataFrame.
    cfg: Parsed configuration dictionary.
    """
    name = "yield"
    _check_missing_ratio(df, name)
    _check_year_coverage(
        df, "year",
        cfg["study_period"]["start_year"],
        cfg["study_period"]["end_year"],
        name,
    )
    _check_positive(df, "yield_kg_ha", name, "kg/ha")
    _check_positive(df, "area_ha",     name, "ha")
    val_logger.info("[%s] Validation complete.", name)


def validate_prices(df: pd.DataFrame, cfg: dict[str, Any]) -> None:
    """Run plausibility checks on the prices DataFrame.

    Parameters
    ----------
    df:  Cleaned prices DataFrame.
    cfg: Parsed configuration dictionary.
    """
    name = "prices"
    _check_missing_ratio(df, name)
    _check_year_coverage(
        df, "year",
        cfg["study_period"]["start_year"],
        cfg["study_period"]["end_year"],
        name,
    )
    _check_positive(df, "price_inr_quintal", name, "₹/quintal")
    val_logger.info("[%s] Validation complete.", name)


def validate_crop_water(df: pd.DataFrame, _cfg: dict[str, Any]) -> None:
    """Run plausibility checks on the crop water requirement DataFrame.

    Parameters
    ----------
    df:   Cleaned crop_water DataFrame.
    _cfg: Unused; kept for consistent signature.
    """
    name = "crop_water"
    _check_missing_ratio(df, name)
    _check_positive(df, "water_req_mm", name, "mm/season")
    _check_range(df, "water_req_mm", 50, 3000, name, "mm/season")
    _check_range(df, "kc_mid",        0,    3, name)
    val_logger.info("[%s] Validation complete.", name)


def validate_cost(df: pd.DataFrame, cfg: dict[str, Any]) -> None:
    """Run plausibility checks on the cost DataFrame.

    Parameters
    ----------
    df:  Cleaned cost DataFrame.
    cfg: Parsed configuration dictionary.
    """
    name = "cost"
    _check_missing_ratio(df, name)
    _check_year_coverage(
        df, "year",
        cfg["study_period"]["start_year"],
        cfg["study_period"]["end_year"],
        name,
    )
    _check_positive(df, "cost_inr_ha", name, "₹/ha")
    val_logger.info("[%s] Validation complete.", name)


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

def validate_all(
    weather_df: pd.DataFrame,
    yield_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    crop_water_df: pd.DataFrame,
    cost_df: pd.DataFrame,
    cfg: dict[str, Any],
) -> None:
    """Run all five validators.

    Parameters
    ----------
    weather_df:    Daily weather DataFrame.
    yield_df:      Cleaned yield DataFrame.
    prices_df:     Cleaned prices DataFrame.
    crop_water_df: Cleaned crop water DataFrame.
    cost_df:       Cleaned cost DataFrame.
    cfg:           Parsed configuration dictionary.
    """
    validate_weather(weather_df,    cfg)
    validate_yield(yield_df,        cfg)
    validate_prices(prices_df,      cfg)
    validate_crop_water(crop_water_df, cfg)
    validate_cost(cost_df,          cfg)
    val_logger.info("All dataset validations complete.")
