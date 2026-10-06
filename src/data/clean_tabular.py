"""
clean_tabular.py
----------------
Loaders and cleaners for the four non-weather raw CSV files:
  yield.csv, prices.csv, crop_water.csv, cost.csv

Key behaviours
--------------
* Drops crops with fewer than ``config.crops.min_data_years`` (default 8)
  usable yield rows, logging the crop name and exact reason.
* When ``config.data.merge_kancheepuram: true``, augments the Tiruvallur
  yield series with Kancheepuram rows for years ≤ 2019, tags them
  ``source = "merged"``, and logs a WARNING with year range and row count.
* Converts price from ₹/quintal to ₹/kg internally for the model layer.
* Forward-fills missing cost records with a WARNING.
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

_KANCHEEPURAM_MERGE_CUTOFF = 2019  # years ≤ this are pre-split


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV file, skipping comment lines (lines starting with #).

    Parameters
    ----------
    path: Path to the CSV file.

    Returns
    -------
    pd.DataFrame
    """
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found: {path}")
    return pd.read_csv(path, comment="#")


# ---------------------------------------------------------------------------
# yield.csv
# ---------------------------------------------------------------------------

def load_yield(cfg: dict[str, Any]) -> pd.DataFrame:
    """Load and clean ``yield.csv``.

    Steps
    -----
    1. Load raw file.
    2. Optionally merge Kancheepuram records (years ≤ 2019).
    3. Drop rows with non-positive yield or area values.
    4. Drop crops below the minimum data-year threshold.

    Parameters
    ----------
    cfg: Parsed configuration dictionary.

    Returns
    -------
    pd.DataFrame
        Columns: year (int), season (str), crop (str), district (str),
        yield_kg_ha (float), area_ha (float), source (str).
    """
    raw_dir  = Path(cfg["data"]["raw_dir"])
    df       = _read_csv(raw_dir / "yield.csv")

    # Enforce types
    df["year"]         = df["year"].astype(int)
    df["yield_kg_ha"]  = pd.to_numeric(df["yield_kg_ha"], errors="coerce")
    df["area_ha"]      = pd.to_numeric(df["area_ha"],      errors="coerce")
    df["season"]       = df["season"].str.strip().str.lower()
    df["crop"]         = df["crop"].str.strip().str.lower()
    df["district"]     = df["district"].str.strip()

    # Kancheepuram merge
    if cfg["data"].get("merge_kancheepuram", False):
        df = _merge_kancheepuram(df)

    # Drop clearly invalid rows
    invalid_mask = (df["yield_kg_ha"] <= 0) | (df["area_ha"] <= 0)
    if invalid_mask.any():
        val_logger.warning(
            "Dropped %d yield rows with non-positive yield or area.",
            invalid_mask.sum(),
        )
    df = df[~invalid_mask].copy()

    # Drop crops below the threshold
    df = _drop_sparse_crops(df, cfg)

    logger.info("yield.csv cleaned: %d rows, crops: %s",
                len(df), sorted(df["crop"].unique()))
    return df


def _merge_kancheepuram(df: pd.DataFrame) -> pd.DataFrame:
    """Augment Tiruvallur yield series with Kancheepuram rows for years ≤ 2019.

    Parameters
    ----------
    df: Raw yield DataFrame.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with merged rows tagged ``source = "merged"``.
    """
    kk_rows = df[
        (df["district"].str.lower() == "kancheepuram") &
        (df["year"] <= _KANCHEEPURAM_MERGE_CUTOFF)
    ].copy()

    if kk_rows.empty:
        val_logger.warning(
            "merge_kancheepuram is enabled but no Kancheepuram rows found in "
            "yield.csv for years ≤ %d. Skipping merge.",
            _KANCHEEPURAM_MERGE_CUTOFF,
        )
        return df

    kk_rows["source"]   = "merged"
    kk_rows["district"] = "Tiruvallur"  # relabel as target district

    year_range = f"{kk_rows['year'].min()}–{kk_rows['year'].max()}"
    val_logger.warning(
        "Merged %d Kancheepuram yield rows into Tiruvallur series "
        "(years %s). Rows tagged source='merged'.",
        len(kk_rows), year_range,
    )

    combined = pd.concat([df, kk_rows], ignore_index=True)
    # Remove duplicates: prefer non-merged rows for the same (year, season, crop)
    combined = combined.sort_values("source").drop_duplicates(
        subset=["year", "season", "crop", "district"], keep="first"
    )
    return combined.reset_index(drop=True)


def _drop_sparse_crops(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Drop crops with fewer than min_data_years usable yield rows.

    Each unique (year, season) pair per crop counts as one "year".

    Parameters
    ----------
    df:  Cleaned yield DataFrame (post invalid-row removal).
    cfg: Configuration dictionary.

    Returns
    -------
    pd.DataFrame with sparse crops removed.
    """
    min_years = cfg["crops"]["min_data_years"]
    keep_crops: list[str] = []

    for crop, group in df.groupby("crop"):
        usable_years = group[["year", "season"]].drop_duplicates().shape[0]
        if usable_years >= min_years:
            keep_crops.append(str(crop))
        else:
            val_logger.warning(
                "Dropping crop '%s': only %d usable (year, season) records "
                "after cleaning — minimum required is %d.",
                crop, usable_years, min_years,
            )

    return df[df["crop"].isin(keep_crops)].copy()


# ---------------------------------------------------------------------------
# prices.csv
# ---------------------------------------------------------------------------

def load_prices(cfg: dict[str, Any]) -> pd.DataFrame:
    """Load and clean ``prices.csv``.

    Adds a ``price_inr_kg`` column (price_inr_quintal / 100).

    Parameters
    ----------
    cfg: Parsed configuration dictionary.

    Returns
    -------
    pd.DataFrame
        Columns: year, month, crop, market, price_inr_quintal,
        price_inr_kg, source.
    """
    raw_dir = Path(cfg["data"]["raw_dir"])
    df      = _read_csv(raw_dir / "prices.csv")

    df["year"]              = df["year"].astype(int)
    df["month"]             = df["month"].astype(int)
    df["price_inr_quintal"] = pd.to_numeric(df["price_inr_quintal"], errors="coerce")
    df["crop"]              = df["crop"].str.strip().str.lower()
    df["market"]            = df["market"].str.strip()

    # Drop non-positive prices
    invalid = df["price_inr_quintal"] <= 0
    if invalid.any():
        val_logger.warning(
            "Dropped %d price rows with non-positive price_inr_quintal.",
            invalid.sum(),
        )
    df = df[~invalid].copy()

    df["price_inr_kg"] = df["price_inr_quintal"] / 100.0

    logger.info("prices.csv cleaned: %d rows.", len(df))
    return df


# ---------------------------------------------------------------------------
# crop_water.csv
# ---------------------------------------------------------------------------

def load_crop_water(cfg: dict[str, Any]) -> pd.DataFrame:
    """Load and clean ``crop_water.csv``.

    Parameters
    ----------
    cfg: Parsed configuration dictionary.

    Returns
    -------
    pd.DataFrame
        Columns: crop, season, water_req_mm, growing_days, kc_mid, source.
    """
    raw_dir = Path(cfg["data"]["raw_dir"])
    df      = _read_csv(raw_dir / "crop_water.csv")

    df["water_req_mm"] = pd.to_numeric(df["water_req_mm"], errors="coerce")
    df["growing_days"] = pd.to_numeric(df["growing_days"], errors="coerce")
    df["kc_mid"]       = pd.to_numeric(df["kc_mid"],       errors="coerce")
    df["crop"]         = df["crop"].str.strip().str.lower()
    df["season"]       = df["season"].str.strip().str.lower()

    invalid = df["water_req_mm"] <= 0
    if invalid.any():
        val_logger.warning(
            "Dropped %d crop_water rows with non-positive water_req_mm.",
            invalid.sum(),
        )
    df = df[~invalid].copy()

    logger.info("crop_water.csv cleaned: %d rows.", len(df))
    return df


# ---------------------------------------------------------------------------
# cost.csv
# ---------------------------------------------------------------------------

def load_cost(cfg: dict[str, Any]) -> pd.DataFrame:
    """Load and clean ``cost.csv``.

    Forward-fills missing (year, crop, season) combinations using the most
    recent available year's cost, with a WARNING logged.

    Parameters
    ----------
    cfg: Parsed configuration dictionary.

    Returns
    -------
    pd.DataFrame
        Columns: year, season, crop, cost_inr_ha, source.
    """
    raw_dir = Path(cfg["data"]["raw_dir"])
    df      = _read_csv(raw_dir / "cost.csv")

    df["year"]        = df["year"].astype(int)
    df["cost_inr_ha"] = pd.to_numeric(df["cost_inr_ha"], errors="coerce")
    df["crop"]        = df["crop"].str.strip().str.lower()
    df["season"]      = df["season"].str.strip().str.lower()

    invalid = df["cost_inr_ha"] <= 0
    if invalid.any():
        val_logger.warning(
            "Dropped %d cost rows with non-positive cost_inr_ha.",
            invalid.sum(),
        )
    df = df[~invalid].copy()

    # Forward-fill missing years per (crop, season)
    crops   = df["crop"].unique()
    seasons = df["season"].unique()
    years   = range(
        cfg["study_period"]["start_year"],
        cfg["study_period"]["end_year"] + 1,
    )
    full_index = pd.MultiIndex.from_product(
        [years, crops, seasons], names=["year", "crop", "season"]
    )
    df_full = (
        df.set_index(["year", "crop", "season"])
          .reindex(full_index)
          .groupby(level=["crop", "season"])
          .ffill()
          .reset_index()
    )
    filled_count = df_full["cost_inr_ha"].isna().sum()
    if filled_count > 0:
        val_logger.warning(
            "%d cost rows could not be filled (no prior data); "
            "they remain NaN.",
            filled_count,
        )
    # Mark forward-filled rows
    original_keys = set(
        zip(df["year"], df["crop"], df["season"])
    )
    df_full["source"] = df_full.apply(
        lambda r: "filled"
        if (r["year"], r["crop"], r["season"]) not in original_keys
        else r.get("source", ""),
        axis=1,
    )

    logger.info("cost.csv cleaned: %d rows.", len(df_full))
    return df_full.dropna(subset=["cost_inr_ha"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Convenience loader for all four files at once
# ---------------------------------------------------------------------------

def load_all_tabular(
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and clean all four tabular raw files.

    Parameters
    ----------
    cfg: Parsed configuration dictionary.

    Returns
    -------
    tuple of (yield_df, prices_df, crop_water_df, cost_df)
    """
    yield_df      = load_yield(cfg)
    prices_df     = load_prices(cfg)
    crop_water_df = load_crop_water(cfg)
    cost_df       = load_cost(cfg)
    return yield_df, prices_df, crop_water_df, cost_df


if __name__ == "__main__":
    cfg = load_config()
    y, p, cw, c = load_all_tabular(cfg)
    print("Yield shape:     ", y.shape)
    print("Prices shape:    ", p.shape)
    print("Crop water shape:", cw.shape)
    print("Cost shape:      ", c.shape)
