"""
build_features.py
-----------------
Merges cleaned weather aggregates, yield, prices, crop water requirements,
and cost into a single feature table and saves it to
``data/processed/features.parquet``.

The merge key is (year, crop, season).  Seasonal price is derived by
averaging monthly prices over the season months.

Run directly to (re-)build the feature table:
    python src/data/build_features.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.clean_weather import clean_weather
from src.data.clean_tabular import load_all_tabular
from src.data.validate import validate_all
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _seasonal_prices(
    prices_df: pd.DataFrame,
    season_months: dict[str, list[int]],
) -> pd.DataFrame:
    """Aggregate monthly prices to seasonal averages per (year, crop, season).

    Parameters
    ----------
    prices_df:     Cleaned monthly prices DataFrame.
    season_months: Dict mapping season name → list of month integers.

    Returns
    -------
    pd.DataFrame with columns: year, crop, season, price_inr_kg (median),
    price_inr_quintal.
    """
    # Tag each price row with a season
    def _month_to_season(month: int) -> str | None:
        for season, months in season_months.items():
            if month in months:
                return season
        return None

    prices_df = prices_df.copy()
    prices_df["season"] = prices_df["month"].apply(_month_to_season)
    prices_df = prices_df.dropna(subset=["season"])

    agg = (
        prices_df.groupby(["year", "crop", "season"])
        .agg(
            price_inr_kg=("price_inr_kg", "mean"),
            price_inr_quintal=("price_inr_quintal", "mean"),
        )
        .reset_index()
    )
    return agg


def build_features(cfg: dict[str, Any]) -> pd.DataFrame:
    """Build and save the merged feature table.

    Parameters
    ----------
    cfg: Parsed configuration dictionary.

    Returns
    -------
    pd.DataFrame  Full feature table indexed by (year, crop, season).
    """
    processed_dir = Path(cfg["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / "features.parquet"

    # ---- Load all cleaned sources ----------------------------------------
    logger.info("Loading and cleaning data sources …")
    weather_agg = clean_weather(cfg)
    yield_df, prices_df, crop_water_df, cost_df = load_all_tabular(cfg)

    # ---- Validate -----------------------------------------------------------
    # Pass raw daily weather to validator; use aggregate for merge
    from src.data.clean_weather import load_raw_weather, _fill_missing  # noqa: PLC0415
    raw_dir  = Path(cfg["data"]["raw_dir"])
    raw_wx   = _fill_missing(load_raw_weather(raw_dir))
    validate_all(raw_wx, yield_df, prices_df, crop_water_df, cost_df, cfg)

    # ---- Aggregate prices to seasonal level ---------------------------------
    season_months: dict[str, list[int]] = cfg["season_months"]
    prices_seasonal = _seasonal_prices(prices_df, season_months)

    # ---- Merge --------------------------------------------------------------
    # Start from yield (the core table)
    df = yield_df.copy()

    # Weather
    df = df.merge(weather_agg, on=["year", "season"], how="left")

    # Seasonal prices
    df = df.merge(prices_seasonal, on=["year", "crop", "season"], how="left")

    # Crop water requirements (crop × season, no year dimension)
    df = df.merge(
        crop_water_df[["crop", "season", "water_req_mm", "growing_days", "kc_mid"]],
        on=["crop", "season"],
        how="left",
    )

    # Cost (year × crop × season)
    df = df.merge(
        cost_df[["year", "crop", "season", "cost_inr_ha"]],
        on=["year", "crop", "season"],
        how="left",
    )

    # ---- Report merge quality -----------------------------------------------
    for col in ["seasonal_rainfall_mm", "price_inr_kg", "water_req_mm", "cost_inr_ha"]:
        null_count = df[col].isna().sum()
        if null_count > 0:
            logger.warning(
                "Feature '%s' has %d NaN(s) after merge.", col, null_count
            )

    logger.info(
        "Feature table built: %d rows × %d columns.", df.shape[0], df.shape[1]
    )

    # ---- Save ---------------------------------------------------------------
    df.to_parquet(out_path, index=False, engine="pyarrow")
    logger.info("Saved features to %s", out_path)
    return df


if __name__ == "__main__":
    cfg = load_config()
    result = build_features(cfg)
    print(result.shape)
    print(result.dtypes)
    print(result.head())
