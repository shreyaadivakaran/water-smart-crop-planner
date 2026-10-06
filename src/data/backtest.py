"""
backtest.py
-----------
Walk-forward backtest for the Water-Smart Crop Planner.

Algorithm (per test year Y, per season S)
-----------------------------------------
1. Train yield + price models on all feature rows with year < Y  (fit_only —
   no internal split, no evaluation overhead).
2. Compute risk scores on the same training window.
3. Run the NSGA-II optimiser using Y's *actual* weather features as the
   scenario (rainfall_scenario_factor=1.0, irrigation_mm=100).
4. Pick the top-ranked crop (recommended_crop).
5. Look up the *realised* profit and water use of:
      a. recommended_crop in year Y from the actual feature table
      b. the actual dominant crop (by sown area) in year Y
      c. paddy in year Y (water baseline)
6. Record profit_delta = realised_recommended_profit − realised_actual_profit
   and water_delta = recommended_water_deficit − actual_water_deficit.

Safe-write contract
-------------------
* Results are written to a temp file first, then renamed to the final path,
  so a partial write never corrupts the output.
* If every year/season fails, a header-only CSV is written, an ERROR is
  logged listing every failure, and the process exits with status 1.
* Per-year log lines report: training rows, crops in scope, result.

Usage
-----
    python src/data/backtest.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Canonical output columns (written even when results is empty)
_COLUMNS = [
    "year",
    "season",
    "train_rows",
    "recommended_crop",
    "actual_dominant_crop",
    "hit",
    "realised_rec_profit_inr_ha",
    "realised_act_profit_inr_ha",
    "profit_delta_inr_ha",
    "rec_water_deficit_mm",
    "act_water_deficit_mm",
    "water_delta_mm",
    "paddy_water_deficit_mm",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dominant_crop(yield_df: pd.DataFrame, year: int, season: str) -> str | None:
    """Return the crop with the largest sown area in (year, season).

    Parameters
    ----------
    yield_df: Cleaned yield DataFrame with columns year, season, crop, area_ha.
    year:     Target year.
    season:   Target season.

    Returns
    -------
    Crop name string, or None if no data found.
    """
    if yield_df.empty or "year" not in yield_df.columns:
        return None
    subset = yield_df[
        (yield_df["year"] == year) & (yield_df["season"] == season)
    ]
    if subset.empty or "area_ha" not in subset.columns:
        return None
    return str(subset.loc[subset["area_ha"].idxmax(), "crop"])


def _realised_row(
    features_df: pd.DataFrame,
    crop: str,
    season: str,
    year: int,
) -> pd.Series | None:
    """Fetch the actual feature row for (crop, season, year).

    Parameters
    ----------
    features_df: Full merged feature table.
    crop:        Crop name.
    season:      Season name.
    year:        Year.

    Returns
    -------
    pd.Series or None if the row is absent.
    """
    rows = features_df[
        (features_df["crop"]   == crop) &
        (features_df["season"] == season) &
        (features_df["year"]   == year)
    ]
    return rows.iloc[0] if not rows.empty else None


def _realised_profit(row: pd.Series | None) -> float:
    """Compute realised profit from an actual feature row.

    profit = yield_kg_ha × price_inr_kg − cost_inr_ha

    Returns NaN if row is None or any required field is missing.
    """
    if row is None:
        return np.nan
    try:
        return float(row["yield_kg_ha"] * row["price_inr_kg"] - row["cost_inr_ha"])
    except (KeyError, TypeError):
        return np.nan


def _realised_water(row: pd.Series | None) -> float:
    """Compute realised water deficit from an actual feature row.

    deficit = max(0, water_req_mm − seasonal_rainfall_mm)

    Returns NaN if row is None.
    """
    if row is None:
        return np.nan
    try:
        return float(max(0.0,
            row["water_req_mm"] - row["seasonal_rainfall_mm"]
        ))
    except (KeyError, TypeError):
        return np.nan


def _safe_write(result_df: pd.DataFrame, out_path: Path) -> None:
    """Write result_df to out_path via a temp file → atomic rename.

    Parameters
    ----------
    result_df: DataFrame to write (may be empty but must have correct columns).
    out_path:  Final destination path.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=out_path.parent, prefix=".tmp_backtest_", suffix=".csv"
    )
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as fh:
            result_df.to_csv(fh, index=False)
        Path(tmp_name).replace(out_path)
    except Exception:
        try:
            Path(tmp_name).unlink(missing_ok=True)
        except OSError:
            pass
        raise
    logger.info("Backtest results written to %s (%d rows).", out_path, len(result_df))


# ---------------------------------------------------------------------------
# Main backtest
# ---------------------------------------------------------------------------

def run_backtest(
    cfg: dict[str, Any],
    features_df: pd.DataFrame | None = None,
    yield_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Execute the walk-forward backtest and return the results DataFrame.

    Parameters
    ----------
    cfg:         Parsed configuration dictionary.
    features_df: Pre-loaded feature table; loaded from parquet if None.
    yield_df:    Pre-loaded yield DataFrame (used for dominant-crop lookup);
                 loaded via :func:`load_yield` if None.  Pass an explicit
                 DataFrame in tests to avoid hitting disk.

    Returns
    -------
    pd.DataFrame with columns defined in ``_COLUMNS``.
    Writes ``outputs/backtest_results.csv`` (header-only if no results).
    Exits with status 1 if every year/season failed.
    """
    from src.data.clean_tabular import load_yield                      # noqa: PLC0415
    from src.models.yield_model import fit_only                        # noqa: PLC0415
    from src.models.price_model import fit_only_price                  # noqa: PLC0415
    from src.risk.risk_scorer import compute_risk_scores               # noqa: PLC0415
    from src.optimization.objectives import compute_objectives_batch   # noqa: PLC0415
    from src.optimization.nsga2_optimizer import run_nsga2             # noqa: PLC0415
    from src.optimization.weighted_ranker import rank_crops            # noqa: PLC0415

    split_year = cfg["models"]["train_test_split_year"]
    end_year   = cfg["study_period"]["end_year"]
    test_years = list(range(split_year + 1, end_year + 1))
    seasons    = cfg["seasons"]
    weights    = cfg["weights_default"]
    out_path   = Path("outputs/backtest_results.csv")

    if features_df is None:
        features_df = pd.read_parquet(
            Path(cfg["data"]["processed_dir"]) / "features.parquet"
        )

    if yield_df is None:
        yield_df = load_yield(cfg)

    records:  list[dict[str, Any]] = []
    failures: list[str]            = []   # "YYYY/season: reason"

    for test_year in test_years:
        train_feat = features_df[features_df["year"] < test_year].copy()
        train_rows = len(train_feat)

        logger.info(
            "Backtest year %d — training on %d rows (years %d–%d).",
            test_year,
            train_rows,
            int(train_feat["year"].min()) if train_rows else 0,
            test_year - 1,
        )

        if train_rows < 2:
            reason = f"only {train_rows} training row(s)"
            logger.error("Year %d skipped: %s.", test_year, reason)
            for s in seasons:
                failures.append(f"{test_year}/{s}: {reason}")
            continue

        # ---- Fit models on training window (no internal split) -----------
        try:
            yb = fit_only(train_feat, cfg)
        except Exception as exc:
            reason = f"yield fit_only failed: {exc}"
            logger.error("Year %d skipped: %s", test_year, reason)
            for s in seasons:
                failures.append(f"{test_year}/{s}: {reason}")
            continue

        try:
            pb = fit_only_price(train_feat, cfg)
        except Exception as exc:
            reason = f"price fit_only failed: {exc}"
            logger.error("Year %d skipped: %s", test_year, reason)
            for s in seasons:
                failures.append(f"{test_year}/{s}: {reason}")
            continue

        # ---- Risk scores on training window only -------------------------
        # Write a temp parquet for compute_risk_scores (it reads a file path).
        # Use a proper system temp dir so this works regardless of cwd.
        import tempfile as _tempfile  # noqa: PLC0415
        tmp_fd, tmp_feat_str = _tempfile.mkstemp(suffix=".parquet", prefix="bt_risk_")
        tmp_feat_path = Path(tmp_feat_str)
        os.close(tmp_fd)
        try:
            train_feat.to_parquet(tmp_feat_path, index=False)
            rd = compute_risk_scores(cfg, features_path=tmp_feat_path)
        except Exception as exc:
            reason = f"risk scoring failed: {exc}"
            logger.error("Year %d skipped: %s", test_year, reason)
            for s in seasons:
                failures.append(f"{test_year}/{s}: {reason}")
            tmp_feat_path.unlink(missing_ok=True)
            continue
        finally:
            tmp_feat_path.unlink(missing_ok=True)

        crops = sorted(train_feat["crop"].unique())
        logger.info("Year %d — crops in scope: %s", test_year, crops)

        # ---- Per-season optimisation ------------------------------------
        for season in seasons:
            try:
                # Use year Y's actual weather for the optimisation scenario
                obj_df = compute_objectives_batch(
                    crops=crops,
                    season=season,
                    year=test_year,
                    irrigation_mm=100,
                    rainfall_scenario_factor=1.0,
                    features_df=features_df,   # full table — Y's row is present
                    risk_df=rd,
                    yield_bundle=yb,
                    price_bundle=pb,
                )

                pareto  = run_nsga2(obj_df, cfg)
                ranked  = rank_crops(pareto, weights)
                rec_crop = ranked.iloc[0]["crop"]

                # ---- Realised outcomes from actual year-Y data ----------
                rec_row  = _realised_row(features_df, rec_crop,  season, test_year)
                act_crop = _dominant_crop(yield_df, test_year, season)
                act_row  = _realised_row(features_df, act_crop,  season, test_year) \
                           if act_crop else None
                pad_row  = _realised_row(features_df, "paddy",   season, test_year)

                rec_profit = _realised_profit(rec_row)
                act_profit = _realised_profit(act_row)
                rec_water  = _realised_water(rec_row)
                act_water  = _realised_water(act_row)
                pad_water  = _realised_water(pad_row)

                record = {
                    "year":                        test_year,
                    "season":                      season,
                    "train_rows":                  train_rows,
                    "recommended_crop":            rec_crop,
                    "actual_dominant_crop":        act_crop,
                    "hit":                         rec_crop == act_crop,
                    "realised_rec_profit_inr_ha":  rec_profit,
                    "realised_act_profit_inr_ha":  act_profit,
                    "profit_delta_inr_ha":         rec_profit - act_profit
                                                   if not (np.isnan(rec_profit)
                                                           or np.isnan(act_profit))
                                                   else np.nan,
                    "rec_water_deficit_mm":        rec_water,
                    "act_water_deficit_mm":        act_water,
                    "water_delta_mm":              rec_water - act_water
                                                   if not (np.isnan(rec_water)
                                                           or np.isnan(act_water))
                                                   else np.nan,
                    "paddy_water_deficit_mm":      pad_water,
                }
                records.append(record)

                logger.info(
                    "Year %d / %s → recommend=%s  actual=%s  hit=%s  "
                    "Δprofit=₹%.0f/ha  Δwater=%.0f mm",
                    test_year, season, rec_crop, act_crop,
                    record["hit"],
                    record["profit_delta_inr_ha"]
                    if not np.isnan(record["profit_delta_inr_ha"]) else float("nan"),
                    record["water_delta_mm"]
                    if not np.isnan(record["water_delta_mm"]) else float("nan"),
                )

            except Exception as exc:
                reason = str(exc)
                logger.error(
                    "Year %d / %s failed: %s", test_year, season, reason
                )
                failures.append(f"{test_year}/{season}: {reason}")

    # ---- Build result DataFrame (always has correct columns) -------------
    if records:
        result_df = pd.DataFrame(records, columns=_COLUMNS)
    else:
        result_df = pd.DataFrame(columns=_COLUMNS)

    # ---- Safe write ------------------------------------------------------
    _safe_write(result_df, out_path)

    # ---- Summary ---------------------------------------------------------
    if records:
        hit_rate    = result_df["hit"].mean() * 100
        mean_pdelta = result_df["profit_delta_inr_ha"].mean()
        mean_wdelta = result_df["water_delta_mm"].mean()
        logger.info(
            "Backtest complete: %d rows, hit_rate=%.1f%%, "
            "mean_profit_delta=₹%.0f/ha, mean_water_delta=%.1f mm.",
            len(result_df), hit_rate, mean_pdelta, mean_wdelta,
        )
        print(
            f"\nBacktest summary ({len(result_df)} year/season rows):\n"
            f"  Hit rate:          {hit_rate:.1f}%\n"
            f"  Mean profit delta: ₹{mean_pdelta:,.0f}/ha\n"
            f"  Mean water delta:  {mean_wdelta:.1f} mm"
        )
    else:
        # All years/seasons failed
        logger.error(
            "Backtest produced ZERO results. All %d year/season "
            "combinations failed:\n  %s",
            len(failures),
            "\n  ".join(failures),
        )
        print(
            f"\nERROR: Backtest produced no results. "
            f"{len(failures)} failure(s) logged above.",
            file=sys.stderr,
        )
        sys.exit(1)

    if failures:
        logger.warning(
            "%d year/season combination(s) failed (partial results saved):\n  %s",
            len(failures),
            "\n  ".join(failures),
        )

    return result_df


if __name__ == "__main__":
    cfg = load_config()
    run_backtest(cfg)
