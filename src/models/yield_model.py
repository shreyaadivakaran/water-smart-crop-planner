"""
yield_model.py
--------------
Trains RandomForest and XGBoost regressors to predict crop yield (kg/ha)
from seasonal weather features and crop/year inputs.

Design decisions
----------------
* Time-aware split: train on years ≤ config.models.train_test_split_year,
  test on years > split year.  No data leakage.
* Both models are trained with the fixed random seed from config.
* The model with lower RMSE on the test set is persisted to models/.
* Evaluation metrics (MAE, RMSE, R²) are saved to outputs/model_metrics.csv.
* An actual-vs-predicted scatter plot is saved to outputs/plots/.

Features used
-------------
seasonal_rainfall_mm, mean_tmax, mean_tmin, heat_stress_days,
dry_spell_days, crop (ordinal-encoded), year
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

_FEATURE_COLS = [
    "seasonal_rainfall_mm",
    "mean_tmax",
    "mean_tmin",
    "heat_stress_days",
    "dry_spell_days",
    "year",
]
_TARGET_COL   = "yield_kg_ha"
_CROP_COL     = "crop"


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, OrdinalEncoder]:
    """Encode the crop column and return the full feature matrix.

    Parameters
    ----------
    df: Feature table from ``data/processed/features.parquet``.

    Returns
    -------
    tuple of (X DataFrame, fitted OrdinalEncoder)
    """
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    crop_encoded = enc.fit_transform(df[[_CROP_COL]])
    X = df[_FEATURE_COLS].copy()
    X.insert(0, "crop_encoded", crop_encoded.ravel())
    return X, enc


def time_split(
    df: pd.DataFrame,
    split_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into train / test sets by year.

    Parameters
    ----------
    df:         Full feature DataFrame.
    split_year: Years ≤ this go to train; years > this go to test.

    Returns
    -------
    tuple of (train_df, test_df)
    """
    train = df[df["year"] <= split_year].copy()
    test  = df[df["year"] >  split_year].copy()
    logger.info(
        "Time split: train %d rows (≤ %d), test %d rows (> %d).",
        len(train), split_year, len(test), split_year,
    )
    return train, test


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
) -> dict[str, Any]:
    """Compute MAE, RMSE, R² and return as a dict.

    Parameters
    ----------
    y_true:     Ground-truth values.
    y_pred:     Model predictions.
    model_name: Label for logging.

    Returns
    -------
    dict with keys model, MAE, RMSE, R2.
    """
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    logger.info(
        "[%s] MAE=%.1f  RMSE=%.1f  R²=%.3f", model_name, mae, rmse, r2
    )
    return {"model": model_name, "MAE": mae, "RMSE": rmse, "R2": r2}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_yield_models(
    cfg: dict[str, Any],
    features_path: Path | None = None,
) -> dict[str, Any]:
    """Train RF and XGBoost yield models, evaluate, persist the best one.

    Parameters
    ----------
    cfg:           Parsed configuration dictionary.
    features_path: Path to features.parquet; defaults to config value.

    Returns
    -------
    dict with keys:
        best_model  : fitted sklearn/xgb estimator
        encoder     : fitted OrdinalEncoder
        metrics     : list of metric dicts
        best_name   : str, "RandomForest" or "XGBoost"
        train_df    : training split
        test_df     : test split
        X_test      : feature matrix for test set
        y_test      : true labels for test set
        y_pred_best : predictions from best model on test set
    """
    if features_path is None:
        features_path = Path(cfg["data"]["processed_dir"]) / "features.parquet"

    df = pd.read_parquet(features_path)
    df = df.dropna(subset=_FEATURE_COLS + [_TARGET_COL, _CROP_COL])

    split_year = cfg["models"]["train_test_split_year"]
    seed       = cfg["models"]["random_seed"]

    train_df, test_df = time_split(df, split_year)

    # Encode on full data so categories are consistent
    X_all, encoder = prepare_features(df)
    y_all = df[_TARGET_COL].to_numpy()

    train_idx = df["year"] <= split_year
    test_idx  = df["year"] >  split_year

    X_train, y_train = X_all[train_idx].to_numpy(), y_all[train_idx]
    X_test,  y_test  = X_all[test_idx].to_numpy(),  y_all[test_idx]

    ym_cfg = cfg["models"]["yield"]

    # ---- Random Forest -------------------------------------------------------
    rf = RandomForestRegressor(
        n_estimators=ym_cfg["rf_n_estimators"],
        max_depth=ym_cfg["rf_max_depth"],
        random_state=seed,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_pred    = rf.predict(X_test)
    rf_metrics = _compute_metrics(y_test, rf_pred, "RandomForest")

    # ---- XGBoost -------------------------------------------------------------
    xgb = XGBRegressor(
        n_estimators=ym_cfg["xgb_n_estimators"],
        max_depth=ym_cfg["xgb_max_depth"],
        learning_rate=ym_cfg["xgb_learning_rate"],
        random_state=seed,
        verbosity=0,
    )
    xgb.fit(X_train, y_train)
    xgb_pred    = xgb.predict(X_test)
    xgb_metrics = _compute_metrics(y_test, xgb_pred, "XGBoost")

    metrics = [rf_metrics, xgb_metrics]

    # ---- Select best ---------------------------------------------------------
    if xgb_metrics["RMSE"] <= rf_metrics["RMSE"]:
        best_model, best_name, y_pred_best = xgb, "XGBoost", xgb_pred
    else:
        best_model, best_name, y_pred_best = rf,  "RandomForest", rf_pred
    logger.info("Best yield model: %s (RMSE=%.1f)", best_name,
                min(rf_metrics["RMSE"], xgb_metrics["RMSE"]))

    # ---- Save metrics --------------------------------------------------------
    metrics_path = Path(cfg["outputs"]["metrics_csv"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(metrics)
    metrics_df["split_year"] = split_year
    if metrics_path.exists():
        existing = pd.read_csv(metrics_path)
        # Remove old yield rows before appending
        existing = existing[~existing["model"].isin(["RandomForest", "XGBoost"])]
        metrics_df = pd.concat([existing, metrics_df], ignore_index=True)
    metrics_df.to_csv(metrics_path, index=False)
    logger.info("Metrics saved to %s", metrics_path)

    # ---- Persist best model --------------------------------------------------
    models_dir = Path(cfg["outputs"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "yield_model.pkl"
    with model_path.open("wb") as fh:
        pickle.dump({"model": best_model, "encoder": encoder, "name": best_name}, fh)
    logger.info("Yield model saved to %s", model_path)

    return {
        "best_model":   best_model,
        "encoder":      encoder,
        "metrics":      metrics,
        "best_name":    best_name,
        "train_df":     train_df,
        "test_df":      test_df,
        "X_test":       pd.DataFrame(X_all[test_idx], columns=X_all.columns),
        "y_test":       y_test,
        "y_pred_best":  y_pred_best,
    }


# ---------------------------------------------------------------------------
# Backtest / walk-forward helper — trains on ALL rows, no split/evaluate
# ---------------------------------------------------------------------------

def fit_only(
    df: pd.DataFrame,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Fit the best yield model on every row in *df* without any train/test split.

    Use this in walk-forward backtests where the caller has already sliced
    the data to ``year < test_year``.  No metrics are computed, no files are
    written.

    Parameters
    ----------
    df:  Feature DataFrame (already filtered to the desired training window).
         Must contain all columns in ``_FEATURE_COLS``, ``_TARGET_COL``, and
         ``_CROP_COL``.
    cfg: Parsed configuration dictionary (used for hyperparameters and seed).

    Returns
    -------
    dict with keys model, encoder, name  — same shape as :func:`load_yield_model`.

    Raises
    ------
    ValueError
        If *df* has fewer than 2 usable rows after dropping NaNs.
    """
    df = df.dropna(subset=_FEATURE_COLS + [_TARGET_COL, _CROP_COL])
    n_rows = len(df)
    logger.info("fit_only: training yield model on %d rows.", n_rows)

    if n_rows < 2:
        raise ValueError(
            f"fit_only requires at least 2 rows, got {n_rows}. "
            "Expand the training window."
        )

    X_all, encoder = prepare_features(df)
    y_all = df[_TARGET_COL].to_numpy()
    X_np  = X_all.to_numpy()

    seed   = cfg["models"]["random_seed"]
    ym_cfg = cfg["models"]["yield"]

    # Always use RandomForest in fit_only (faster, no eval needed)
    model = RandomForestRegressor(
        n_estimators=ym_cfg["rf_n_estimators"],
        max_depth=ym_cfg["rf_max_depth"],
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_np, y_all)

    return {"model": model, "encoder": encoder, "name": "RandomForest"}


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

def load_yield_model(cfg: dict[str, Any]) -> dict[str, Any]:
    """Load the persisted yield model bundle.

    Parameters
    ----------
    cfg: Parsed configuration dictionary.

    Returns
    -------
    dict with keys model, encoder, name.
    """
    model_path = Path(cfg["outputs"]["models_dir"]) / "yield_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Yield model not found at {model_path}. Run train_yield_models() first."
        )
    with model_path.open("rb") as fh:
        bundle = pickle.load(fh)
    logger.info("Loaded yield model: %s", bundle["name"])
    return bundle


def predict_yield(
    bundle: dict[str, Any],
    crop: str,
    season: str,
    year: int,
    seasonal_rainfall_mm: float,
    mean_tmax: float,
    mean_tmin: float,
    heat_stress_days: float,
    dry_spell_days: float,
) -> float:
    """Predict yield for a single (crop, season, year, weather) scenario.

    Parameters
    ----------
    bundle:               Model bundle from :func:`load_yield_model`.
    crop:                 Crop name (must match training categories).
    season:               Season name (unused directly — weather features carry it).
    year:                 Year integer.
    seasonal_rainfall_mm: Total seasonal rainfall (mm).
    mean_tmax:            Mean daily max temperature (°C).
    mean_tmin:            Mean daily min temperature (°C).
    heat_stress_days:     Days with T2M_MAX > 35 °C.
    dry_spell_days:       Longest consecutive dry-day run.

    Returns
    -------
    float  Predicted yield in kg/ha.
    """
    enc    = bundle["encoder"]
    model  = bundle["model"]
    crop_enc = enc.transform([[crop]])[0][0]

    X = np.array([[
        crop_enc,
        seasonal_rainfall_mm,
        mean_tmax,
        mean_tmin,
        heat_stress_days,
        dry_spell_days,
        year,
    ]])
    return float(model.predict(X)[0])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg    = load_config()
    result = train_yield_models(cfg)
    print("\nModel comparison:")
    for m in result["metrics"]:
        print(f"  {m['model']:15s}  MAE={m['MAE']:.0f}  RMSE={m['RMSE']:.0f}  R²={m['R2']:.3f}")
    print(f"\nBest: {result['best_name']}")
