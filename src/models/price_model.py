"""
price_model.py
--------------
Trains three quantile GradientBoosting regressors (P10, P50, P90) to
forecast seasonal crop prices (₹/kg).

Design
------
* Features: year (int), crop (ordinal-encoded), season (ordinal-encoded).
* Three ``GradientBoostingRegressor(loss='quantile')`` models for α ∈ {0.10,
  0.50, 0.90} — giving a P10 lower bound, P50 point estimate, P90 upper bound.
* All three models are bundled into a single pickle with metadata indicating
  whether price data was synthetic.
* When ``use_synthetic.prices: true`` the metadata flag ``synthetic=True`` is
  set; the dashboard uses this to surface the disclaimer note.

Usage
-----
    python src/models/price_model.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

_SEASON_MONTHS: dict[str, list[int]] = {
    "kharif": [6, 7, 8, 9],
    "rabi":   [10, 11, 12, 1],
    "summer": [2, 3, 4, 5],
}


class PriceInterval(NamedTuple):
    """Container for a three-quantile price prediction."""
    p10: float
    p50: float
    p90: float


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def _month_to_season(month: int) -> str | None:
    for s, months in _SEASON_MONTHS.items():
        if month in months:
            return s
    return None


def _prepare_price_features(
    prices_df: pd.DataFrame,
) -> tuple[pd.DataFrame, OrdinalEncoder, OrdinalEncoder]:
    """Aggregate monthly prices to seasonal averages and encode categoricals.

    Parameters
    ----------
    prices_df: Cleaned monthly prices DataFrame with price_inr_kg column.

    Returns
    -------
    tuple of (feature DataFrame with target, crop encoder, season encoder)
    """
    df = prices_df.copy()
    df["season"] = df["month"].apply(_month_to_season)
    df = df.dropna(subset=["season"])

    agg = (
        df.groupby(["year", "crop", "season"])
        .agg(price_inr_kg=("price_inr_kg", "mean"))
        .reset_index()
    )

    crop_enc   = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    season_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    agg["crop_enc"]   = crop_enc.fit_transform(agg[["crop"]]).ravel()
    agg["season_enc"] = season_enc.fit_transform(agg[["season"]]).ravel()

    return agg, crop_enc, season_enc


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_price_model(
    cfg: dict[str, Any],
    features_path: Path | None = None,
) -> dict[str, Any]:
    """Train P10/P50/P90 quantile price models and persist them.

    Parameters
    ----------
    cfg:           Parsed configuration dictionary.
    features_path: Path to features.parquet; defaults to config value.

    Returns
    -------
    dict with keys:
        models      : {0.10: estimator, 0.50: estimator, 0.90: estimator}
        crop_enc    : fitted OrdinalEncoder for crops
        season_enc  : fitted OrdinalEncoder for seasons
        synthetic   : bool — True when prices data is synthetic
        metrics     : dict with train-set RMSE for P50
    """
    if features_path is None:
        features_path = Path(cfg["data"]["processed_dir"]) / "features.parquet"

    df_feat   = pd.read_parquet(features_path)
    prices_df = df_feat[
        ["year", "crop", "price_inr_quintal", "price_inr_kg"]
    ].dropna().copy()
    # Recover month-level data isn't available after merge, so we use seasonal
    # averages that are already in features.parquet (one row per year/crop/season)
    prices_df_seasonal = df_feat[
        ["year", "crop", "season", "price_inr_kg"]
    ].dropna().copy()

    # Build feature matrix directly from seasonal prices in features
    crop_enc   = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    season_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    prices_df_seasonal = prices_df_seasonal.copy()
    prices_df_seasonal["crop_enc"]   = crop_enc.fit_transform(
        prices_df_seasonal[["crop"]]
    ).ravel()
    prices_df_seasonal["season_enc"] = season_enc.fit_transform(
        prices_df_seasonal[["season"]]
    ).ravel()

    X = prices_df_seasonal[["year", "crop_enc", "season_enc"]].to_numpy()
    y = prices_df_seasonal["price_inr_kg"].to_numpy()

    quantiles = cfg["models"]["price"]["quantiles"]  # [0.10, 0.50, 0.90]
    seed      = cfg["models"]["random_seed"]
    n_est     = cfg["models"]["price"]["n_estimators"]
    lr        = cfg["models"]["price"]["learning_rate"]

    models: dict[float, GradientBoostingRegressor] = {}
    for q in quantiles:
        gbr = GradientBoostingRegressor(
            loss="quantile",
            alpha=q,
            n_estimators=n_est,
            learning_rate=lr,
            random_state=seed,
        )
        gbr.fit(X, y)
        models[q] = gbr
        logger.info("Trained price quantile model q=%.2f", q)

    # Quick train-set RMSE on P50
    p50_pred  = models[0.50].predict(X)
    train_rmse = float(np.sqrt(np.mean((y - p50_pred) ** 2)))
    logger.info("Price model P50 train RMSE: %.4f ₹/kg", train_rmse)

    # Synthetic flag
    is_synthetic = cfg["data"]["use_synthetic"].get("prices", False)

    bundle = {
        "models":     models,
        "crop_enc":   crop_enc,
        "season_enc": season_enc,
        "synthetic":  is_synthetic,
        "metrics":    {"p50_train_rmse": train_rmse},
        "quantiles":  quantiles,
    }

    # Save
    models_dir = Path(cfg["outputs"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "price_model.pkl"
    with model_path.open("wb") as fh:
        pickle.dump(bundle, fh)
    logger.info("Price model saved to %s", model_path)

    return bundle


# ---------------------------------------------------------------------------
# Backtest / walk-forward helper — trains on ALL rows, no eval
# ---------------------------------------------------------------------------

def fit_only_price(
    df: pd.DataFrame,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Fit P10/P50/P90 price models on every row in *df* without any split.

    Use this in walk-forward backtests.  No metrics are computed and no
    files are written.

    Parameters
    ----------
    df:  Feature DataFrame already filtered to ``year < test_year``.
         Must contain columns year, crop, season, price_inr_kg.
    cfg: Parsed configuration dictionary.

    Returns
    -------
    dict with the same schema as :func:`train_price_model` (minus file I/O).

    Raises
    ------
    ValueError
        If *df* has fewer than 2 usable price rows after dropping NaNs.
    """
    prices_df_seasonal = df[["year", "crop", "season", "price_inr_kg"]].dropna().copy()
    n_rows = len(prices_df_seasonal)
    logger.info("fit_only_price: training price models on %d rows.", n_rows)

    if n_rows < 2:
        raise ValueError(
            f"fit_only_price requires at least 2 rows, got {n_rows}."
        )

    crop_enc   = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    season_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    prices_df_seasonal["crop_enc"]   = crop_enc.fit_transform(
        prices_df_seasonal[["crop"]]
    ).ravel()
    prices_df_seasonal["season_enc"] = season_enc.fit_transform(
        prices_df_seasonal[["season"]]
    ).ravel()

    X = prices_df_seasonal[["year", "crop_enc", "season_enc"]].to_numpy()
    y = prices_df_seasonal["price_inr_kg"].to_numpy()

    quantiles = cfg["models"]["price"]["quantiles"]
    seed      = cfg["models"]["random_seed"]
    n_est     = cfg["models"]["price"]["n_estimators"]
    lr        = cfg["models"]["price"]["learning_rate"]

    models: dict[float, GradientBoostingRegressor] = {}
    for q in quantiles:
        gbr = GradientBoostingRegressor(
            loss="quantile",
            alpha=q,
            n_estimators=n_est,
            learning_rate=lr,
            random_state=seed,
        )
        gbr.fit(X, y)
        models[q] = gbr

    is_synthetic = cfg["data"]["use_synthetic"].get("prices", False)
    return {
        "models":     models,
        "crop_enc":   crop_enc,
        "season_enc": season_enc,
        "synthetic":  is_synthetic,
        "metrics":    {},
        "quantiles":  quantiles,
    }


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def load_price_model(cfg: dict[str, Any]) -> dict[str, Any]:
    """Load the persisted price model bundle.

    Parameters
    ----------
    cfg: Parsed configuration dictionary.

    Returns
    -------
    Bundle dict with keys models, crop_enc, season_enc, synthetic, metrics.
    """
    model_path = Path(cfg["outputs"]["models_dir"]) / "price_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Price model not found at {model_path}. Run train_price_model() first."
        )
    with model_path.open("rb") as fh:
        bundle = pickle.load(fh)
    logger.info("Loaded price model (synthetic=%s)", bundle["synthetic"])
    return bundle


def predict_price(
    bundle: dict[str, Any],
    crop: str,
    season: str,
    year: int,
) -> PriceInterval:
    """Predict P10/P50/P90 price interval for a crop-season-year scenario.

    Parameters
    ----------
    bundle: Price model bundle from :func:`load_price_model`.
    crop:   Crop name.
    season: Season name (kharif | rabi | summer).
    year:   Year integer.

    Returns
    -------
    PriceInterval(p10, p50, p90) in ₹/kg.
    """
    crop_enc_val   = bundle["crop_enc"].transform([[crop]])[0][0]
    season_enc_val = bundle["season_enc"].transform([[season]])[0][0]
    X = np.array([[year, crop_enc_val, season_enc_val]])

    preds = {q: float(m.predict(X)[0]) for q, m in bundle["models"].items()}
    return PriceInterval(
        p10=preds[0.10],
        p50=preds[0.50],
        p90=preds[0.90],
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg    = load_config()
    bundle = train_price_model(cfg)
    print(f"Synthetic flag: {bundle['synthetic']}")
    print(f"P50 train RMSE: {bundle['metrics']['p50_train_rmse']:.4f} ₹/kg")

    interval = predict_price(bundle, "paddy", "kharif", 2023)
    print(f"Paddy kharif 2023 — P10: {interval.p10:.2f}  P50: {interval.p50:.2f}  P90: {interval.p90:.2f} ₹/kg")
