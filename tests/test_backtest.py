"""
test_backtest.py
----------------
Unit tests for src/data/backtest.py and the fit_only helpers.

Tests
-----
TestFitOnly
  * fit_only() returns the expected bundle keys
  * fit_only() raises ValueError when given fewer than 2 rows
  * fit_only_price() returns the expected bundle keys
  * fit_only_price() raises ValueError when given fewer than 2 rows
  * predict_yield() works on a bundle returned by fit_only()
  * predict_price() works on a bundle returned by fit_only_price()

TestHelpers
  * _dominant_crop() returns correct crop
  * _dominant_crop() returns None on empty DataFrame
  * _realised_profit() computes yield × price − cost correctly
  * _realised_profit() returns NaN when row is None
  * _realised_water() computes max(0, water_req − rainfall) correctly
  * _realised_water() never returns negative

TestSafeWrite
  * _safe_write() creates the file with header when df is empty
  * _safe_write() writes correct row count for non-empty df
  * _safe_write() leaves no temp files on success

TestRunBacktest (integration, uses synthetic data on disk)
  * Happy-path: run_backtest() produces ≥ 1 result row
  * Result has all required columns
  * All-failures case: when features_df is empty, result CSV has header only
    and run_backtest exits with SystemExit(1)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config_loader import load_config


# ---------------------------------------------------------------------------
# Minimal synthetic feature DataFrame factory
# ---------------------------------------------------------------------------

def _make_features(
    years: list[int],
    crops: list[str] | None = None,
    seasons: list[str] | None = None,
) -> pd.DataFrame:
    """Create a minimal feature DataFrame that satisfies all model requirements."""
    if crops is None:
        crops = ["paddy", "maize"]
    if seasons is None:
        seasons = ["kharif"]

    rows = []
    rng = np.random.default_rng(42)
    for year in years:
        for season in seasons:
            rain = 850.0 if season == "kharif" else 280.0
            for crop in crops:
                base_yield = {"paddy": 3200.0, "maize": 4500.0}.get(crop, 2000.0)
                base_price = {"paddy": 19.0,   "maize": 17.0}.get(crop, 18.0)
                base_cost  = {"paddy": 38000.0, "maize": 22000.0}.get(crop, 25000.0)
                base_water = {"paddy": 1200.0,  "maize": 500.0}.get(crop, 800.0)
                rows.append({
                    "year":                  year,
                    "season":                season,
                    "crop":                  crop,
                    "district":              "Tiruvallur",
                    "yield_kg_ha":           base_yield + rng.normal(0, base_yield * 0.1),
                    "area_ha":               rng.uniform(1000, 5000),
                    "price_inr_kg":          base_price  + rng.normal(0, 1),
                    "price_inr_quintal":     (base_price + rng.normal(0, 1)) * 100,
                    "seasonal_rainfall_mm":  rain        + rng.normal(0, 80),
                    "mean_tmax":             32.0 + rng.normal(0, 1),
                    "mean_tmin":             23.0 + rng.normal(0, 1),
                    "heat_stress_days":      max(0, rng.normal(10, 3)),
                    "dry_spell_days":        max(0, rng.normal(15, 5)),
                    "mean_rh2m":             70.0,
                    "mean_ws2m":             2.5,
                    "mean_solar_rad":        18.0,
                    "n_days":                120,
                    "water_req_mm":          base_water,
                    "growing_days":          120,
                    "kc_mid":                1.2,
                    "cost_inr_ha":           base_cost + year * 500,
                    "source":                "synthetic",
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# TestFitOnly
# ---------------------------------------------------------------------------

class TestFitOnly:
    """Tests for yield_model.fit_only() and price_model.fit_only_price()."""

    @pytest.fixture(scope="class")
    def cfg(self):
        return load_config()

    @pytest.fixture(scope="class")
    def train_df(self):
        return _make_features(years=list(range(2010, 2021)), crops=["paddy", "maize"])

    def test_fit_only_returns_bundle_keys(self, cfg, train_df):
        from src.models.yield_model import fit_only
        bundle = fit_only(train_df, cfg)
        assert set(bundle.keys()) == {"model", "encoder", "name"}

    def test_fit_only_name_is_randomforest(self, cfg, train_df):
        from src.models.yield_model import fit_only
        bundle = fit_only(train_df, cfg)
        assert bundle["name"] == "RandomForest"

    def test_fit_only_raises_on_too_few_rows(self, cfg):
        from src.models.yield_model import fit_only
        tiny = _make_features(years=[2010], crops=["paddy"])[:1]
        with pytest.raises(ValueError, match="at least 2"):
            fit_only(tiny, cfg)

    def test_fit_only_price_returns_bundle_keys(self, cfg, train_df):
        from src.models.price_model import fit_only_price
        bundle = fit_only_price(train_df, cfg)
        assert {"models", "crop_enc", "season_enc", "synthetic", "quantiles"}.issubset(
            bundle.keys()
        )

    def test_fit_only_price_has_three_quantile_models(self, cfg, train_df):
        from src.models.price_model import fit_only_price
        bundle = fit_only_price(train_df, cfg)
        assert set(bundle["models"].keys()) == {0.10, 0.50, 0.90}

    def test_fit_only_price_raises_on_too_few_rows(self, cfg):
        from src.models.price_model import fit_only_price
        tiny = _make_features(years=[2010], crops=["paddy"])[:1]
        with pytest.raises(ValueError, match="at least 2"):
            fit_only_price(tiny, cfg)

    def test_predict_yield_after_fit_only(self, cfg, train_df):
        from src.models.yield_model import fit_only, predict_yield
        bundle = fit_only(train_df, cfg)
        result = predict_yield(
            bundle=bundle,
            crop="paddy",
            season="kharif",
            year=2021,
            seasonal_rainfall_mm=850.0,
            mean_tmax=32.0,
            mean_tmin=23.0,
            heat_stress_days=10.0,
            dry_spell_days=15.0,
        )
        assert isinstance(result, float)
        assert result > 0

    def test_predict_price_after_fit_only_price(self, cfg, train_df):
        from src.models.price_model import fit_only_price, predict_price
        bundle = fit_only_price(train_df, cfg)
        interval = predict_price(bundle, "paddy", "kharif", 2021)
        assert interval.p10 <= interval.p50 <= interval.p90
        assert interval.p50 > 0


# ---------------------------------------------------------------------------
# TestHelpers
# ---------------------------------------------------------------------------

class TestHelpers:
    """Tests for the pure helper functions in backtest.py."""

    def test_dominant_crop_returns_correct_crop(self):
        from src.data.backtest import _dominant_crop
        yield_df = pd.DataFrame([
            {"year": 2021, "season": "kharif", "crop": "paddy",  "area_ha": 5000},
            {"year": 2021, "season": "kharif", "crop": "maize",  "area_ha": 2000},
            {"year": 2021, "season": "kharif", "crop": "ragi",   "area_ha": 1000},
        ])
        assert _dominant_crop(yield_df, 2021, "kharif") == "paddy"

    def test_dominant_crop_returns_none_on_empty(self):
        from src.data.backtest import _dominant_crop
        assert _dominant_crop(pd.DataFrame(), 2021, "kharif") is None

    def test_dominant_crop_returns_none_missing_year(self):
        from src.data.backtest import _dominant_crop
        yield_df = pd.DataFrame([
            {"year": 2020, "season": "kharif", "crop": "paddy", "area_ha": 5000},
        ])
        assert _dominant_crop(yield_df, 2021, "kharif") is None

    def test_realised_profit_known_value(self):
        from src.data.backtest import _realised_profit
        row = pd.Series({
            "yield_kg_ha":  3000.0,
            "price_inr_kg": 20.0,
            "cost_inr_ha":  38000.0,
        })
        # 3000 × 20 − 38000 = 22000
        assert _realised_profit(row) == pytest.approx(22000.0)

    def test_realised_profit_returns_nan_for_none(self):
        from src.data.backtest import _realised_profit
        assert np.isnan(_realised_profit(None))

    def test_realised_water_known_value(self):
        from src.data.backtest import _realised_water
        row = pd.Series({"water_req_mm": 1200.0, "seasonal_rainfall_mm": 800.0})
        assert _realised_water(row) == pytest.approx(400.0)

    def test_realised_water_never_negative(self):
        from src.data.backtest import _realised_water
        # Surplus rainfall → deficit must be 0
        row = pd.Series({"water_req_mm": 500.0, "seasonal_rainfall_mm": 2000.0})
        assert _realised_water(row) == pytest.approx(0.0)

    def test_realised_water_returns_nan_for_none(self):
        from src.data.backtest import _realised_water
        assert np.isnan(_realised_water(None))


# ---------------------------------------------------------------------------
# TestSafeWrite
# ---------------------------------------------------------------------------

class TestSafeWrite:
    """Tests for the atomic temp→rename CSV write."""

    def test_safe_write_creates_file(self, tmp_path):
        from src.data.backtest import _safe_write, _COLUMNS
        out = tmp_path / "sub" / "bt.csv"
        _safe_write(pd.DataFrame(columns=_COLUMNS), out)
        assert out.exists()

    def test_safe_write_header_only_for_empty_df(self, tmp_path):
        from src.data.backtest import _safe_write, _COLUMNS
        out = tmp_path / "bt_empty.csv"
        _safe_write(pd.DataFrame(columns=_COLUMNS), out)
        df = pd.read_csv(out)
        assert list(df.columns) == _COLUMNS
        assert len(df) == 0

    def test_safe_write_correct_row_count(self, tmp_path):
        from src.data.backtest import _safe_write, _COLUMNS
        out = tmp_path / "bt_rows.csv"
        data = pd.DataFrame(
            [{"year": 2021, "season": "kharif",
              **{c: None for c in _COLUMNS if c not in ("year", "season")}}],
            columns=_COLUMNS,
        )
        _safe_write(data, out)
        assert len(pd.read_csv(out)) == 1

    def test_safe_write_no_temp_files_on_success(self, tmp_path):
        from src.data.backtest import _safe_write, _COLUMNS
        out = tmp_path / "bt.csv"
        _safe_write(pd.DataFrame(columns=_COLUMNS), out)
        leftover = list(tmp_path.glob(".tmp_backtest_*"))
        assert leftover == [], f"Temp files left: {leftover}"


# ---------------------------------------------------------------------------
# TestRunBacktest (integration)
# ---------------------------------------------------------------------------

class TestRunBacktest:
    """Integration tests using a small synthetic feature table in memory."""

    @pytest.fixture(scope="class")
    def cfg(self):
        c = load_config()
        # Speed up NSGA-II for testing
        c["optimization"]["population_size"] = 10
        c["optimization"]["n_generations"]   = 10
        # Only 2021 is the test year (split at 2020)
        c["models"]["train_test_split_year"] = 2020
        c["study_period"]["end_year"]        = 2021
        return c

    @pytest.fixture(scope="class")
    def features_df(self):
        # 2010–2021, two crops, one season → 2×12 = 24 rows
        return _make_features(
            years=list(range(2010, 2022)),
            crops=["paddy", "maize"],
            seasons=["kharif"],
        )

    @pytest.fixture(scope="class")
    def yield_df(self, features_df):
        """Derive a minimal yield DataFrame from the synthetic features."""
        return features_df[["year", "season", "crop", "area_ha"]].copy()

    def test_happy_path_produces_result_rows(self, cfg, features_df, yield_df, tmp_path):
        """run_backtest() must produce at least one result row."""
        from src.data.backtest import run_backtest
        import os
        (tmp_path / "outputs").mkdir()
        orig_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            result = run_backtest(cfg, features_df=features_df, yield_df=yield_df)
        finally:
            os.chdir(orig_cwd)

        assert len(result) >= 1, (
            "run_backtest() produced 0 rows — walk-forward training likely failed"
        )

    def test_happy_path_has_all_columns(self, cfg, features_df, yield_df, tmp_path):
        """Result DataFrame must contain every column in _COLUMNS."""
        from src.data.backtest import run_backtest, _COLUMNS
        import os
        (tmp_path / "outputs").mkdir(exist_ok=True)
        orig_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            result = run_backtest(cfg, features_df=features_df, yield_df=yield_df)
        finally:
            os.chdir(orig_cwd)

        for col in _COLUMNS:
            assert col in result.columns, f"Missing column: {col}"

    def test_all_failures_writes_header_only_and_exits_1(self, cfg, yield_df, tmp_path):
        """An empty features_df must trigger SystemExit(1) and a header-only CSV."""
        from src.data.backtest import run_backtest, _COLUMNS
        import os
        (tmp_path / "outputs").mkdir(exist_ok=True)
        orig_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            with pytest.raises(SystemExit) as exc_info:
                run_backtest(
                    cfg,
                    features_df=pd.DataFrame(columns=_COLUMNS),
                    yield_df=yield_df,
                )
            assert exc_info.value.code == 1
            csv = tmp_path / "outputs" / "backtest_results.csv"
            assert csv.exists(), "CSV must be written even on all-failures"
            df = pd.read_csv(csv)
            assert list(df.columns) == _COLUMNS
            assert len(df) == 0, "CSV must be header-only on all-failures"
        finally:
            os.chdir(orig_cwd)
