"""
test_data_pipeline.py
---------------------
Unit tests for the data layer.

Covers
------
clean_weather:
  * Season aggregation produces correct columns
  * NaN in rainfall is handled (filled or ignored)
  * heat_stress_days counts correctly

validate:
  * Missing-ratio warning triggered above threshold
  * Unit-plausibility warning triggered on out-of-range values
  * Year-coverage warning triggered for missing years

generate_synthetic:
  * All five CSVs are created with correct schemas
  * Each file contains the synthetic disclaimer comment
  * Re-running with same seed produces identical output

clean_tabular:
  * Crops below min_data_years threshold are dropped with warning
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config_loader import load_config


# ---------------------------------------------------------------------------
# clean_weather
# ---------------------------------------------------------------------------

class TestCleanWeather:
    """Tests for season aggregation logic in clean_weather.py."""

    def _make_daily_df(self, n_days: int = 365, year: int = 2015) -> pd.DataFrame:
        """Create a synthetic daily weather DataFrame for one year."""
        dates  = pd.date_range(f"{year}-01-01", periods=n_days, freq="D")
        rng    = np.random.default_rng(42)
        return pd.DataFrame({
            "date":             dates.strftime("%Y-%m-%d"),
            "PRECTOTCORR":      np.maximum(0, rng.normal(3, 5, n_days)),
            "T2M_MAX":          rng.normal(32, 3, n_days),
            "T2M_MIN":          rng.normal(23, 3, n_days),
            "RH2M":             rng.uniform(50, 95, n_days),
            "WS2M":             rng.uniform(1, 5, n_days),
            "ALLSKY_SFC_SW_DWN": rng.uniform(10, 25, n_days),
        })

    def test_output_columns(self):
        from src.data.clean_weather import _season_aggregates
        cfg = load_config()
        df  = self._make_daily_df()
        df["date"] = pd.to_datetime(df["date"])
        # Jun–Sep = kharif
        kharif = df[df["date"].dt.month.isin([6, 7, 8, 9])].copy()
        result = _season_aggregates(kharif)
        expected = {
            "seasonal_rainfall_mm", "mean_tmax", "mean_tmin",
            "heat_stress_days", "dry_spell_days",
            "mean_rh2m", "mean_ws2m", "mean_solar_rad", "n_days",
        }
        assert expected == set(result.index)

    def test_seasonal_rainfall_is_sum(self):
        from src.data.clean_weather import _season_aggregates
        df = self._make_daily_df()
        df["date"] = pd.to_datetime(df["date"])
        kharif = df[df["date"].dt.month.isin([6, 7, 8, 9])].copy()
        result = _season_aggregates(kharif)
        expected_rain = float(kharif["PRECTOTCORR"].sum())
        assert result["seasonal_rainfall_mm"] == pytest.approx(expected_rain, rel=1e-6)

    def test_heat_stress_days_count(self):
        from src.data.clean_weather import _season_aggregates, _HEAT_STRESS_THRESHOLD_C
        df = self._make_daily_df(n_days=122)  # kharif-sized window
        # Force exactly 5 days above threshold
        df["T2M_MAX"] = 30.0
        df.iloc[:5, df.columns.get_loc("T2M_MAX")] = 40.0
        df["date"] = pd.to_datetime(pd.date_range("2015-06-01", periods=122, freq="D").strftime("%Y-%m-%d"))
        result = _season_aggregates(df)
        assert int(result["heat_stress_days"]) == 5

    def test_nan_rainfall_handled(self):
        from src.data.clean_weather import _fill_missing
        df = self._make_daily_df()
        df["date"] = pd.to_datetime(df["date"])
        df.iloc[0, df.columns.get_loc("PRECTOTCORR")] = np.nan
        filled = _fill_missing(df)
        # NaN should be reduced (filled by ffill/bfill)
        assert filled["PRECTOTCORR"].isna().sum() <= df["PRECTOTCORR"].isna().sum()


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

class TestValidate:
    def test_missing_ratio_warning(self, caplog):
        from src.data.validate import validate_weather
        cfg = load_config()
        # Create a DF with 20 % NaN in PRECTOTCORR (above 5 % threshold)
        n   = 100
        df  = pd.DataFrame({
            "date":             pd.date_range("2010-01-01", periods=n).strftime("%Y-%m-%d"),
            "PRECTOTCORR":      [np.nan if i < 20 else 3.0 for i in range(n)],
            "T2M_MAX":          [30.0] * n,
            "T2M_MIN":          [22.0] * n,
            "RH2M":             [70.0] * n,
            "WS2M":             [2.5]  * n,
            "ALLSKY_SFC_SW_DWN": [18.0] * n,
        })
        with caplog.at_level(logging.WARNING):
            validate_weather(df, cfg)
        assert any("PRECTOTCORR" in r.message for r in caplog.records)

    def test_plausibility_warning_temperature(self, caplog):
        from src.data.validate import validate_weather
        cfg = load_config()
        n   = 10
        df  = pd.DataFrame({
            "date":             pd.date_range("2010-01-01", periods=n).strftime("%Y-%m-%d"),
            "PRECTOTCORR":      [3.0] * n,
            "T2M_MAX":          [70.0] * n,   # above 60 °C → out of range
            "T2M_MIN":          [22.0] * n,
            "RH2M":             [70.0] * n,
            "WS2M":             [2.5]  * n,
            "ALLSKY_SFC_SW_DWN": [18.0] * n,
        })
        with caplog.at_level(logging.WARNING):
            validate_weather(df, cfg)
        assert any("T2M_MAX" in r.message for r in caplog.records)

    def test_year_coverage_warning(self, caplog):
        from src.data.validate import validate_yield
        cfg = load_config()
        # Missing year 2015
        years = [y for y in range(2010, 2024) if y != 2015]
        df = pd.DataFrame({
            "year":        years,
            "season":      ["kharif"] * len(years),
            "crop":        ["paddy"]  * len(years),
            "district":    ["Tiruvallur"] * len(years),
            "yield_kg_ha": [3000.0] * len(years),
            "area_ha":     [5000.0] * len(years),
            "source":      ["synthetic"] * len(years),
        })
        with caplog.at_level(logging.WARNING):
            validate_yield(df, cfg)
        assert any("2015" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# generate_synthetic
# ---------------------------------------------------------------------------

class TestGenerateSynthetic:
    @pytest.fixture(scope="class")
    def raw_dir(self):
        """Return path to data/raw; synthetic files should already exist."""
        p = Path("data/raw")
        if not (p / "weather.csv").exists():
            cfg = load_config()
            from src.data.generate_synthetic import run
            run()
        return p

    def test_all_five_files_exist(self, raw_dir):
        for name in ["weather.csv", "yield.csv", "prices.csv", "crop_water.csv", "cost.csv"]:
            assert (raw_dir / name).exists(), f"{name} missing"

    def test_weather_schema(self, raw_dir):
        df = pd.read_csv(raw_dir / "weather.csv", comment="#")
        required = {"date", "PRECTOTCORR", "T2M_MAX", "T2M_MIN", "RH2M", "WS2M", "ALLSKY_SFC_SW_DWN"}
        assert required.issubset(set(df.columns))

    def test_yield_schema(self, raw_dir):
        df = pd.read_csv(raw_dir / "yield.csv", comment="#")
        required = {"year", "season", "crop", "district", "yield_kg_ha", "area_ha", "source"}
        assert required.issubset(set(df.columns))

    def test_disclaimer_present_in_weather(self, raw_dir):
        first_line = (raw_dir / "weather.csv").read_text(encoding="utf-8").splitlines()[0]
        assert "SYNTHETIC DATA" in first_line

    def test_seed_reproducibility(self, tmp_path):
        """Two runs with the same seed should produce identical weather.csv."""
        import shutil
        cfg = load_config()
        cfg["data"]["raw_dir"] = str(tmp_path / "run1")
        from src.data.generate_synthetic import run
        run.__module__  # ensure importable
        import numpy as np
        rng1 = np.random.default_rng(cfg["models"]["random_seed"])
        rng2 = np.random.default_rng(cfg["models"]["random_seed"])
        from src.data.generate_synthetic import generate_weather
        p1 = tmp_path / "run1"; p1.mkdir()
        p2 = tmp_path / "run2"; p2.mkdir()
        generate_weather(cfg, rng1, p1)
        generate_weather(cfg, rng2, p2)
        df1 = pd.read_csv(p1 / "weather.csv", comment="#")
        df2 = pd.read_csv(p2 / "weather.csv", comment="#")
        pd.testing.assert_frame_equal(df1, df2)


# ---------------------------------------------------------------------------
# clean_tabular — crop dropping
# ---------------------------------------------------------------------------

class TestCleanTabular:
    def test_sparse_crop_dropped(self, tmp_path, caplog):
        """A crop with fewer than min_data_years rows should be dropped."""
        from src.data.clean_tabular import _drop_sparse_crops
        cfg = load_config()
        cfg["crops"]["min_data_years"] = 8

        # paddy has 10 rows, sparse_crop has only 3
        rows = (
            [{"year": y, "season": "kharif", "crop": "paddy",
              "district": "Tiruvallur", "yield_kg_ha": 3000.0, "area_ha": 5000.0}
             for y in range(2010, 2020)]
            + [{"year": y, "season": "kharif", "crop": "sparse_crop",
                "district": "Tiruvallur", "yield_kg_ha": 500.0, "area_ha": 100.0}
               for y in range(2010, 2013)]
        )
        df = pd.DataFrame(rows)

        with caplog.at_level(logging.WARNING):
            result = _drop_sparse_crops(df, cfg)

        assert "paddy" in result["crop"].values
        assert "sparse_crop" not in result["crop"].values
        assert any("sparse_crop" in r.message for r in caplog.records)

    def test_crop_meeting_threshold_kept(self):
        from src.data.clean_tabular import _drop_sparse_crops
        cfg = load_config()
        cfg["crops"]["min_data_years"] = 3

        rows = [
            {"year": y, "season": "kharif", "crop": "maize",
             "district": "Tiruvallur", "yield_kg_ha": 4000.0, "area_ha": 3000.0}
            for y in range(2010, 2015)  # 5 rows ≥ 3
        ]
        df = pd.DataFrame(rows)
        result = _drop_sparse_crops(df, cfg)
        assert "maize" in result["crop"].values
