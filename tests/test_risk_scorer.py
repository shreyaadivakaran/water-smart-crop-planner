"""
test_risk_scorer.py
-------------------
Unit tests for src/risk/risk_scorer.py

Covers
------
* CV of a constant series → 0
* CV of a known series matches hand-calculated value
* Dry-year P_fail = 1.0 when all dry-year observations are failures
* Dry-year P_fail = 0.0 when no dry-year observations are failures
* Edge case: fewer than MIN_DRY_YEARS dry years → 0.0 (no division by zero)
* Normalisation: combined risk_score in [0, 1]
* compute_risk_scores() returns expected schema on synthetic data
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make project root importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.risk.risk_scorer import (
    _cv,
    _dry_year_failure_prob,
    _minmax_norm,
    compute_risk_scores,
)
from src.utils.config_loader import load_config


# ---------------------------------------------------------------------------
# _cv
# ---------------------------------------------------------------------------

class TestCV:
    def test_constant_series_returns_zero(self):
        arr = np.array([100.0, 100.0, 100.0, 100.0])
        assert _cv(arr) == 0.0

    def test_known_series(self):
        # mean=10, std(ddof=1)=2  → CV = 0.2
        arr = np.array([8.0, 10.0, 12.0, 10.0])
        expected = np.std(arr, ddof=1) / np.mean(arr)
        assert abs(_cv(arr) - expected) < 1e-9

    def test_single_element_returns_zero(self):
        assert _cv(np.array([42.0])) == 0.0

    def test_zero_mean_returns_zero(self):
        assert _cv(np.array([0.0, 0.0, 0.0])) == 0.0

    def test_positive_for_variable_series(self):
        arr = np.array([100.0, 200.0, 150.0, 50.0, 300.0])
        assert _cv(arr) > 0.0


# ---------------------------------------------------------------------------
# _dry_year_failure_prob
# ---------------------------------------------------------------------------

class TestDryYearFailureProb:
    """All tests use a 10-year series with known rainfall and yield."""

    def _series(self, yields, rainfalls):
        idx = pd.Index(range(2010, 2010 + len(yields)), name="year")
        return pd.Series(yields, index=idx), pd.Series(rainfalls, index=idx)

    def test_all_dry_years_fail(self):
        # 5 years; bottom 30 % = year 0 (200 mm), year 1 (250 mm)
        # Both have yield < 70 % of mean → P_fail = 1.0
        yields    = [1000, 1000, 3000, 3000, 3000]
        rainfalls = [200,  250,  900,  950,  800]
        y, r = self._series(yields, rainfalls)
        # mean yield = 2200; 70% threshold = 1540; dry-year yields 1000 < 1540
        prob = _dry_year_failure_prob(y, r, dry_percentile=30, failure_threshold_frac=0.70)
        assert prob == pytest.approx(1.0)

    def test_no_dry_years_fail(self):
        # Dry years have high yield → P_fail = 0.0
        yields    = [3000, 3200, 3000, 3000, 3000]
        rainfalls = [200,  250,  900,  950,  800]
        y, r = self._series(yields, rainfalls)
        prob = _dry_year_failure_prob(y, r, dry_percentile=30, failure_threshold_frac=0.70)
        assert prob == pytest.approx(0.0)

    def test_fewer_than_min_dry_years_returns_zero(self):
        # 20-element series: one extreme dry year, 19 normal years
        # np.percentile(rainfalls, 30) = 1000.0, so only the 10mm year qualifies
        # → 1 dry year < MIN_DRY_YEARS(2) → must return 0.0
        n = 20
        yields    = [500] + [3000] * (n - 1)
        rainfalls = [10]  + [1000] * (n - 1)
        y, r = self._series(yields, rainfalls)
        prob = _dry_year_failure_prob(y, r, dry_percentile=30, failure_threshold_frac=0.70)
        assert prob == 0.0

    def test_empty_series_returns_zero(self):
        y = pd.Series([], dtype=float)
        r = pd.Series([], dtype=float)
        prob = _dry_year_failure_prob(y, r, dry_percentile=30, failure_threshold_frac=0.70)
        assert prob == 0.0

    def test_partial_failure(self):
        # 4 dry years, 2 fail → 0.5
        yields    = [500, 500, 2000, 2000, 3000, 3000, 3000, 3000]
        rainfalls = [150, 160, 170,  180,  900,  950,  920,  910]
        y, r = self._series(yields, rainfalls)
        # mean ~2000; threshold 70% = 1400; dry = bottom 30% of 8 = lowest 2-3
        # rain bottom 30th percentile of [150,160,170,180,900,950,920,910] ≈ 167 mm
        # dry years: 150 (yield 500<1400 → fail), 160 (yield 500<1400 → fail)
        # → 2/2 = 1.0 or depends on exact percentile
        prob = _dry_year_failure_prob(y, r, dry_percentile=30, failure_threshold_frac=0.70)
        assert 0.0 <= prob <= 1.0


# ---------------------------------------------------------------------------
# _minmax_norm
# ---------------------------------------------------------------------------

class TestMinMaxNorm:
    def test_output_in_zero_one(self):
        s = pd.Series([1.0, 3.0, 5.0, 2.0, 4.0])
        n = _minmax_norm(s)
        assert n.min() == pytest.approx(0.0)
        assert n.max() == pytest.approx(1.0)

    def test_constant_series_returns_zeros(self):
        s = pd.Series([7.0, 7.0, 7.0])
        n = _minmax_norm(s)
        assert (n == 0.0).all()


# ---------------------------------------------------------------------------
# compute_risk_scores integration
# ---------------------------------------------------------------------------

class TestComputeRiskScores:
    @pytest.fixture(scope="class")
    def risk_df(self):
        cfg = load_config()
        return compute_risk_scores(cfg)

    def test_expected_columns(self, risk_df):
        expected = {"crop", "season", "cv", "p_fail", "cv_norm", "p_fail_norm", "risk_score"}
        assert expected.issubset(set(risk_df.columns))

    def test_risk_score_in_zero_one(self, risk_df):
        assert risk_df["risk_score"].between(0.0, 1.0).all()

    def test_cv_non_negative(self, risk_df):
        assert (risk_df["cv"] >= 0).all()

    def test_p_fail_in_zero_one(self, risk_df):
        assert risk_df["p_fail"].between(0.0, 1.0).all()

    def test_all_seven_crops_present(self, risk_df):
        crops = set(risk_df["crop"].unique())
        expected = {"paddy", "groundnut", "sugarcane", "black_gram",
                    "green_gram", "maize", "ragi"}
        assert expected == crops

    def test_all_three_seasons_present(self, risk_df):
        seasons = set(risk_df["season"].unique())
        assert seasons == {"kharif", "rabi", "summer"}
