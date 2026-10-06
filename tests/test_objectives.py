"""
test_objectives.py
------------------
Unit tests for src/optimization/objectives.py

Covers
------
* Water deficit is always ≥ 0 (never negative)
* Profit sign is correct (positive yield × positive price − cost)
* Known inputs produce expected objective values
* compute_objectives_batch() returns expected shape and columns
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.optimization.objectives import compute_objectives, compute_objectives_batch


# ---------------------------------------------------------------------------
# Fixtures — lightweight mock bundles so tests run without disk I/O
# ---------------------------------------------------------------------------

def _make_yield_bundle(fixed_yield: float = 3000.0):
    """Return a mock yield bundle that always predicts ``fixed_yield``."""
    from src.models.yield_model import predict_yield  # import real function

    bundle = MagicMock()

    def _predict(**kwargs):
        return fixed_yield

    bundle.__class__.__name__ = "mock_yield_bundle"
    return bundle, fixed_yield


def _make_price_bundle(fixed_price: float = 20.0):
    """Return a mock price bundle that always returns fixed P10/P50/P90."""
    from src.models.price_model import PriceInterval
    bundle = MagicMock()
    bundle.get.return_value = False  # synthetic flag
    bundle.__getitem__ = lambda self, k: False if k == "synthetic" else MagicMock()
    return bundle, fixed_price


def _make_feature_row(
    crop: str = "paddy",
    season: str = "kharif",
    year: int = 2022,
    rainfall: float = 850.0,
    water_req: float = 1200.0,
    cost: float = 38000.0,
) -> pd.DataFrame:
    return pd.DataFrame([{
        "crop":                  crop,
        "season":                season,
        "year":                  year,
        "seasonal_rainfall_mm":  rainfall,
        "mean_tmax":             32.0,
        "mean_tmin":             24.0,
        "heat_stress_days":      10.0,
        "dry_spell_days":        15.0,
        "water_req_mm":          water_req,
        "cost_inr_ha":           cost,
    }])


def _make_risk_df(crop: str = "paddy", season: str = "kharif", score: float = 0.3):
    return pd.DataFrame([{"crop": crop, "season": season, "risk_score": score}])


# ---------------------------------------------------------------------------
# Water deficit non-negativity
# ---------------------------------------------------------------------------

class TestWaterDeficit:
    """Water deficit must never be negative."""

    def test_surplus_rainfall_gives_zero_deficit(self, monkeypatch):
        """When rainfall + irrigation > water_req, deficit = 0."""
        features = _make_feature_row(rainfall=2000.0, water_req=1200.0)
        risk_df  = _make_risk_df()

        # Monkeypatch predict_yield and predict_price to avoid real model load
        monkeypatch.setattr(
            "src.optimization.objectives.predict_yield",
            lambda **kw: 3000.0,
        )
        monkeypatch.setattr(
            "src.optimization.objectives.predict_price",
            lambda *a, **kw: type("P", (), {"p50": 20.0})(),
        )
        monkeypatch.setattr(
            "src.optimization.objectives.get_risk_score",
            lambda *a, **kw: 0.3,
        )

        neg_profit, water_def, risk = compute_objectives(
            crop="paddy", season="kharif", year=2022,
            irrigation_mm=0.0,
            rainfall_scenario_factor=1.0,
            features_df=features,
            risk_df=risk_df,
            yield_bundle=MagicMock(),
            price_bundle=MagicMock(),
        )
        assert water_def == pytest.approx(0.0), "Deficit should be 0 when water is surplus"

    def test_deficit_rainfall_gives_positive_deficit(self, monkeypatch):
        """When rainfall + irrigation < water_req, deficit > 0."""
        features = _make_feature_row(rainfall=200.0, water_req=1200.0)
        risk_df  = _make_risk_df()

        monkeypatch.setattr("src.optimization.objectives.predict_yield",  lambda **kw: 3000.0)
        monkeypatch.setattr("src.optimization.objectives.predict_price",   lambda *a, **kw: type("P",(),{"p50":20.0})())
        monkeypatch.setattr("src.optimization.objectives.get_risk_score",  lambda *a, **kw: 0.3)

        _, water_def, _ = compute_objectives(
            crop="paddy", season="kharif", year=2022,
            irrigation_mm=100.0,
            rainfall_scenario_factor=1.0,
            features_df=features,
            risk_df=risk_df,
            yield_bundle=MagicMock(),
            price_bundle=MagicMock(),
        )
        expected = max(0.0, 1200.0 - (200.0 + 100.0))
        assert water_def == pytest.approx(expected)

    def test_water_deficit_never_negative(self, monkeypatch):
        """Regardless of parameters, water_deficit ≥ 0."""
        monkeypatch.setattr("src.optimization.objectives.predict_yield",  lambda **kw: 3000.0)
        monkeypatch.setattr("src.optimization.objectives.predict_price",   lambda *a, **kw: type("P",(),{"p50":20.0})())
        monkeypatch.setattr("src.optimization.objectives.get_risk_score",  lambda *a, **kw: 0.3)

        for rain in [0, 100, 500, 1500, 5000]:
            features = _make_feature_row(rainfall=float(rain), water_req=1200.0)
            _, water_def, _ = compute_objectives(
                crop="paddy", season="kharif", year=2022,
                irrigation_mm=0.0,
                rainfall_scenario_factor=1.0,
                features_df=features,
                risk_df=_make_risk_df(),
                yield_bundle=MagicMock(),
                price_bundle=MagicMock(),
            )
            assert water_def >= 0.0, f"Negative deficit for rainfall={rain}"


# ---------------------------------------------------------------------------
# Profit sign and magnitude
# ---------------------------------------------------------------------------

class TestProfit:
    def test_profit_sign(self, monkeypatch):
        """neg_profit = −(yield × price − cost); should be negative when profitable."""
        yield_val  = 3000.0   # kg/ha
        price_val  = 20.0     # ₹/kg
        cost_val   = 38000.0  # ₹/ha
        # expected profit = 3000 × 20 − 38000 = 22000
        expected_neg_profit = -(yield_val * price_val - cost_val)  # −22000

        monkeypatch.setattr("src.optimization.objectives.predict_yield",  lambda **kw: yield_val)
        monkeypatch.setattr("src.optimization.objectives.predict_price",   lambda *a, **kw: type("P",(),{"p50": price_val})())
        monkeypatch.setattr("src.optimization.objectives.get_risk_score",  lambda *a, **kw: 0.3)

        features = _make_feature_row(cost=cost_val)
        neg_profit, _, _ = compute_objectives(
            crop="paddy", season="kharif", year=2022,
            irrigation_mm=100.0,
            rainfall_scenario_factor=1.0,
            features_df=features,
            risk_df=_make_risk_df(),
            yield_bundle=MagicMock(),
            price_bundle=MagicMock(),
        )
        assert neg_profit == pytest.approx(expected_neg_profit, rel=1e-3)

    def test_negative_profit_scenario(self, monkeypatch):
        """When cost > revenue, profit is negative, neg_profit is positive."""
        monkeypatch.setattr("src.optimization.objectives.predict_yield",  lambda **kw: 100.0)
        monkeypatch.setattr("src.optimization.objectives.predict_price",   lambda *a, **kw: type("P",(),{"p50": 5.0})())
        monkeypatch.setattr("src.optimization.objectives.get_risk_score",  lambda *a, **kw: 0.5)

        features   = _make_feature_row(cost=50000.0)
        neg_profit, _, _ = compute_objectives(
            crop="paddy", season="kharif", year=2022,
            irrigation_mm=0.0,
            rainfall_scenario_factor=1.0,
            features_df=features,
            risk_df=_make_risk_df(),
            yield_bundle=MagicMock(),
            price_bundle=MagicMock(),
        )
        # revenue = 100 × 5 = 500; cost = 50000 → profit = −49500 → neg_profit = 49500
        assert neg_profit > 0


# ---------------------------------------------------------------------------
# Batch computation
# ---------------------------------------------------------------------------

class TestBatch:
    def test_batch_shape_and_columns(self, monkeypatch):
        monkeypatch.setattr("src.optimization.objectives.predict_yield",  lambda **kw: 3000.0)
        monkeypatch.setattr("src.optimization.objectives.predict_price",   lambda *a, **kw: type("P",(),{"p50": 20.0})())
        monkeypatch.setattr("src.optimization.objectives.get_risk_score",  lambda *a, **kw: 0.3)

        crops = ["paddy", "maize", "ragi"]
        rows  = []
        for c in crops:
            rows.append({
                "crop": c, "season": "kharif", "year": 2022,
                "seasonal_rainfall_mm": 850.0, "mean_tmax": 32.0,
                "mean_tmin": 24.0, "heat_stress_days": 10.0,
                "dry_spell_days": 15.0, "water_req_mm": 800.0,
                "cost_inr_ha": 30000.0,
            })
        features = pd.DataFrame(rows)
        risk_df  = pd.DataFrame([
            {"crop": c, "season": "kharif", "risk_score": 0.3} for c in crops
        ])

        df = compute_objectives_batch(
            crops=crops, season="kharif", year=2022,
            irrigation_mm=100.0, rainfall_scenario_factor=1.0,
            features_df=features, risk_df=risk_df,
            yield_bundle=MagicMock(), price_bundle=MagicMock(),
        )
        assert len(df) == 3
        required = {"crop", "season", "neg_profit", "profit_inr_ha", "water_deficit", "risk_score"}
        assert required.issubset(set(df.columns))

    def test_profit_inr_ha_is_neg_of_neg_profit(self, monkeypatch):
        monkeypatch.setattr("src.optimization.objectives.predict_yield",  lambda **kw: 3000.0)
        monkeypatch.setattr("src.optimization.objectives.predict_price",   lambda *a, **kw: type("P",(),{"p50": 20.0})())
        monkeypatch.setattr("src.optimization.objectives.get_risk_score",  lambda *a, **kw: 0.3)

        features = _make_feature_row()
        risk_df  = _make_risk_df()
        df = compute_objectives_batch(
            crops=["paddy"], season="kharif", year=2022,
            irrigation_mm=0.0, rainfall_scenario_factor=1.0,
            features_df=features, risk_df=risk_df,
            yield_bundle=MagicMock(), price_bundle=MagicMock(),
        )
        assert df.iloc[0]["profit_inr_ha"] == pytest.approx(
            -df.iloc[0]["neg_profit"], rel=1e-9
        )
