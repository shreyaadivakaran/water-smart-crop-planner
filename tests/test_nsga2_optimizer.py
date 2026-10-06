"""
test_nsga2_optimizer.py
-----------------------
Unit tests for src/optimization/nsga2_optimizer.py and
src/optimization/weighted_ranker.py

Covers
------
* Pareto front solutions are non-dominated
* Result has expected columns
* Seed reproducibility: same seed → identical Pareto front
* Weighted ranker: correct column presence, rank starts at 1
* Weighted ranker raises ValueError when weights don't sum to 1
* water_saved_vs_paddy computes correct delta
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.optimization.nsga2_optimizer import run_nsga2, _is_nondominated
from src.optimization.weighted_ranker import rank_crops, water_saved_vs_paddy
from src.utils.config_loader import load_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_obj_df(n: int = 7, seed: int = 0) -> pd.DataFrame:
    """Generate a simple synthetic objectives DataFrame."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "crop":          [f"crop_{i}" for i in range(n)],
        "season":        ["kharif"] * n,
        "neg_profit":    -rng.uniform(10000, 200000, n),
        "profit_inr_ha":  rng.uniform(10000, 200000, n),
        "water_deficit":  rng.uniform(0, 1000, n),
        "risk_score":     rng.uniform(0, 1, n),
    })


# ---------------------------------------------------------------------------
# _is_nondominated
# ---------------------------------------------------------------------------

class TestIsNonDominated:
    def test_single_solution_is_nondominated(self):
        F = np.array([[1.0, 2.0, 3.0]])
        mask = _is_nondominated(F)
        assert bool(mask[0]) is True

    def test_dominated_solution_flagged(self):
        # Solution 0 dominates solution 1 (strictly better on all objectives)
        F = np.array([
            [1.0, 1.0, 1.0],  # dominates
            [2.0, 2.0, 2.0],  # dominated
        ])
        mask = _is_nondominated(F)
        assert bool(mask[0]) is True
        assert bool(mask[1]) is False

    def test_two_nondominated_solutions(self):
        # Trade-off: sol 0 better on obj 1, sol 1 better on obj 2
        F = np.array([
            [0.5, 1.5],
            [1.5, 0.5],
        ])
        mask = _is_nondominated(F)
        assert bool(mask[0]) is True
        assert bool(mask[1]) is True


# ---------------------------------------------------------------------------
# run_nsga2
# ---------------------------------------------------------------------------

class TestRunNSGA2:
    @pytest.fixture(scope="class")
    def cfg(self):
        c = load_config()
        # Speed up for testing
        c["optimization"]["population_size"] = 20
        c["optimization"]["n_generations"]   = 30
        return c

    @pytest.fixture(scope="class")
    def obj_df(self):
        # Deterministic fixture: 4 crops on a clear Pareto front
        # Crop 0: high profit, high water, low risk
        # Crop 1: low profit, low water, low risk
        # Crop 2: medium profit, zero water, high risk
        # Crop 3: medium profit, medium water, medium risk (dominated by mix)
        return pd.DataFrame({
            "crop":          ["crop_A", "crop_B", "crop_C", "crop_D"],
            "season":        ["kharif"] * 4,
            "neg_profit":    [-200000, -10000, -100000, -80000],
            "profit_inr_ha": [ 200000,  10000,  100000,  80000],
            "water_deficit": [  500.0,    0.0,      0.0,  200.0],
            "risk_score":    [    0.1,    0.1,      0.9,    0.5],
        })

    @pytest.fixture(scope="class")
    def pareto(self, obj_df, cfg):
        return run_nsga2(obj_df, cfg)

    def test_result_has_expected_columns(self, pareto):
        required = {"crop", "season", "profit_inr_ha", "water_deficit", "risk_score"}
        assert required.issubset(set(pareto.columns))

    def test_pareto_solutions_are_nondominated(self, pareto):
        """All returned solutions must be non-dominated (with 1e-6 tolerance)."""
        F = pareto[["profit_inr_ha", "water_deficit", "risk_score"]].to_numpy()
        # Minimisation space: negate profit
        F_min = np.column_stack([-F[:, 0], F[:, 1], F[:, 2]])
        n = len(F_min)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                dominated = (
                    np.all(F_min[j] <= F_min[i] + 1e-6) and
                    np.any(F_min[j] < F_min[i] - 1e-6)
                )
                assert not dominated, (
                    f"Solution {i} ({pareto.iloc[i]['crop']}) is dominated by "
                    f"solution {j} ({pareto.iloc[j]['crop']})"
                )

    def test_pareto_subset_of_candidates(self, obj_df, pareto):
        pareto_crops = set(pareto["crop"].unique())
        candidate_crops = set(obj_df["crop"].unique())
        assert pareto_crops.issubset(candidate_crops)

    def test_seed_reproducibility(self, obj_df, cfg):
        p1 = run_nsga2(obj_df, cfg)
        p2 = run_nsga2(obj_df, cfg)
        # Same crops on the Pareto front
        assert set(p1["crop"]) == set(p2["crop"])


# ---------------------------------------------------------------------------
# rank_crops
# ---------------------------------------------------------------------------

class TestRankCrops:
    @pytest.fixture
    def pareto_df(self):
        return pd.DataFrame({
            "crop":          ["paddy", "maize", "ragi", "groundnut"],
            "season":        ["kharif"] * 4,
            "profit_inr_ha": [15000, 88000, 11000, 105000],
            "water_deficit": [314, 0, 0, 0],
            "risk_score":    [0.16, 0.17, 0.13, 0.54],
        })

    def test_ranked_df_has_rank_column(self, pareto_df):
        ranked = rank_crops(pareto_df, {"profit": 0.5, "water": 0.3, "risk": 0.2})
        assert "rank" in ranked.columns

    def test_rank_starts_at_one(self, pareto_df):
        ranked = rank_crops(pareto_df, {"profit": 0.5, "water": 0.3, "risk": 0.2})
        assert ranked["rank"].min() == 1

    def test_rank_is_ascending(self, pareto_df):
        ranked = rank_crops(pareto_df, {"profit": 0.5, "water": 0.3, "risk": 0.2})
        scores = ranked["weighted_score"].tolist()
        assert scores == sorted(scores)

    def test_all_rows_present(self, pareto_df):
        ranked = rank_crops(pareto_df, {"profit": 0.5, "water": 0.3, "risk": 0.2})
        assert len(ranked) == len(pareto_df)

    def test_invalid_weights_raise(self, pareto_df):
        with pytest.raises(ValueError, match="sum to 1"):
            rank_crops(pareto_df, {"profit": 0.5, "water": 0.5, "risk": 0.5})

    def test_missing_weight_key_raises(self, pareto_df):
        with pytest.raises(ValueError):
            rank_crops(pareto_df, {"profit": 0.5, "water": 0.5})

    def test_zero_weights_normalised(self, pareto_df):
        # All zero except profit → should rank by profit only
        ranked = rank_crops(pareto_df, {"profit": 1.0, "water": 0.0, "risk": 0.0})
        # Highest profit crop (groundnut=105000) should be rank 1
        assert ranked.iloc[0]["crop"] == "groundnut"

    def test_water_heavy_weights(self, pareto_df):
        # Water-heavy: should favour zero-deficit crops over paddy(314 mm deficit)
        ranked = rank_crops(pareto_df, {"profit": 0.1, "water": 0.8, "risk": 0.1})
        assert ranked.iloc[0]["water_deficit"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# water_saved_vs_paddy
# ---------------------------------------------------------------------------

class TestWaterSavedVsPaddy:
    @pytest.fixture
    def ranked_with_paddy(self):
        return pd.DataFrame({
            "rank":          [1, 2, 3],
            "crop":          ["maize", "paddy", "ragi"],
            "season":        ["kharif"] * 3,
            "profit_inr_ha": [88000, 15000, 11000],
            "water_deficit": [0, 314, 0],
            "risk_score":    [0.17, 0.16, 0.13],
            "weighted_score":[0.3, 0.6, 0.8],
        })

    def test_correct_saving(self, ranked_with_paddy):
        stats = water_saved_vs_paddy(ranked_with_paddy)
        assert stats["top_crop"] == "maize"
        assert stats["water_saved_mm"] == pytest.approx(314.0)
        assert stats["water_saved_pct"] == pytest.approx(100.0)

    def test_no_paddy_returns_nan(self):
        ranked = pd.DataFrame({
            "rank": [1, 2],
            "crop": ["maize", "ragi"],
            "water_deficit": [0.0, 50.0],
        })
        stats = water_saved_vs_paddy(ranked)
        assert np.isnan(stats["water_saved_mm"])

    def test_top_uses_more_water_than_paddy(self):
        ranked = pd.DataFrame({
            "rank": [1, 2],
            "crop": ["sugarcane", "paddy"],
            "water_deficit": [900.0, 314.0],
            "profit_inr_ha": [1e6, 15000],
            "season": ["kharif", "kharif"],
            "risk_score": [0.0, 0.16],
            "weighted_score": [0.3, 0.7],
        })
        stats = water_saved_vs_paddy(ranked)
        assert stats["water_saved_mm"] < 0  # negative saving
