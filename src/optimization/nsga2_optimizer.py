"""
nsga2_optimizer.py
------------------
Wraps pymoo's NSGA-II to solve the three-objective crop selection problem.

Decision variable
-----------------
Integer index into the list of candidate crops (discrete choice).

Objectives (all minimised)
--------------------------
f1 = −profit        (₹/ha)
f2 = water_deficit  (mm)
f3 = risk_score     (normalised [0,1])

The optimiser returns the full Pareto-optimal front as a tidy DataFrame.

Usage
-----
    python src/optimization/nsga2_optimizer.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.termination import get_termination

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# pymoo Problem definition
# ---------------------------------------------------------------------------

class CropSelectionProblem(Problem):
    """Three-objective crop selection problem for NSGA-II.

    The single decision variable is an integer in [0, n_crops − 1]
    representing the index of the chosen crop.

    Parameters
    ----------
    obj_df:
        DataFrame with columns [crop, neg_profit, water_deficit, risk_score]
        pre-computed for all candidate crops under the current scenario.
    """

    def __init__(self, obj_df: pd.DataFrame) -> None:
        self.obj_df  = obj_df.reset_index(drop=True)
        self.n_crops = len(obj_df)
        super().__init__(
            n_var=1,
            n_obj=3,
            n_ieq_constr=0,
            xl=np.array([0]),
            xu=np.array([self.n_crops - 1]),
            vtype=int,
        )

    def _evaluate(
        self,
        x: np.ndarray,
        out: dict[str, np.ndarray],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        indices = x[:, 0].astype(int).clip(0, self.n_crops - 1)
        F = self.obj_df.iloc[indices][
            ["neg_profit", "water_deficit", "risk_score"]
        ].to_numpy(dtype=float)
        out["F"] = F


# ---------------------------------------------------------------------------
# Pareto front extraction helpers
# ---------------------------------------------------------------------------

def _is_nondominated(F: np.ndarray) -> np.ndarray:
    """Return a boolean mask of non-dominated solutions.

    Parameters
    ----------
    F: Array of shape (n_solutions, n_objectives). All objectives minimised.

    Returns
    -------
    np.ndarray of bool, True for non-dominated solutions.
    """
    n = F.shape[0]
    is_nd = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # j dominates i if j ≤ i on all objectives and < i on at least one
            if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                is_nd[i] = False
                break
    return is_nd


# ---------------------------------------------------------------------------
# Main optimiser
# ---------------------------------------------------------------------------

def run_nsga2(
    obj_df: pd.DataFrame,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """Run NSGA-II and return the Pareto-optimal crop set.

    Parameters
    ----------
    obj_df:
        Pre-computed objectives DataFrame (from
        :func:`src.optimization.objectives.compute_objectives_batch`).
        Must contain columns: crop, season, neg_profit, profit_inr_ha,
        water_deficit, risk_score.
    cfg:
        Parsed configuration dictionary.

    Returns
    -------
    pd.DataFrame of Pareto-optimal crops with columns:
        crop, season, profit_inr_ha, water_deficit, risk_score
    Rows are non-dominated in the (neg_profit, water_deficit, risk_score) space.
    """
    pop_size  = cfg["optimization"]["population_size"]
    n_gen     = cfg["optimization"]["n_generations"]
    seed      = cfg["optimization"]["random_seed"]

    problem     = CropSelectionProblem(obj_df)
    algorithm   = NSGA2(pop_size=pop_size)
    termination = get_termination("n_gen", n_gen)

    logger.info(
        "Running NSGA-II: %d crops, pop=%d, generations=%d, seed=%d",
        len(obj_df), pop_size, n_gen, seed,
    )

    result = minimize(
        problem,
        algorithm,
        termination,
        seed=seed,
        verbose=False,
    )

    # Extract unique crop indices on the Pareto front
    if result.X is None:
        logger.warning("NSGA-II returned no solutions — returning full obj_df.")
        return obj_df[["crop", "season", "profit_inr_ha", "water_deficit", "risk_score"]].copy()

    pareto_indices = np.unique(result.X.astype(int).clip(0, len(obj_df) - 1))
    pareto_df = obj_df.iloc[pareto_indices].copy()

    # Double-check non-dominance (pymoo result may contain duplicates)
    F = pareto_df[["neg_profit", "water_deficit", "risk_score"]].to_numpy()
    nd_mask = _is_nondominated(F)
    pareto_df = pareto_df[nd_mask].reset_index(drop=True)

    logger.info(
        "Pareto front: %d non-dominated crops out of %d candidates.",
        len(pareto_df), len(obj_df),
    )

    return pareto_df[
        ["crop", "season", "profit_inr_ha", "water_deficit", "risk_score"]
    ].copy()


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pandas as pd

    cfg = load_config()

    # Build a quick obj_df from features + trained models
    from src.data.build_features import build_features  # noqa: PLC0415
    from src.models.yield_model import train_yield_models  # noqa: PLC0415
    from src.models.price_model import train_price_model  # noqa: PLC0415
    from src.risk.risk_scorer import compute_risk_scores  # noqa: PLC0415
    from src.optimization.objectives import compute_objectives_batch  # noqa: PLC0415

    features_df  = pd.read_parquet("data/processed/features.parquet")
    yield_bundle = train_yield_models(cfg)
    price_bundle = train_price_model(cfg)
    risk_df      = compute_risk_scores(cfg)

    crops  = cfg["crops"]["candidates"]
    obj_df = compute_objectives_batch(
        crops=crops,
        season="kharif",
        year=2022,
        irrigation_mm=100,
        rainfall_scenario_factor=1.0,
        features_df=features_df,
        risk_df=risk_df,
        yield_bundle=yield_bundle,
        price_bundle=price_bundle,
    )

    pareto = run_nsga2(obj_df, cfg)
    print("\nPareto front:")
    print(pareto.to_string(index=False))
