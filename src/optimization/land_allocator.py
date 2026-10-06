"""
land_allocator.py
-----------------
Optional LP-based land allocation across Pareto-optimal crops.

Solves
------
    maximise   sum_i( area_i × profit_i )
    subject to:
        sum_i( area_i ) = total_acreage        [ha]
        sum_i( area_i × water_req_i ) ≤ water_budget   [mm·ha]
        area_i ≥ 0

Uses ``scipy.optimize.linprog`` — no external LP solver required.

Usage
-----
    from src.optimization.land_allocator import allocate_land
    alloc = allocate_land(pareto_df, total_acreage=100, water_budget=80000)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.logger import get_logger

logger = get_logger(__name__)


def allocate_land(
    pareto_df: pd.DataFrame,
    total_acreage: float,
    water_budget: float,
) -> pd.DataFrame:
    """Allocate land area across Pareto-optimal crops to maximise profit.

    Parameters
    ----------
    pareto_df:
        Pareto front DataFrame with columns crop, profit_inr_ha,
        water_deficit (mm).  water_deficit is used as a proxy for
        water demand per hectare (the deficit that would still need to
        be met from irrigation/groundwater).
    total_acreage:
        Total cultivable area available (ha).
    water_budget:
        Total seasonal water budget (mm × ha).  Each hectare of crop_i
        consumes ``water_deficit_i`` mm of water, so the constraint is
        sum(area_i × water_deficit_i) ≤ water_budget.

    Returns
    -------
    pd.DataFrame with columns crop, area_ha, area_fraction.
    Rows with area_ha < 0.01 ha are omitted.

    Raises
    ------
    ValueError
        If the LP is infeasible (water budget too small for any allocation).
    """
    n = len(pareto_df)
    if n == 0:
        raise ValueError("Pareto front is empty — cannot allocate land.")

    profits   = pareto_df["profit_inr_ha"].to_numpy(dtype=float)
    water_use = pareto_df["water_deficit"].to_numpy(dtype=float)

    # linprog minimises, so negate profit
    c = -profits

    # Inequality constraint: sum(area_i * water_use_i) <= water_budget
    A_ub = water_use.reshape(1, n)
    b_ub = np.array([water_budget])

    # Equality constraint: sum(area_i) == total_acreage
    A_eq = np.ones((1, n))
    b_eq = np.array([total_acreage])

    bounds = [(0, total_acreage)] * n

    result = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    if not result.success:
        if result.status == 2:  # infeasible
            raise ValueError(
                f"Land allocation is infeasible: water budget ({water_budget:.0f} mm·ha) "
                f"is too small to grow any crop on {total_acreage:.0f} ha. "
                "Increase the water budget or reduce total acreage."
            )
        raise ValueError(f"LP solver failed: {result.message}")

    areas = result.x
    fracs = areas / total_acreage

    alloc = pd.DataFrame({
        "crop":          pareto_df["crop"].tolist(),
        "area_ha":       np.round(areas, 2),
        "area_fraction": np.round(fracs, 4),
    })

    # Drop negligible allocations
    alloc = alloc[alloc["area_ha"] >= 0.01].reset_index(drop=True)

    logger.info(
        "Land allocated: %d crop(s), total %.1f ha, "
        "water used %.0f mm·ha of %.0f budget.",
        len(alloc),
        alloc["area_ha"].sum(),
        (alloc["area_ha"] * pareto_df.set_index("crop").loc[
            alloc["crop"], "water_deficit"
        ].values).sum(),
        water_budget,
    )
    return alloc


if __name__ == "__main__":
    from src.utils.config_loader import load_config
    import pandas as pd

    cfg = load_config()
    # Quick test with a toy Pareto front
    toy_pareto = pd.DataFrame({
        "crop":          ["paddy", "maize", "groundnut"],
        "season":        ["kharif"] * 3,
        "profit_inr_ha": [15000, 88000, 105000],
        "water_deficit": [314, 0, 0],
        "risk_score":    [0.16, 0.17, 0.54],
    })
    result = allocate_land(toy_pareto, total_acreage=100, water_budget=50000)
    print(result)
