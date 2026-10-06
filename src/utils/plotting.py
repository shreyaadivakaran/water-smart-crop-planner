"""
plotting.py
-----------
Shared Plotly figure factories for the Water-Smart Crop Planner.

Every function returns a ``plotly.graph_objects.Figure`` and also saves a
static PNG to ``outputs/plots/`` so outputs exist even when running headless.

Functions
---------
pareto_scatter        -- Pareto front scatter (profit × water, bubble = risk)
actual_vs_pred_plot   -- Actual vs predicted scatter for model evaluation
sensitivity_heatmap   -- Crop rank stability heatmap
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

_PLOTS_DIR = Path("outputs/plots")
_CROP_COLORS = px.colors.qualitative.Safe  # accessible palette


def _save_png(fig: go.Figure, filename: str) -> None:
    """Save figure as PNG; silently skip if kaleido is unavailable.

    Parameters
    ----------
    fig:      Plotly figure to save.
    filename: Filename (without path) ending in .png.
    """
    _PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = _PLOTS_DIR / filename
    try:
        fig.write_image(str(out), width=1000, height=600, scale=2)
    except Exception:  # kaleido not installed or other export error
        pass


# ---------------------------------------------------------------------------
# Pareto scatter
# ---------------------------------------------------------------------------

def pareto_scatter(
    pareto_df: pd.DataFrame,
    title: str = "Pareto Front — Profit vs Water Deficit",
) -> go.Figure:
    """Bubble chart of the Pareto-optimal crop set.

    X-axis : profit_inr_ha  (higher = better, so reversed axis optional)
    Y-axis : water_deficit (mm, lower = better)
    Bubble : proportional to risk_score (larger = riskier)
    Colour : one colour per crop

    Parameters
    ----------
    pareto_df:
        DataFrame with columns crop, season, profit_inr_ha,
        water_deficit, risk_score.
    title: Chart title string.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    df = pareto_df.copy()

    # Bubble size: scale risk [0,1] to pixel range [12, 50]
    df["bubble_size"] = 12 + df["risk_score"] * 38
    df["label"]       = df["crop"] + " (" + df["season"] + ")"
    df["profit_lakhs"] = df["profit_inr_ha"] / 1e5  # convert to lakhs for display

    fig = px.scatter(
        df,
        x="profit_lakhs",
        y="water_deficit",
        size="bubble_size",
        color="crop",
        text="label",
        hover_data={
            "crop":          True,
            "season":        True,
            "profit_inr_ha": ":,.0f",
            "water_deficit": ":.1f",
            "risk_score":    ":.3f",
            "bubble_size":   False,
            "label":         False,
            "profit_lakhs":  False,
        },
        size_max=50,
        color_discrete_sequence=_CROP_COLORS,
        title=title,
        labels={
            "profit_lakhs":  "Profit (₹ lakhs / ha)",
            "water_deficit": "Water Deficit (mm)",
        },
    )

    fig.update_traces(textposition="top center", textfont_size=11)
    fig.update_layout(
        xaxis_title="Profit (₹ lakhs / ha)  →  higher is better",
        yaxis_title="Water Deficit (mm)  →  lower is better",
        legend_title="Crop",
        font=dict(size=13),
        height=520,
        margin=dict(l=60, r=40, t=60, b=60),
    )

    _save_png(fig, "pareto_scatter.png")
    return fig


# ---------------------------------------------------------------------------
# Actual vs predicted
# ---------------------------------------------------------------------------

def actual_vs_pred_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Actual vs Predicted Yield",
    units: str = "kg/ha",
) -> go.Figure:
    """Scatter plot of actual vs predicted values with a diagonal reference line.

    Parameters
    ----------
    y_true: Array of true target values.
    y_pred: Array of predicted values.
    title:  Chart title.
    units:  Unit string for axis labels.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    vmin = min(y_true.min(), y_pred.min()) * 0.95
    vmax = max(y_true.max(), y_pred.max()) * 1.05

    fig = go.Figure()

    # Perfect-prediction diagonal
    fig.add_trace(go.Scatter(
        x=[vmin, vmax], y=[vmin, vmax],
        mode="lines",
        line=dict(color="grey", dash="dash", width=1),
        name="Perfect prediction",
        showlegend=True,
    ))

    # Data points
    fig.add_trace(go.Scatter(
        x=y_true, y=y_pred,
        mode="markers",
        marker=dict(color="#2563EB", size=8, opacity=0.7),
        name="Observations",
    ))

    fig.update_layout(
        title=title,
        xaxis_title=f"Actual ({units})",
        yaxis_title=f"Predicted ({units})",
        font=dict(size=13),
        height=480,
        margin=dict(l=60, r=40, t=60, b=60),
    )

    safe_title = title.lower().replace(" ", "_").replace("/", "_")
    _save_png(fig, f"{safe_title}.png")
    return fig


# ---------------------------------------------------------------------------
# Sensitivity heatmap
# ---------------------------------------------------------------------------

def sensitivity_heatmap(
    results_df: pd.DataFrame,
    title: str = "Rank Stability — Sensitivity Analysis",
) -> go.Figure:
    """Heatmap showing how crop ranks change across scenarios.

    Parameters
    ----------
    results_df:
        DataFrame with columns: scenario, crop, rank.
        ``scenario`` is a string label for each parameter combination.
    title: Chart title.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    pivot = results_df.pivot_table(
        index="crop", columns="scenario", values="rank", aggfunc="mean"
    )

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale="RdYlGn_r",   # low rank (1 = best) → green
        text=np.round(pivot.values, 1),
        texttemplate="%{text}",
        colorbar=dict(title="Mean Rank"),
        hoverongaps=False,
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Scenario",
        yaxis_title="Crop",
        font=dict(size=12),
        height=460,
        margin=dict(l=80, r=40, t=60, b=120),
        xaxis=dict(tickangle=35),
    )

    _save_png(fig, "sensitivity_heatmap.png")
    return fig


# ---------------------------------------------------------------------------
# Price interval bar chart
# ---------------------------------------------------------------------------

def price_interval_chart(
    crops: list[str],
    p10s: list[float],
    p50s: list[float],
    p90s: list[float],
    season: str,
) -> go.Figure:
    """Horizontal bar chart showing P10/P50/P90 price intervals per crop.

    Parameters
    ----------
    crops: Crop name list.
    p10s:  P10 price estimates (₹/kg).
    p50s:  P50 price estimates (₹/kg).
    p90s:  P90 price estimates (₹/kg).
    season: Season label for the title.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    fig = go.Figure()

    for i, crop in enumerate(crops):
        # Error bar from P10 to P90, point at P50
        fig.add_trace(go.Scatter(
            x=[p50s[i]],
            y=[crop],
            mode="markers",
            marker=dict(color=_CROP_COLORS[i % len(_CROP_COLORS)], size=12),
            error_x=dict(
                type="data",
                symmetric=False,
                array=[p90s[i] - p50s[i]],
                arrayminus=[p50s[i] - p10s[i]],
                thickness=3,
                width=6,
            ),
            name=crop,
            showlegend=False,
            hovertemplate=(
                f"<b>{crop}</b><br>"
                f"P10: ₹{p10s[i]:.2f}/kg<br>"
                f"P50: ₹{p50s[i]:.2f}/kg<br>"
                f"P90: ₹{p90s[i]:.2f}/kg<extra></extra>"
            ),
        ))

    fig.update_layout(
        title=f"Predicted Price Intervals — {season.capitalize()}",
        xaxis_title="Price (₹/kg)",
        yaxis_title="Crop",
        font=dict(size=13),
        height=420,
        margin=dict(l=100, r=40, t=60, b=60),
    )

    _save_png(fig, f"price_intervals_{season}.png")
    return fig
