"""
pareto_chart.py
---------------
Pareto scatter chart and price interval chart for the dashboard.

Changes from v1
---------------
* Colour-blind safe palette (Okabe-Ito) with a legend explaining bubble size
  and axes.
* "How to read this chart" expander below the figure.
* Price interval chart included when available.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Okabe-Ito colour-blind safe palette
_CB_PALETTE = [
    "#E69F00", "#56B4E9", "#009E73", "#F0E442",
    "#0072B2", "#D55E00", "#CC79A7", "#000000",
]

_DISCLAIMER = (
    "> **Disclaimer:** Outputs are indicative estimates based on historical data "
    "and synthetic modelling. Consult local agricultural experts before making "
    "planting decisions."
)

_HOW_TO_READ = """
**X-axis (Profit):** Higher = more income per hectare.

**Y-axis (Water Deficit):** Lower = less extra water needed beyond rainfall.

**Bubble size:** Larger bubble = higher climate risk (more year-to-year yield variability).

**Colour:** Each colour is a different crop.

**What "Pareto-optimal" means:** Every crop shown here is the best choice for *some* combination
of priorities — no other crop beats it on all three objectives simultaneously.
Moving right makes you richer; moving down saves water; smaller bubbles mean more stability.
Pick the crop that matches your priorities, or use the weight sliders to let the model rank them.
"""


def _pareto_scatter_cb(
    pareto_df: pd.DataFrame,
    title: str,
) -> go.Figure:
    """Build a colour-blind safe Pareto scatter with labelled bubbles."""
    df = pareto_df.copy()
    df["bubble_size"]   = 12 + df["risk_score"] * 38
    df["label"]         = df["crop"].str.capitalize()
    df["profit_lakhs"]  = df["profit_inr_ha"] / 1e5
    df["risk_pct"]      = (df["risk_score"] * 100).round(1)

    crops  = sorted(df["crop"].unique())
    colour = {c: _CB_PALETTE[i % len(_CB_PALETTE)] for i, c in enumerate(crops)}
    df["colour"] = df["crop"].map(colour)

    fig = go.Figure()

    for _, row in df.iterrows():
        fig.add_trace(go.Scatter(
            x=[row["profit_lakhs"]],
            y=[row["water_deficit"]],
            mode="markers+text",
            marker=dict(
                color=row["colour"],
                size=row["bubble_size"],
                opacity=0.85,
                line=dict(width=1.5, color="white"),
            ),
            text=[row["label"]],
            textposition="top center",
            textfont=dict(size=12, color="#1a1a2e"),
            name=row["label"],
            hovertemplate=(
                f"<b>{row['label']}</b><br>"
                f"Profit: ₹{row['profit_inr_ha']:,.0f}/ha<br>"
                f"Water deficit: {row['water_deficit']:.0f} mm<br>"
                f"Risk score: {row['risk_pct']:.1f}%"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis=dict(
            title="Estimated Profit (₹ lakhs / ha)  →  higher is better",
            gridcolor="#e0e0e0",
        ),
        yaxis=dict(
            title="Water Deficit (mm)  →  lower is better",
            gridcolor="#e0e0e0",
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=True,
        legend=dict(
            title="Crop",
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5,
        ),
        font=dict(size=13, color="#1a1a2e"),
        height=500,
        margin=dict(l=60, r=40, t=60, b=100),
    )
    return fig


def render_pareto_tab(
    pareto_df: pd.DataFrame,
    price_intervals: list[dict] | None,
    season: str,
    rainfall_label: str,
) -> None:
    """Render the Pareto scatter, how-to-read expander, and price intervals.

    Parameters
    ----------
    pareto_df:       Pareto-optimal crops DataFrame.
    price_intervals: List of dicts {crop, p10, p50, p90} or None.
    season:          Current season name.
    rainfall_label:  Human-readable rainfall scenario label.
    """
    st.subheader(f"Pareto Front — {season.capitalize()} · {rainfall_label}")

    n_crops = len(pareto_df)
    st.caption(
        f"{n_crops} crop{'s' if n_crops != 1 else ''} reached the Pareto front "
        f"(no other crop beats them on all three objectives simultaneously)."
    )

    fig = _pareto_scatter_cb(
        pareto_df,
        title=f"Profit vs Water Deficit — {season.capitalize()} / {rainfall_label}",
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📖 How to read this chart"):
        st.markdown(_HOW_TO_READ)
        st.markdown(
            "**Colour accessibility:** This chart uses the Okabe-Ito palette, "
            "which is distinguishable by people with the most common forms of "
            "colour vision deficiency."
        )

    if price_intervals:
        st.markdown("#### Predicted Wholesale Price Intervals (P10 / P50 / P90)")
        st.caption(
            "Whiskers show the range of likely prices. The dot is the median estimate. "
            "Actual farm-gate prices may differ."
        )
        from src.utils.plotting import price_interval_chart  # noqa: PLC0415
        crops = [d["crop"] for d in price_intervals]
        p10s  = [d["p10"]  for d in price_intervals]
        p50s  = [d["p50"]  for d in price_intervals]
        p90s  = [d["p90"]  for d in price_intervals]
        fig2  = price_interval_chart(crops, p10s, p50s, p90s, season)
        # Apply same colour palette to price chart
        fig2.update_layout(paper_bgcolor="white", plot_bgcolor="white")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown(_DISCLAIMER)
