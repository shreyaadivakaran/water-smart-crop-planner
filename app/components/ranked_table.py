"""
ranked_table.py
---------------
Ranked crop table, water-saving metrics, download buttons, and
optional land allocation results.

Changes from v1
---------------
* Download CSV button for the ranked table.
* Download plain-text summary button.
* Land allocation section unchanged.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import streamlit as st

_DISCLAIMER = (
    "> **Disclaimer:** Outputs are indicative estimates based on historical data "
    "and synthetic modelling. Consult local agricultural experts before making "
    "planting decisions."
)


def _build_summary_text(
    ranked_df: pd.DataFrame,
    water_stats: dict,
    params: dict,
) -> str:
    """Build a one-page plain-text summary for download.

    Parameters
    ----------
    ranked_df:   Ranked crops DataFrame.
    water_stats: Water savings dict.
    params:      Sidebar params dict.

    Returns
    -------
    str  Plain text content.
    """
    top   = ranked_df.iloc[0]
    lines = [
        "Water-Smart Crop Planner — One-Page Summary",
        "=" * 50,
        f"Season:            {params.get('season', '—').capitalize()}",
        f"Year:              {params.get('year', '—')}",
        f"Rainfall scenario: {params.get('rainfall_scenario_label', '—')} "
        f"(×{params.get('rainfall_scenario_factor', 1):.2f}, "
        f"~{params.get('est_rain_mm', 0):.0f} mm)",
        f"Irrigation:        {params.get('irrigation_mm', 0):.0f} mm",
        f"Objective preset:  {params.get('active_preset', 'Custom')}",
        "",
        "TOP RECOMMENDATION",
        "-" * 30,
        f"Crop:              {str(top['crop']).capitalize()}",
        f"Profit (est.):     ₹{top['profit_inr_ha']:,.0f}/ha  "
        f"(₹{top['profit_inr_ha'] / 2.47:.0f}/acre)",
        f"Water deficit:     {top['water_deficit']:.0f} mm",
        f"Risk score:        {top['risk_score']:.2f} / 1.00",
        "",
        "WATER SAVINGS vs PADDY BASELINE",
        "-" * 30,
    ]
    saved_mm  = water_stats.get("water_saved_mm", float("nan"))
    saved_pct = water_stats.get("water_saved_pct", float("nan"))
    paddy_def = water_stats.get("paddy_water_deficit", float("nan"))
    if not np.isnan(saved_mm):
        lines.append(f"Paddy deficit:     {paddy_def:.0f} mm")
        lines.append(f"Water saved:       {saved_mm:.0f} mm  ({saved_pct:.1f}%)")
    else:
        lines.append("Paddy not in current Pareto set.")

    lines += [
        "",
        "FULL RANKING",
        "-" * 30,
    ]
    for _, row in ranked_df.iterrows():
        lines.append(
            f"  #{int(row['rank'])}  {str(row['crop']).capitalize():<14}  "
            f"Profit ₹{row['profit_inr_ha']:>10,.0f}/ha  "
            f"Water {row['water_deficit']:>6.0f} mm  "
            f"Risk {row['risk_score']:.2f}"
        )

    lines += [
        "",
        "DISCLAIMER",
        "-" * 30,
        "Outputs are indicative estimates based on historical data and",
        "synthetic modelling. Consult local agricultural experts before",
        "making planting decisions.",
    ]
    return "\n".join(lines)


def render_ranked_table_tab(
    ranked_df: pd.DataFrame,
    water_stats: dict,
    allocation_df: pd.DataFrame | None,
    price_synthetic: bool,
    params: dict | None = None,
) -> None:
    """Render ranked table, water metrics, and download buttons.

    Parameters
    ----------
    ranked_df:       Output of weighted_ranker.rank_crops().
    water_stats:     Output of weighted_ranker.water_saved_vs_paddy().
    allocation_df:   Optional land allocation DataFrame.
    price_synthetic: If True, show synthetic-data note.
    params:          Sidebar params dict (used for summary text).
    """
    st.subheader("Ranked Crop Recommendations")

    # ---- Display columns -------------------------------------------------
    display_cols = [
        "rank", "crop", "season",
        "profit_inr_ha", "water_deficit",
        "risk_score", "weighted_score",
    ]
    display_cols = [c for c in display_cols if c in ranked_df.columns]

    rename_map = {
        "rank":           "Rank",
        "crop":           "Crop",
        "season":         "Season",
        "profit_inr_ha":  "Profit (₹/ha)",
        "water_deficit":  "Water Deficit (mm)",
        "risk_score":     "Risk Score",
        "weighted_score": "Weighted Score",
    }
    table = ranked_df[display_cols].rename(columns=rename_map)

    # Add ₹/acre column next to ₹/ha
    if "Profit (₹/ha)" in table.columns:
        table.insert(
            table.columns.get_loc("Profit (₹/ha)") + 1,
            "Profit (₹/acre)",
            (table["Profit (₹/ha)"] / 2.47).round(0),
        )

    def _highlight_top(row: pd.Series) -> list[str]:
        return (
            ["background-color: #d1fae5; font-weight: bold"] * len(row)
            if row["Rank"] == 1
            else [""] * len(row)
        )

    fmt: dict[str, str] = {}
    if "Profit (₹/ha)" in table.columns:
        fmt["Profit (₹/ha)"]   = "{:,.0f}"
        fmt["Profit (₹/acre)"] = "{:,.0f}"
    if "Water Deficit (mm)" in table.columns:
        fmt["Water Deficit (mm)"] = "{:.1f}"
    if "Risk Score" in table.columns:
        fmt["Risk Score"] = "{:.3f}"
    if "Weighted Score" in table.columns:
        fmt["Weighted Score"] = "{:.3f}"

    st.dataframe(
        table.style.apply(_highlight_top, axis=1).format(fmt),
        use_container_width=True,
        height=min(420, 60 + 38 * len(table)),
    )

    # ---- Download buttons ------------------------------------------------
    dl_col1, dl_col2, _ = st.columns([1, 1, 2])

    csv_bytes = table.to_csv(index=False).encode("utf-8")
    dl_col1.download_button(
        "⬇️ Download table (CSV)",
        data=csv_bytes,
        file_name="crop_recommendations.csv",
        mime="text/csv",
        use_container_width=True,
        help="Download the ranked table as a CSV file.",
    )

    if params:
        summary_txt = _build_summary_text(ranked_df, water_stats, params)
        dl_col2.download_button(
            "⬇️ Download summary (TXT)",
            data=summary_txt.encode("utf-8"),
            file_name="crop_planner_summary.txt",
            mime="text/plain",
            use_container_width=True,
            help="Download a one-page plain-text summary.",
        )

    # ---- Water saved vs paddy -------------------------------------------
    st.markdown("#### 💧 Water Savings vs Paddy Baseline")
    col1, col2, col3 = st.columns(3)

    top_crop  = water_stats.get("top_crop", "—")
    saved_mm  = water_stats.get("water_saved_mm", float("nan"))
    saved_pct = water_stats.get("water_saved_pct", float("nan"))
    paddy_def = water_stats.get("paddy_water_deficit", float("nan"))
    top_def   = water_stats.get("top_water_deficit", float("nan"))

    col1.metric("Top Crop", str(top_crop).capitalize())
    col2.metric(
        "Water Deficit (mm)",
        f"{top_def:.0f} mm" if not np.isnan(top_def) else "—",
        delta=f"{-saved_mm:.0f} mm vs paddy" if not np.isnan(saved_mm) else None,
        delta_color="normal" if (not np.isnan(saved_mm) and saved_mm >= 0) else "inverse",
    )
    col3.metric(
        "Saving vs Paddy",
        f"{saved_pct:.1f}%" if not np.isnan(saved_pct) else "N/A",
    )

    if not np.isnan(paddy_def):
        if not np.isnan(saved_mm) and saved_mm < 0:
            st.info(
                f"ℹ️ **{str(top_crop).capitalize()}** uses **{abs(saved_mm):.0f} mm more** "
                f"water than paddy ({paddy_def:.0f} mm deficit). "
                "Consider increasing the water weight in the sidebar."
            )
        elif not np.isnan(saved_mm):
            st.success(
                f"✅ **{str(top_crop).capitalize()}** saves **{saved_mm:.0f} mm "
                f"({saved_pct:.1f}%)** of water compared to paddy "
                f"({paddy_def:.0f} mm deficit)."
            )

    # ---- Price synthetic note -------------------------------------------
    if price_synthetic:
        st.warning(
            "⚠️ Price forecasts use **synthetic data**. "
            "Replace `data/raw/prices.csv` with real Agmarknet data and set "
            "`use_synthetic.prices: false` in `config/config.yaml` for market-realistic results."
        )

    # ---- Land allocation -------------------------------------------------
    if allocation_df is not None and not allocation_df.empty:
        st.markdown("#### 🌾 Recommended Land Allocation")
        st.dataframe(
            allocation_df.style.format({
                "area_ha":       "{:.1f}",
                "area_fraction": "{:.1%}",
            }),
            use_container_width=True,
        )

    st.markdown(_DISCLAIMER)
