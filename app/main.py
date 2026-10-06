"""
main.py
-------
Streamlit entry point for the Water-Smart Crop Planner dashboard.

Run with:
    streamlit run app/main.py

UI improvements (v2)
--------------------
1. Auto-runs on page load with default settings; Recalculate button pinned
   at top of sidebar.
2. Four metric cards above tabs: top crop, profit (INR/acre + /ha), water
   saved vs paddy, risk level (colour-coded).
3. Sidebar: weight presets + Advanced expander for fine-grained sliders.
4. Help text on every input; renamed rainfall options; live mm estimate.
5. Compact single-line synthetic-data banner with detail expander.
6. Pareto tab: Okabe-Ito palette, bubble size legend, How-to-read expander.
7. Explanation tab: richer text — why this crop, what it trades off.
8. About & Method tab: data sources, assumptions, limitations, disclaimer.
9. Download buttons: CSV table + plain-text summary.
10. .streamlit/config.toml green/blue theme, wide layout.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config_loader import load_config, any_synthetic, synthetic_dataset_names
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Page config — MUST be the first Streamlit call
st.set_page_config(
    page_title="Water-Smart Crop Planner",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Cached resource loaders
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading configuration…")
def _load_cfg() -> dict:
    return load_config()


@st.cache_resource(show_spinner="Loading feature data…")
def _load_features() -> pd.DataFrame:
    cfg  = _load_cfg()
    path = Path(cfg["data"]["processed_dir"]) / "features.parquet"
    if not path.exists():
        from src.data.build_features import build_features  # noqa: PLC0415
        return build_features(cfg)
    return pd.read_parquet(path)


@st.cache_resource(show_spinner="Loading yield model…")
def _load_yield_bundle() -> dict:
    cfg  = _load_cfg()
    path = Path(cfg["outputs"]["models_dir"]) / "yield_model.pkl"
    if not path.exists():
        from src.models.yield_model import train_yield_models  # noqa: PLC0415
        res = train_yield_models(cfg)
        return {"model": res["best_model"], "encoder": res["encoder"], "name": res["best_name"]}
    from src.models.yield_model import load_yield_model  # noqa: PLC0415
    return load_yield_model(cfg)


@st.cache_resource(show_spinner="Loading price model…")
def _load_price_bundle() -> dict:
    cfg  = _load_cfg()
    path = Path(cfg["outputs"]["models_dir"]) / "price_model.pkl"
    if not path.exists():
        from src.models.price_model import train_price_model  # noqa: PLC0415
        return train_price_model(cfg)
    from src.models.price_model import load_price_model  # noqa: PLC0415
    return load_price_model(cfg)


@st.cache_resource(show_spinner="Computing risk scores…")
def _load_risk_df() -> pd.DataFrame:
    cfg  = _load_cfg()
    path = Path(cfg["data"]["processed_dir"]) / "risk_scores.parquet"
    if not path.exists():
        from src.risk.risk_scorer import compute_risk_scores  # noqa: PLC0415
        return compute_risk_scores(cfg)
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def _run_pipeline(
    params: dict,
    cfg: dict,
    features_df: pd.DataFrame,
    yield_bundle: dict,
    price_bundle: dict,
    risk_df: pd.DataFrame,
) -> dict:
    """Run objectives → NSGA-II → ranker and return results dict."""
    from src.optimization.objectives import compute_objectives_batch  # noqa: PLC0415
    from src.optimization.nsga2_optimizer import run_nsga2             # noqa: PLC0415
    from src.optimization.weighted_ranker import rank_crops, water_saved_vs_paddy  # noqa: PLC0415
    from src.models.price_model import predict_price                   # noqa: PLC0415

    season  = params["season"]
    year    = params["year"]
    irr     = params["irrigation_mm"]
    rf_fac  = params["rainfall_scenario_factor"]
    weights = params["weights"]

    crops  = list(features_df["crop"].unique())

    obj_df = compute_objectives_batch(
        crops=crops, season=season, year=year,
        irrigation_mm=irr, rainfall_scenario_factor=rf_fac,
        features_df=features_df, risk_df=risk_df,
        yield_bundle=yield_bundle, price_bundle=price_bundle,
    )
    pareto_df   = run_nsga2(obj_df, cfg)
    ranked_df   = rank_crops(pareto_df, weights)
    water_stats = water_saved_vs_paddy(ranked_df)

    price_intervals = []
    for _, row in pareto_df.iterrows():
        pi = predict_price(price_bundle, row["crop"], season, year)
        price_intervals.append({"crop": row["crop"], "p10": pi.p10,
                                 "p50": pi.p50, "p90": pi.p90})

    allocation_df = None
    if params.get("run_allocator") and params.get("total_acreage"):
        try:
            from src.optimization.land_allocator import allocate_land  # noqa: PLC0415
            allocation_df = allocate_land(
                pareto_df=pareto_df,
                total_acreage=float(params["total_acreage"]),
                water_budget=float(params["water_budget"]),
            )
        except Exception as exc:
            st.warning(f"Land allocator: {exc}")

    return {
        "obj_df":          obj_df,
        "pareto_df":       pareto_df,
        "ranked_df":       ranked_df,
        "water_stats":     water_stats,
        "allocation_df":   allocation_df,
        "price_intervals": price_intervals,
    }


# ---------------------------------------------------------------------------
# Metric cards row
# ---------------------------------------------------------------------------

def _render_metric_cards(ranked_df: pd.DataFrame, water_stats: dict) -> None:
    """Render the four summary metric cards above the tabs.

    Parameters
    ----------
    ranked_df:   Ranked crops DataFrame.
    water_stats: Water savings dict.
    """
    from app.components.explanation import risk_label  # noqa: PLC0415

    top         = ranked_df.iloc[0]
    top_crop    = str(top["crop"]).capitalize()
    profit_ha   = float(top["profit_inr_ha"])
    profit_acre = profit_ha / 2.47
    risk_sc     = float(top["risk_score"])
    r_label, r_emoji, r_colour = risk_label(risk_sc)

    saved_mm    = water_stats.get("water_saved_mm", float("nan"))
    saved_pct   = water_stats.get("water_saved_pct", float("nan"))

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        label="🌾 Top Recommendation",
        value=top_crop,
        help="The crop ranked #1 under your current objective weights.",
    )
    c2.metric(
        label="💰 Expected Profit",
        value=f"₹{profit_acre:,.0f}/acre",
        delta=f"₹{profit_ha:,.0f}/ha",
        delta_color="off",
        help="Estimated profit = (yield × price) − input cost. Indicative only.",
    )

    if not np.isnan(saved_mm):
        saved_str = f"{saved_mm:.0f} mm saved vs paddy"
        delta_col = "normal" if saved_mm >= 0 else "inverse"
        c3.metric(
            label="💧 Water Saving",
            value=f"{saved_pct:.0f}%" if not np.isnan(saved_pct) else "—",
            delta=saved_str,
            delta_color=delta_col,
            help="Water deficit of top crop vs paddy baseline.",
        )
    else:
        c3.metric(
            label="💧 Water Saving",
            value="—",
            help="Paddy not in current Pareto set.",
        )

    # Risk card — coloured via HTML
    c4.markdown(
        f"""
        <div style="background:{r_colour}18; border-left:4px solid {r_colour};
                    padding:10px 14px; border-radius:6px; line-height:1.4;">
            <div style="font-size:0.78rem; color:#555; margin-bottom:2px;">
                ⚠️ Risk Level
            </div>
            <div style="font-size:1.55rem; font-weight:700; color:{r_colour};">
                {r_emoji} {r_label}
            </div>
            <div style="font-size:0.78rem; color:#555;">
                score {risk_sc:.2f} / 1.00
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# About & Method tab
# ---------------------------------------------------------------------------

def _render_about_tab(cfg: dict) -> None:
    st.subheader("About This Tool")
    st.markdown(
        "The **Water-Smart Crop Planner** is a decision-support system for "
        "smallholder farmers and agricultural extension officers in water-stressed "
        "districts. It scores candidate crops on three objectives simultaneously — "
        "profit, water use, and climate risk — and shows the trade-offs as a "
        "Pareto front."
    )

    st.markdown("### 🗂️ Data Sources")
    st.markdown(
        """
| Dataset | Source | Notes |
|---------|--------|-------|
| Daily weather | [NASA POWER Community Ag](https://power.larc.nasa.gov/) | Auto-fetched; cached locally |
| Crop yield | Dept. of Agriculture, Tamil Nadu / [data.gov.in](https://data.gov.in) | District-level annual |
| Wholesale prices | [Agmarknet](https://agmarknet.gov.in) / Koyambedu mandi | Monthly modal price |
| Crop water needs | FAO-56 + TNAU reference ET₀ | Seasonal water requirement |
| Input costs | TNAU Cost-of-Cultivation Bulletins | Annual; nominal terms |
        """
    )

    st.markdown("### ⚙️ Method Summary")
    st.markdown(
        """
1. **Yield model** — Random Forest / XGBoost trained on seasonal weather features
   (rainfall, temperature, heat-stress days, dry-spell length) with a time-aware
   train/test split to prevent leakage.
2. **Price model** — Three quantile gradient-boosted regressors giving P10, P50,
   and P90 price intervals.
3. **Risk scoring** — Coefficient of variation of historical yield + probability of
   yield failure in dry years (rainfall below the 30th percentile).
4. **NSGA-II optimisation** — Multi-objective evolutionary algorithm finds the set
   of crops that cannot be improved on one objective without worsening another
   (Pareto front).
5. **Weighted ranking** — User-supplied weights convert the Pareto set to a single
   ranked list.
        """
    )

    st.markdown("### ⚠️ Assumptions & Limitations")
    st.markdown(
        """
- **District-level aggregates** — spatial variability within Tiruvallur is not
  captured; field-level conditions may differ significantly.
- **Synthetic data** — when real data files are absent the tool generates
  statistically plausible but artificial data. Models trained on it are for
  development only.
- **2019 district split** — Tiruvallur was carved from Kancheepuram in 2019;
  enable `merge_kancheepuram: true` in `config/config.yaml` to extend the
  yield series.
- **Price forecasts** are short-horizon and do not model government procurement
  prices, supply shocks, or seasonal demand patterns.
- **Irrigation model** — user-supplied availability only; groundwater depletion,
  canal schedules, and distribution losses are not modelled.
- **Climate change** — models are trained on 2010–2023 historical data and do
  not extrapolate to future climate trajectories.
        """
    )

    st.markdown("### 📄 Disclaimer")
    st.warning(
        "All outputs from this system are **indicative estimates** based on "
        "historical data and synthetic modelling. They do not constitute "
        "agronomic, financial, or legal advice. Consult local agricultural "
        "experts before making planting decisions."
    )

    st.markdown("### 🔧 Configuration")
    st.markdown(
        f"- **District:** {cfg['district']['name']}, {cfg['district']['state']} "
        f"({cfg['district']['lat']}° N, {cfg['district']['lon']}° E)  \n"
        f"- **Study period:** {cfg['study_period']['start_year']}–"
        f"{cfg['study_period']['end_year']}  \n"
        f"- **Candidates:** {', '.join(cfg['crops']['candidates'])}  \n"
        f"- **Min data years:** {cfg['crops']['min_data_years']}  \n"
        f"- **Kancheepuram merge:** {cfg['data'].get('merge_kancheepuram', False)}"
    )


# ---------------------------------------------------------------------------
# Backtest tab
# ---------------------------------------------------------------------------

def _render_backtest_tab() -> None:
    st.subheader("📊 Walk-Forward Backtest")
    st.caption(
        "For each test year, the model was trained on all prior years only, "
        "then evaluated against the actual crop data for that year."
    )
    backtest_path = Path("outputs/backtest_results.csv")
    if backtest_path.exists():
        bt = pd.read_csv(backtest_path)
        if bt.empty:
            st.info(
                "The backtest file exists but has no rows. "
                "Run `python src/data/backtest.py` to populate it."
            )
        else:
            # Summary metrics
            hit_rate = bt["hit"].mean() * 100 if "hit" in bt.columns else float("nan")
            mean_pdelta = bt["profit_delta_inr_ha"].mean() if "profit_delta_inr_ha" in bt.columns else float("nan")
            bc1, bc2, bc3 = st.columns(3)
            bc1.metric("Test years", len(bt["year"].unique()) if "year" in bt.columns else "—")
            bc2.metric(
                "Hit rate",
                f"{hit_rate:.0f}%" if not np.isnan(hit_rate) else "—",
                help="% of years where top recommendation matched actual dominant crop.",
            )
            bc3.metric(
                "Mean profit delta",
                f"₹{mean_pdelta:,.0f}/ha" if not np.isnan(mean_pdelta) else "—",
                help="Realised profit of recommendation minus actual crop profit.",
            )

            st.dataframe(bt, use_container_width=True)

            csv = bt.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download backtest results (CSV)",
                data=csv,
                file_name="backtest_results.csv",
                mime="text/csv",
            )
    else:
        st.info(
            "Backtest results not yet generated. "
            "Run `python src/data/backtest.py` from the project root, "
            "then reload this page."
        )
    st.markdown(
        "> **Disclaimer:** Backtest results are on synthetic data and do not "
        "reflect real historical performance."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Main Streamlit application entry point."""
    cfg = _load_cfg()

    # ---- Compact synthetic-data banner (req 5) ---------------------------
    if any_synthetic(cfg):
        syn_names = synthetic_dataset_names(cfg)
        short_names = ", ".join(syn_names)
        st.warning(
            f"⚠️ Running on **synthetic data** ({short_names}). "
            "Results are for development only.",
            icon="⚠️",
        )
        with st.expander("ℹ️ What does 'synthetic data' mean?"):
            st.markdown(
                "The following datasets were generated by the synthetic data "
                f"generator rather than sourced from real observations: **{short_names}**.\n\n"
                "To switch to real data:\n"
                "1. Place your real CSV files in `data/raw/` "
                "(schemas in `data/raw/README.txt`).\n"
                "2. Set the corresponding `use_synthetic` flags to `false` in "
                "`config/config.yaml`.\n"
                "3. Re-run `python src/data/build_features.py` and retrain the models."
            )

    # ---- Header ----------------------------------------------------------
    st.title("🌾 Water-Smart Crop Planner")
    st.caption(
        f"**{cfg['district']['name']}, {cfg['district']['state']}** "
        f"({cfg['district']['lat']}° N, {cfg['district']['lon']}° E)  ·  "
        f"{cfg['study_period']['start_year']}–{cfg['study_period']['end_year']}"
    )

    # ---- Load resources --------------------------------------------------
    features_df  = _load_features()
    yield_bundle = _load_yield_bundle()
    price_bundle = _load_price_bundle()
    risk_df      = _load_risk_df()

    # ---- Sidebar (always returns params — auto-runs on first load) -------
    from app.components.sidebar import render_sidebar  # noqa: PLC0415
    params = render_sidebar(cfg)

    # ---- Run pipeline ----------------------------------------------------
    # params is always a dict now (sidebar auto-runs on first load)
    run_key = str(params)   # cache key: re-run only when params change
    if st.session_state.get("_last_run_key") != run_key:
        with st.spinner("Running optimisation…"):
            results = _run_pipeline(
                params=params,
                cfg=cfg,
                features_df=features_df,
                yield_bundle=yield_bundle,
                price_bundle=price_bundle,
                risk_df=risk_df,
            )
        st.session_state["results"]       = results
        st.session_state["_last_run_key"] = run_key
    else:
        results = st.session_state["results"]

    ranked_df   = results["ranked_df"]
    pareto_df   = results["pareto_df"]
    water_stats = results["water_stats"]

    # ---- Metric cards (req 2) --------------------------------------------
    _render_metric_cards(ranked_df, water_stats)
    st.markdown("---")

    # ---- Tabs (req 8: About & Method added) ------------------------------
    tab_pareto, tab_ranked, tab_why, tab_backtest, tab_about = st.tabs([
        "📈 Pareto Chart",
        "🏆 Ranked Table",
        "💡 Why This Crop?",
        "📊 Backtest",
        "ℹ️ About & Method",
    ])

    with tab_pareto:
        from app.components.pareto_chart import render_pareto_tab  # noqa: PLC0415
        render_pareto_tab(
            pareto_df=pareto_df,
            price_intervals=results["price_intervals"],
            season=params["season"],
            rainfall_label=params["rainfall_scenario_label"],
        )

    with tab_ranked:
        from app.components.ranked_table import render_ranked_table_tab  # noqa: PLC0415
        render_ranked_table_tab(
            ranked_df=ranked_df,
            water_stats=water_stats,
            allocation_df=results["allocation_df"],
            price_synthetic=price_bundle.get("synthetic", False),
            params=params,
        )

    with tab_why:
        from app.components.explanation import render_explanation_tab  # noqa: PLC0415
        render_explanation_tab(
            ranked_df=ranked_df,
            water_stats=water_stats,
        )

    with tab_backtest:
        _render_backtest_tab()

    with tab_about:
        _render_about_tab(cfg)

    # ---- Persist for potential external re-use ---------------------------
    st.session_state["params"] = params


if __name__ == "__main__":
    main()
