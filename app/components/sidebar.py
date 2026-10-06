"""
sidebar.py
----------
Streamlit sidebar — collects scenario inputs and returns a params dict.

Changes from v1
---------------
* "Recalculate" button pinned at the very top; auto-runs on first load
  via st.session_state.
* Weight presets (Balanced / Maximise profit / Save water / Play safe)
  replace the three top-level sliders.  Individual sliders live in an
  "Advanced settings" expander, behind a "Use custom weights" checkbox.
* Rainfall options renamed to Normal / Dry (-10 %) / Drought (-30 %) and
  the estimated seasonal rainfall in mm is shown next to the selection.
* All inputs carry plain-language help text.

Fix in this version
-------------------
The previous file had an ``else:`` attached to a ``with ...expander()`` block,
which is a SyntaxError (a ``with`` block cannot have an ``else``).  Streamlit
also cannot tell whether an expander is open.  It is replaced by an explicit
"Use custom weights" checkbox and a normal if/else.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SEASONS = ["kharif", "rabi", "summer"]
_YEAR_MIN = 2010
_YEAR_MAX = 2023

# Renamed rainfall options -> multiplier
_RAINFALL_SCENARIOS: dict[str, float] = {
    "Normal":          1.00,
    "Dry (−10%)":      0.90,
    "Drought (−30%)":  0.70,
}

# Approximate mean seasonal rainfall (mm) used to show context to the user.
# Values are Tamil Nadu / Tiruvallur district averages.
_SEASON_MEAN_RAIN: dict[str, float] = {
    "kharif": 850,
    "rabi":   280,
    "summer": 120,
}

# Weight presets — (profit, water, risk)
_WEIGHT_PRESETS: dict[str, tuple[float, float, float]] = {
    "Balanced":          (0.40, 0.30, 0.30),
    "Maximise profit":   (0.70, 0.20, 0.10),
    "Save water":        (0.20, 0.65, 0.15),
    "Play safe":         (0.20, 0.30, 0.50),
}

_PRESET_ICONS: dict[str, str] = {
    "Balanced":        "⚖️",
    "Maximise profit": "💰",
    "Save water":      "💧",
    "Play safe":       "🛡️",
}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _normalise(p: float, w: float, r: float) -> tuple[float, float, float]:
    total = p + w + r
    if total < 0.01:
        return 1 / 3, 1 / 3, 1 / 3
    return p / total, w / total, r / total


def _set_slider_state(preset: str) -> None:
    """Copy a preset's weights into the advanced-slider session-state keys."""
    p, w, r = _WEIGHT_PRESETS[preset]
    st.session_state["adv_w_profit"] = p
    st.session_state["adv_w_water"] = w
    st.session_state["adv_w_risk"] = r


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_sidebar(cfg: dict[str, Any]) -> dict[str, Any]:
    """Render sidebar widgets and return a params dict.

    On first load the default params are returned immediately so the
    analysis runs without requiring a button click.  Subsequent runs are
    triggered by the "Recalculate" button at the top of the sidebar.

    Parameters
    ----------
    cfg: Parsed configuration dictionary.

    Returns
    -------
    dict with keys: season, year, irrigation_mm, rainfall_scenario_factor,
    rainfall_scenario_label, est_rain_mm, weights, active_preset,
    run_allocator, total_acreage, water_budget.
    """
    sb = st.sidebar

    # ---- Recalculate button (pinned top) ---------------------------------
    sb.markdown("## 🌾 Scenario Settings")
    recalc = sb.button(
        "🔄 Recalculate",
        use_container_width=True,
        help="Re-run the analysis with the current settings.",
        type="primary",
    )

    # Track whether we have ever run; if not, force an initial run.
    if "has_run" not in st.session_state:
        st.session_state["has_run"] = False

    sb.markdown("---")

    # ---- Season ----------------------------------------------------------
    season = sb.selectbox(
        "Season",
        options=_SEASONS,
        index=0,
        help=(
            "The growing season to evaluate. "
            "Kharif = Jun–Sep (monsoon), Rabi = Oct–Jan (winter), "
            "Summer = Feb–May."
        ),
    )

    # ---- Year ------------------------------------------------------------
    year = sb.slider(
        "Reference year",
        min_value=_YEAR_MIN,
        max_value=_YEAR_MAX,
        value=_YEAR_MAX,
        step=1,
        help=(
            "Year used for price and cost model inference. "
            "Use a recent year for current conditions."
        ),
    )

    # ---- Irrigation ------------------------------------------------------
    irrigation_mm = sb.number_input(
        "Irrigation water available (mm)",
        min_value=0,
        max_value=2000,
        value=100,
        step=10,
        help=(
            "Extra water you can provide beyond rainfall. "
            "100 mm ≈ 4 inches. A bore-well typically supplies 150–300 mm/season."
        ),
    )

    # ---- Rainfall scenario -----------------------------------------------
    mean_rain = _SEASON_MEAN_RAIN.get(season, 500)

    scenario_label = sb.radio(
        "Rainfall scenario",
        options=list(_RAINFALL_SCENARIOS.keys()),
        index=0,
        help=(
            "How much rain do you expect this season? "
            f"Normal for {season} ≈ {mean_rain:.0f} mm. "
            "Dry years typically receive 10–30 % less."
        ),
    )
    rf_factor = _RAINFALL_SCENARIOS[scenario_label]
    est_rain_mm = mean_rain * rf_factor
    sb.caption(
        f"Estimated {season} rainfall: **{est_rain_mm:.0f} mm** "
        f"({est_rain_mm / 25.4:.1f} inches)"
    )

    # ---- Objective weights — presets + advanced -------------------------
    sb.markdown("---")
    sb.subheader("What matters most?")
    sb.caption("Pick a preset or fine-tune in Advanced settings below.")

    presets = list(_WEIGHT_PRESETS.keys())

    # Initialise session state once.
    if "active_preset" not in st.session_state:
        st.session_state["active_preset"] = "Balanced"
    if "adv_w_profit" not in st.session_state:
        _set_slider_state(st.session_state["active_preset"])

    # 2 x 2 grid of preset buttons.  Clicking one also resets the advanced
    # sliders to that preset's values (done BEFORE the sliders are created,
    # which is the only time Streamlit allows it).
    col_a, col_b = sb.columns(2)
    for col, preset in zip([col_a, col_b, col_a, col_b], presets):
        icon = _PRESET_ICONS[preset]
        is_active = st.session_state["active_preset"] == preset
        btn_type = "primary" if is_active else "secondary"
        if col.button(
            f"{icon} {preset}",
            use_container_width=True,
            type=btn_type,
            key=f"preset_{preset}",
        ):
            st.session_state["active_preset"] = preset
            _set_slider_state(preset)

    active_preset = st.session_state["active_preset"]
    p_def, w_def, r_def = _WEIGHT_PRESETS[active_preset]

    # Advanced sliders.  FIX: no "else:" on a `with` block.  An explicit
    # checkbox decides whether the custom sliders or the preset is used.
    with sb.expander("⚙️ Advanced weight settings"):
        use_custom = st.checkbox(
            "Use custom weights",
            value=False,
            key="use_custom_weights",
            help="Tick to set the three weights yourself instead of using the preset.",
        )
        if use_custom:
            st.caption(
                "Fine-tune the three objectives. "
                "Weights are normalised automatically — they don't need to sum to 1."
            )
            w_profit_raw = st.slider(
                "💰 Profit weight",
                0.0, 1.0,
                step=0.05,
                key="adv_w_profit",
                help="How important is maximising income per hectare.",
            )
            w_water_raw = st.slider(
                "💧 Water weight",
                0.0, 1.0,
                step=0.05,
                key="adv_w_water",
                help="How important is minimising irrigation water use.",
            )
            w_risk_raw = st.slider(
                "🛡️ Risk weight",
                0.0, 1.0,
                step=0.05,
                key="adv_w_risk",
                help="How important is avoiding yield failure in bad years.",
            )
            w_profit, w_water, w_risk = _normalise(w_profit_raw, w_water_raw, w_risk_raw)
            st.caption(
                f"Effective weights → profit **{w_profit:.2f}** · "
                f"water **{w_water:.2f}** · risk **{w_risk:.2f}**"
            )
        else:
            # Use the preset values.
            w_profit, w_water, w_risk = _normalise(p_def, w_def, r_def)
            st.caption(
                f"Using the **{active_preset}** preset → profit **{w_profit:.2f}** · "
                f"water **{w_water:.2f}** · risk **{w_risk:.2f}**"
            )

    # ---- Land allocator (optional) ---------------------------------------
    sb.markdown("---")
    with sb.expander("🌾 Land allocation (optional)"):
        st.caption(
            "Allocate your total farm area across the recommended crops "
            "to maximise profit within a water budget."
        )
        run_allocator = st.checkbox(
            "Enable land allocator",
            value=False,
            help="Solves a simple optimisation to split area across crops.",
        )
        total_acreage = None
        water_budget = None
        if run_allocator:
            total_acreage = st.number_input(
                "Total farm area (ha)",
                min_value=1, max_value=10000, value=100, step=10,
                help="Total cultivable area you want to plan for.",
            )
            water_budget = st.number_input(
                "Total water budget (mm × ha)",
                min_value=100, max_value=500000, value=50000, step=1000,
                help=(
                    "Total water available for the whole farm this season. "
                    "Example: 100 ha × 500 mm = 50,000 mm·ha."
                ),
            )

    # ---- Decide whether to run ------------------------------------------
    # Run if: first load (has_run=False) OR recalculate clicked
    should_run = (not st.session_state["has_run"]) or recalc
    if should_run:
        st.session_state["has_run"] = True

    if not should_run and "last_params" in st.session_state:
        # Keep the page populated with the last results until Recalculate.
        return st.session_state["last_params"]

    params = {
        "season":                   season,
        "year":                     int(year),
        "irrigation_mm":            float(irrigation_mm),
        "rainfall_scenario_factor": rf_factor,
        "rainfall_scenario_label":  scenario_label,
        "est_rain_mm":              est_rain_mm,
        "weights": {
            "profit": round(w_profit, 4),
            "water":  round(w_water,  4),
            "risk":   round(w_risk,   4),
        },
        "active_preset":  active_preset,
        "run_allocator":  run_allocator,
        "total_acreage":  total_acreage,
        "water_budget":   water_budget,
    }
    st.session_state["last_params"] = params
    return params