"""
explanation.py
--------------
Plain-language explanation card for the top recommendation.

Changes from v1
---------------
* Richer explanation: why this crop wins, and what it gives up (trade-offs).
* risk_label() helper exposed for use in the metric cards in main.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

_DISCLAIMER = (
    "> **Disclaimer:** Outputs are indicative estimates based on historical data "
    "and synthetic modelling. Consult local agricultural experts before making "
    "planting decisions."
)

_RISK_THRESHOLDS = [
    (0.00, 0.33, "Low",    "🟢", "#16a34a"),   # green
    (0.33, 0.67, "Medium", "🟡", "#d97706"),   # amber
    (0.67, 1.01, "High",   "🔴", "#dc2626"),   # red
]


# ---------------------------------------------------------------------------
# Public helpers (also imported by main.py for metric cards)
# ---------------------------------------------------------------------------

def risk_label(score: float) -> tuple[str, str, str]:
    """Return (text, emoji, hex_colour) for a normalised risk score.

    Parameters
    ----------
    score: Risk score in [0, 1].

    Returns
    -------
    tuple of (label_text, emoji, hex_colour)
    """
    for lo, hi, label, emoji, colour in _RISK_THRESHOLDS:
        if lo <= score < hi:
            return label, emoji, colour
    return "High", "🔴", "#dc2626"


def _tradeoff_sentence(
    ranked_df: pd.DataFrame,
    top_crop: str,
) -> str:
    """Describe what the top crop gives up vs the runner-up.

    Parameters
    ----------
    ranked_df:  Full ranked DataFrame.
    top_crop:   Name of rank-1 crop.

    Returns
    -------
    str  One sentence about the main trade-off, or empty string if only 1 crop.
    """
    if len(ranked_df) < 2:
        return ""

    top    = ranked_df.iloc[0]
    second = ranked_df.iloc[1]
    s_crop = str(second["crop"]).capitalize()

    profit_delta = top["profit_inr_ha"] - second["profit_inr_ha"]
    water_delta  = top["water_deficit"] - second["water_deficit"]

    parts = []
    if profit_delta > 0:
        parts.append(
            f"earns **₹{profit_delta:,.0f}/ha more** than {s_crop}"
        )
    elif profit_delta < 0:
        parts.append(
            f"earns **₹{abs(profit_delta):,.0f}/ha less** than {s_crop}"
        )

    if water_delta > 5:
        parts.append(
            f"needs **{water_delta:.0f} mm more** irrigation water"
        )
    elif water_delta < -5:
        parts.append(
            f"uses **{abs(water_delta):.0f} mm less** water"
        )

    if not parts:
        return ""

    joined = " but ".join(parts)
    return f"Compared to the next-best option ({s_crop}), it {joined}."


def _water_sentence(water_deficit: float, paddy_def: float | None) -> str:
    if water_deficit <= 0:
        base = "It needs **no supplemental irrigation** under this scenario."
    else:
        base = (
            f"It has an estimated water deficit of **{water_deficit:.0f} mm** — "
            f"roughly **{water_deficit / 25.4:.1f} inches** of extra water needed."
        )

    if paddy_def is not None and paddy_def > 0:
        saved = paddy_def - water_deficit
        if saved > 5:
            base += (
                f" That is **{saved:.0f} mm ({saved / paddy_def * 100:.0f}%) less** "
                "than paddy — a significant water saving."
            )
        elif saved < -5:
            base += (
                f" That is **{abs(saved):.0f} mm more** than paddy. "
                "Adjust the water weight slider if irrigation is limited."
            )
    return base


def generate_explanation(
    ranked_df: pd.DataFrame,
    water_stats: dict,
) -> str:
    """Generate a plain-language explanation for the top crop.

    Parameters
    ----------
    ranked_df:   Output of weighted_ranker.rank_crops().
    water_stats: Output of weighted_ranker.water_saved_vs_paddy().

    Returns
    -------
    str  Markdown-formatted multi-paragraph explanation.
    """
    top       = ranked_df.iloc[0]
    crop      = str(top["crop"]).capitalize()
    season    = str(top.get("season", "")).capitalize()
    profit    = float(top["profit_inr_ha"])
    water     = float(top["water_deficit"])
    risk_sc   = float(top["risk_score"])

    r_label, r_emoji, _ = risk_label(risk_sc)
    paddy_def = water_stats.get("paddy_water_deficit")

    profit_lakh  = profit / 1e5
    profit_acre  = profit / 2.47

    why_sentence = (
        f"**{crop}** ({season.lower()}) scores highest under your current "
        f"objective settings — it offers the best balance of profit, "
        f"water use, and climate stability."
    )
    profit_sentence = (
        f"Expected profit is **₹{profit_lakh:.2f} lakh/ha** "
        f"(₹{profit:,.0f}/ha · ₹{profit_acre:,.0f}/acre)."
    )
    water_sentence  = _water_sentence(water, paddy_def)
    risk_sentence   = (
        f"{r_emoji} Climate risk is **{r_label.lower()}** "
        f"(score {risk_sc:.2f}/1.00) — based on historical yield variability "
        f"and how often yields dropped in dry years."
    )
    tradeoff = _tradeoff_sentence(ranked_df, str(top["crop"]))

    parts = [why_sentence, profit_sentence, water_sentence, risk_sentence]
    if tradeoff:
        parts.append(tradeoff)

    return "\n\n".join(parts)


def render_explanation_tab(
    ranked_df: pd.DataFrame,
    water_stats: dict,
) -> None:
    """Render the explanation card.

    Parameters
    ----------
    ranked_df:   Output of weighted_ranker.rank_crops().
    water_stats: Output of weighted_ranker.water_saved_vs_paddy().
    """
    st.subheader("📋 Why This Crop?")
    st.caption(
        "A plain-language explanation of the top recommendation, "
        "what it offers, and what it trades off."
    )

    explanation = generate_explanation(ranked_df, water_stats)
    st.info(explanation)

    st.markdown(_DISCLAIMER)
