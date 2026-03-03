"""
DER-specific card / summary components.
"""

import streamlit as st
from typing import Any, Dict, List, Optional


DER_ICONS = {
    "solar_pv": "☀️",
    "wind": "🌬️",
    "bess": "🔋",
    "ev_charging": "⚡",
    "demand_response": "📉",
}

DER_COLORS = {
    "solar_pv": "#FFD700",
    "wind": "#00BFFF",
    "bess": "#00C853",
    "ev_charging": "#AA00FF",
    "demand_response": "#FF6D00",
}

DER_LABELS = {
    "solar_pv": "Solar PV",
    "wind": "Wind",
    "bess": "Battery (BESS)",
    "ev_charging": "EV Charging",
    "demand_response": "Demand Response",
}


def der_state_card(ds: Dict) -> None:
    """Render a single DER unit card."""
    dt = ds.get("der_type", "").lower()
    icon = DER_ICONS.get(dt, "⚙️")
    color = DER_COLORS.get(dt, "#888")
    name = ds.get("name", "Unknown")
    output = ds.get("current_output_mw", 0)
    cap = ds.get("capacity_mw", 1)
    pct = (output / cap * 100) if cap else 0

    extras = []
    soc = ds.get("soc")
    if soc is not None:
        soc_pct = soc * 100 if soc <= 1.0 else soc
        extras.append(f"SOC: {soc_pct:.1f}%")
    util = ds.get("utilization")
    if util is not None:
        extras.append(f"Util: {util:.0%}")
    curt = ds.get("curtailment_level")
    if curt is not None:
        extras.append(f"Curtail: {curt:.0%}")
    ext_str = " | ".join(extras) if extras else ""

    # progress bar colour
    bar_color = color
    bar_pct = min(pct, 100)

    st.markdown(f"""
    <div style="
        background:#1a1a2e;
        border-left:4px solid {color};
        border-radius:10px;
        padding:12px 16px;
        margin-bottom:8px;
    ">
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <span style="font-size:1rem;font-weight:600;color:#eee;">{icon} {name}</span>
            <span style="color:{color};font-weight:700;font-size:0.9rem;">Bus {ds.get('bus','?')}</span>
        </div>
        <div style="margin:6px 0;">
            <div style="background:#333;border-radius:4px;height:6px;overflow:hidden;">
                <div style="width:{bar_pct:.0f}%;height:100%;background:{bar_color};border-radius:4px;"></div>
            </div>
        </div>
        <div style="display:flex;justify-content:space-between;color:#aaa;font-size:0.75rem;">
            <span>{output:.2f} / {cap:.1f} MW ({pct:.0f}%)</span>
            <span>{ext_str}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def der_summary_row(der_summary: Dict) -> None:
    """Render a horizontal summary of all DER types."""
    cols = st.columns(5)
    order = ["solar", "wind", "battery", "ev_charger", "demand_response"]
    type_map = {
        "solar": "solar_pv",
        "wind": "wind",
        "battery": "bess",
        "ev_charger": "ev_charging",
        "demand_response": "demand_response",
    }
    for i, key in enumerate(order):
        data = der_summary.get(key, {})
        dt = type_map.get(key, key)
        icon = DER_ICONS.get(dt, "⚙️")
        color = DER_COLORS.get(dt, "#888")
        label = DER_LABELS.get(dt, key.title())
        count = data.get("unit_count", 0)
        cap = data.get("total_capacity_mw", data.get("capacity_mwh", data.get("max_power_mw", 0)))
        output = data.get("current_output_mw", data.get("current_power_mw", 0))
        with cols[i]:
            st.markdown(f"""
            <div style="background:#1a1a2e;border-radius:10px;padding:12px;text-align:center;border-top:3px solid {color};">
                <div style="font-size:1.5rem;">{icon}</div>
                <div style="color:#eee;font-weight:600;font-size:0.85rem;">{label}</div>
                <div style="color:{color};font-size:1.2rem;font-weight:700;">{count} units</div>
                <div style="color:#aaa;font-size:0.7rem;">{output:.1f} / {cap:.1f} MW</div>
            </div>
            """, unsafe_allow_html=True)
