"""
Reusable metric / KPI card components rendered via st.markdown + HTML/CSS.
"""

import streamlit as st
from typing import Optional


def _delta_arrow(delta: float, invert: bool = False) -> str:
    """Green-up / red-down (or inverted for loss-type metrics)."""
    if delta == 0:
        return '<span style="color:#888;">—</span>'
    positive_good = not invert
    is_positive = delta > 0
    color = "#00C853" if (is_positive == positive_good) else "#FF5252"
    arrow = "▲" if is_positive else "▼"
    return f'<span style="color:{color};font-size:0.75rem;">{arrow} {abs(delta):.2f}</span>'


def metric_card(
    label: str,
    value: str,
    delta: Optional[float] = None,
    color: str = "#4ECDC4",
    icon: str = "⚡",
    invert_delta: bool = False,
    unit: str = "",
) -> None:
    """Render a single KPI card with gradient accent."""
    delta_html = _delta_arrow(delta, invert_delta) if delta is not None else ""
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #1a1a2e 60%, {color}22);
        border-left: 4px solid {color};
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 8px;
    ">
        <div style="color:#999;font-size:0.75rem;text-transform:uppercase;letter-spacing:1px;">
            {icon} {label}
        </div>
        <div style="font-size:1.6rem;font-weight:700;color:#eee;margin:4px 0;">
            {value} <span style="font-size:0.85rem;color:#aaa;">{unit}</span>
        </div>
        <div>{delta_html}</div>
    </div>
    """, unsafe_allow_html=True)


def status_badge(status: str) -> None:
    """Render a coloured status badge (SECURE / ALERT / EMERGENCY / CRITICAL)."""
    colors = {
        "SECURE":    ("#00C853", "#00C85322"),
        "ALERT":     ("#FFD600", "#FFD60022"),
        "EMERGENCY": ("#FF9100", "#FF910022"),
        "CRITICAL":  ("#FF1744", "#FF174422"),
        "UNKNOWN":   ("#888",    "#88888822"),
    }
    fg, bg = colors.get(status.upper(), colors["UNKNOWN"])
    st.markdown(f"""
    <div style="
        display:inline-block;
        background:{bg};
        border:1px solid {fg};
        border-radius:20px;
        padding:4px 16px;
        color:{fg};
        font-weight:700;
        font-size:0.85rem;
        letter-spacing:1px;
    ">{status.upper()}</div>
    """, unsafe_allow_html=True)


def small_stat(label: str, value: str, color: str = "#aaa") -> None:
    """Compact inline stat."""
    st.markdown(f"""
    <div style="margin-bottom:4px;">
        <span style="color:#777;font-size:0.7rem;text-transform:uppercase;">{label}</span><br/>
        <span style="color:{color};font-size:1.1rem;font-weight:600;">{value}</span>
    </div>
    """, unsafe_allow_html=True)
