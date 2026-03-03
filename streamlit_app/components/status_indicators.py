"""
Status indicator helpers – stability badges, convergence dots, etc.
"""

import streamlit as st


def stability_indicator(is_secure: bool, num_violations: int, num_overloads: int) -> str:
    """Derive a stability status string from power flow results."""
    if not is_secure:
        if num_violations > 5 or num_overloads > 3:
            return "CRITICAL"
        if num_violations > 2 or num_overloads > 1:
            return "EMERGENCY"
        return "ALERT"
    return "SECURE"


def convergence_dot(converged: bool) -> None:
    """Small coloured dot for convergence status."""
    color = "#00C853" if converged else "#FF5252"
    label = "Converged" if converged else "Not Converged"
    st.markdown(f"""
    <span style="display:inline-flex;align-items:center;gap:6px;">
        <span style="width:10px;height:10px;border-radius:50%;background:{color};display:inline-block;"></span>
        <span style="color:{color};font-size:0.8rem;font-weight:600;">{label}</span>
    </span>
    """, unsafe_allow_html=True)


def sim_clock(hour: float, step: int) -> None:
    """Display simulated time-of-day clock."""
    h = int(hour)
    m = int((hour - h) * 60)
    ampm = "AM" if h < 12 else "PM"
    h12 = h % 12 or 12
    st.markdown(f"""
    <div style="text-align:center;padding:8px;">
        <div style="color:#777;font-size:0.7rem;text-transform:uppercase;">Simulation Time</div>
        <div style="color:#eee;font-size:1.4rem;font-weight:700;">{h12}:{m:02d} {ampm}</div>
        <div style="color:#555;font-size:0.7rem;">Step {step}</div>
    </div>
    """, unsafe_allow_html=True)
