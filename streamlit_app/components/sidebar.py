"""
Shared sidebar controls for every page.
Renders Play / Pause / Step buttons, weather overrides, grid events,
and handles auto-step logic so simulation is dynamic on every page.
"""

import time
import streamlit as st
from streamlit_app.simulation.state_manager import (
    init_session, get_engine, step_simulation, get_current,
)


def render_sidebar() -> None:
    """Call this at the top of every page to get simulation controls."""
    init_session()

    with st.sidebar:
        st.markdown("## ⚡ DEMS Grid")
        st.markdown(
            '<div class="sidebar-section">Simulation</div>',
            unsafe_allow_html=True,
        )

        snap = get_current()
        step_num = snap.step
        sim_h = snap.sim_hour

        # ── Step / clock display ──────────────────────────────────────
        st.markdown(
            f'<div style="text-align:center;margin-bottom:8px;">'
            f'<span style="color:#4ECDC4;font-size:1.4rem;font-weight:700;">Step {step_num}</span>'
            f'<span style="color:#555;margin:0 8px;">|</span>'
            f'<span style="color:#FFE66D;font-size:1.1rem;">{int(sim_h):02d}:{int((sim_h%1)*60):02d}h</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── Play / Pause / Step buttons ──────────────────────────────
        col1, col2, col3 = st.columns(3)
        with col1:
            play_clicked = st.button(
                "▶ Play", use_container_width=True, key="sb_play",
                type="primary" if not st.session_state.get("running") else "secondary",
            )
        with col2:
            pause_clicked = st.button("⏸ Pause", use_container_width=True, key="sb_pause")
        with col3:
            step_clicked = st.button("⏭ Step", use_container_width=True, key="sb_step")

        if play_clicked:
            st.session_state["running"] = True
        if pause_clicked:
            st.session_state["running"] = False
        if step_clicked:
            st.session_state["running"] = False
            step_simulation()
            st.rerun()

        st.session_state["sim_speed_hours"] = st.slider(
            "Step size (hours)", 0.25, 4.0, 1.0, 0.25,
            help="Simulated hours per step", key="sb_dt",
        )
        st.session_state["refresh_rate_s"] = st.slider(
            "Refresh rate (s)", 1, 10, 2,
            help="Seconds between auto-steps when running", key="sb_rr",
        )

        # ── Weather Override ──────────────────────────────────────────
        st.markdown(
            '<div class="sidebar-section">Weather Override</div>',
            unsafe_allow_html=True,
        )
        weather_mode = st.radio(
            "Weather",
            ["Auto (24h profile)", "Manual"],
            horizontal=True,
            label_visibility="collapsed",
            key="sb_weather_mode",
        )
        irr_override = None
        ws_override = None
        if weather_mode == "Manual":
            irr_override = st.slider("Irradiance (W/m²)", 0, 1200, 600, 50, key="sb_irr")
            ws_override = st.slider("Wind Speed (m/s)", 0.0, 30.0, 8.0, 0.5, key="sb_ws")

        # ── Grid Events ──────────────────────────────────────────────
        st.markdown(
            '<div class="sidebar-section">Grid Events</div>',
            unsafe_allow_html=True,
        )
        event = st.selectbox(
            "Inject event",
            ["None", "Load spike Area A (+20%)", "Load drop Area B (-15%)", "Load spike Area C (+25%)"],
            label_visibility="collapsed",
            key="sb_event",
        )

        st.markdown("---")
        st.markdown(
            '<div class="sidebar-section">Mode</div>',
            unsafe_allow_html=True,
        )
        st.session_state["sim_mode"] = st.radio(
            "Simulation mode",
            ["embedded", "api"],
            horizontal=True,
            label_visibility="collapsed",
            key="sb_mode",
        )
        if st.session_state["sim_mode"] == "api":
            st.session_state["api_url"] = st.text_input(
                "API URL", st.session_state.get("api_url", "http://localhost:8000"),
                key="sb_api_url",
            )

    # ── Auto-step logic (runs AFTER sidebar renders) ──────────────────
    if st.session_state.get("running"):
        engine = get_engine()
        # Apply weather overrides
        if irr_override is not None:
            engine.set_irradiance(irr_override)
        if ws_override is not None:
            engine.set_wind_speed(ws_override)
        # Apply grid events
        if event and event != "None":
            if "Area A" in event and "+20%" in event:
                engine.scale_area_load("A", 1.20)
            elif "Area B" in event and "-15%" in event:
                engine.scale_area_load("B", 0.85)
            elif "Area C" in event and "+25%" in event:
                engine.scale_area_load("C", 1.25)
        step_simulation()
        rate = st.session_state.get("refresh_rate_s", 2)
        time.sleep(max(0.3, rate * 0.5))   # shorter sleep for responsiveness
        st.rerun()
