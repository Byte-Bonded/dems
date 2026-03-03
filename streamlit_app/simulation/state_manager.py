"""
State Manager – bridges the SimEngine with Streamlit's session_state.
Provides helper functions to initialise, step, and query simulation state.
"""

import streamlit as st
from typing import Optional
from .sim_engine import SimEngine, SimSnapshot


def init_session() -> None:
    """Initialise all session-state keys (idempotent)."""
    if "engine" not in st.session_state:
        st.session_state["engine"] = None      # will be created lazily
    if "running" not in st.session_state:
        st.session_state["running"] = False
    if "step_count" not in st.session_state:
        st.session_state["step_count"] = 0
    if "sim_speed_hours" not in st.session_state:
        st.session_state["sim_speed_hours"] = 1.0
    if "refresh_rate_s" not in st.session_state:
        st.session_state["refresh_rate_s"] = 2.0
    if "sim_mode" not in st.session_state:
        st.session_state["sim_mode"] = "embedded"
    if "api_url" not in st.session_state:
        st.session_state["api_url"] = "http://localhost:8000"


def get_engine() -> SimEngine:
    """Return the shared SimEngine (create on first call)."""
    if st.session_state.get("engine") is None:
        with st.spinner("Building 117-bus SuperGrid … (first load takes a few seconds)"):
            st.session_state["engine"] = SimEngine()
    return st.session_state["engine"]


def step_simulation() -> SimSnapshot:
    """Advance one simulation step and return the snapshot."""
    engine = get_engine()
    dt = st.session_state.get("sim_speed_hours", 1.0)
    snap = engine.step(dt_hours=dt)
    st.session_state["step_count"] = snap.step
    return snap


def get_current() -> SimSnapshot:
    """Return the latest snapshot without advancing."""
    engine = get_engine()
    return engine.current


def get_history(n: int = 100):
    engine = get_engine()
    return engine.get_history_list(n)
