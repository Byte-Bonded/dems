"""
DEMS Grid Dashboard — Main Entry Point
=======================================
A Streamlit-based real-time visualisation of the 117-bus Tri-Area SuperGrid
with 23 DER units, dynamic simulation, and Microgrid view.

Run with:
    cd dems/
    streamlit run streamlit_app/app.py
"""

import sys, os

# Ensure project root on path
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _root not in sys.path:
    sys.path.insert(0, _root)

import streamlit as st

# ── Page configuration (must be FIRST Streamlit call) ──────────────────────
st.set_page_config(
    page_title="DEMS Grid Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject global dark theme CSS ──────────────────────────────────────────
st.markdown("""
<style>
/* Base overrides */
[data-testid="stAppViewContainer"] {
    background: #0E1117;
}
[data-testid="stSidebar"] {
    background: #12131a;
    border-right: 1px solid #222;
}
[data-testid="stHeader"] {
    background: transparent;
}

/* Section headers */
h1, h2, h3 { color: #eee !important; }

/* Metric cards */
[data-testid="stMetricValue"] { color: #eee !important; }

/* Tabs */
button[data-baseweb="tab"] {
    color: #aaa !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #4ECDC4 !important;
    border-bottom-color: #4ECDC4 !important;
}

/* Dataframes */
[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }

/* Plotly chart containers */
.js-plotly-plot .plotly .main-svg { border-radius: 8px; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-thumb { background: #333; border-radius: 3px; }

/* Sidebar section labels */
.sidebar-section {
    color: #777;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 16px;
    margin-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)

# ── Initialise session state ──────────────────────────────────────────────
from streamlit_app.simulation.state_manager import init_session, get_engine, step_simulation, get_current, get_history

init_session()

# ── Sidebar controls (persistent across pages) ───────────────────────────
with st.sidebar:
    st.markdown("## ⚡ DEMS Grid")
    st.markdown('<div class="sidebar-section">Simulation</div>', unsafe_allow_html=True)

    # Play / Pause / Step
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("▶ Play", use_container_width=True, type="primary" if not st.session_state["running"] else "secondary"):
            st.session_state["running"] = True
    with col2:
        if st.button("⏸ Pause", use_container_width=True):
            st.session_state["running"] = False
    with col3:
        if st.button("⏭ Step", use_container_width=True):
            st.session_state["running"] = False
            step_simulation()

    st.session_state["sim_speed_hours"] = st.slider(
        "Step size (hours)", 0.25, 4.0, 1.0, 0.25,
        help="Simulated hours per step"
    )
    st.session_state["refresh_rate_s"] = st.slider(
        "Refresh rate (s)", 1, 10, 2,
        help="Seconds between auto-steps when running"
    )

    st.markdown('<div class="sidebar-section">Weather Override</div>', unsafe_allow_html=True)
    weather_mode = st.radio("Weather", ["Auto (24h profile)", "Manual"], horizontal=True, label_visibility="collapsed")
    if weather_mode == "Manual":
        irr_override = st.slider("Irradiance (W/m²)", 0, 1200, 600, 50)
        ws_override = st.slider("Wind Speed (m/s)", 0.0, 30.0, 8.0, 0.5)
    else:
        irr_override = None
        ws_override = None

    st.markdown('<div class="sidebar-section">Grid Events</div>', unsafe_allow_html=True)
    event = st.selectbox("Inject event", ["None", "Load spike Area A (+20%)", "Load drop Area B (-15%)", "Load spike Area C (+25%)"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown('<div class="sidebar-section">Mode</div>', unsafe_allow_html=True)
    st.session_state["sim_mode"] = st.radio("Simulation mode", ["embedded", "api"], horizontal=True, label_visibility="collapsed")
    if st.session_state["sim_mode"] == "api":
        st.session_state["api_url"] = st.text_input("API URL", st.session_state["api_url"])

# ── Auto-step logic ──────────────────────────────────────────────────────
if st.session_state["running"]:
    engine = get_engine()
    # Apply weather overrides before stepping
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
    import time
    time.sleep(0.1)  # small sleep before rerun
    st.rerun()

# ── Page router (Streamlit >= 1.30 native multi-page) ────────────────────
# The pages are auto-discovered from streamlit_app/pages/

# Show a landing hero when no page is selected
snap = get_current()
st.markdown(f"""
<div style="text-align:center;padding:40px 0 20px 0;">
    <h1 style="font-size:2.4rem;margin:0;">⚡ DEMS Grid Dashboard</h1>
    <p style="color:#888;font-size:1.05rem;margin-top:8px;">
        117-Bus Tri-Area SuperGrid · 23 DER Units · Real-Time Simulation
    </p>
</div>
""", unsafe_allow_html=True)

# Quick overview cards on landing page
from streamlit_app.components.metrics_cards import metric_card, status_badge
from streamlit_app.components.status_indicators import stability_indicator, convergence_dot, sim_clock

c1, c2, c3, c4, c5, c6 = st.columns(6)
prev = get_history(2)
prev_snap = prev[-2] if len(prev) >= 2 else snap

with c1:
    metric_card("Generation", f"{snap.total_generation_mw:.1f}", delta=snap.total_generation_mw - prev_snap.total_generation_mw, color="#00C853", icon="🔌", unit="MW")
with c2:
    metric_card("Load", f"{snap.total_load_mw:.1f}", delta=snap.total_load_mw - prev_snap.total_load_mw, color="#448AFF", icon="🏭", unit="MW")
with c3:
    metric_card("Losses", f"{snap.total_losses_mw:.1f}", delta=snap.total_losses_mw - prev_snap.total_losses_mw, color="#FF9100", icon="🔥", unit="MW", invert_delta=True)
with c4:
    metric_card("Frequency", f"{snap.system_frequency_hz:.3f}", delta=snap.system_frequency_hz - prev_snap.system_frequency_hz, color="#00E5FF", icon="〰️", unit="Hz")
with c5:
    metric_card("Avg Voltage", f"{snap.avg_voltage_pu:.4f}", delta=snap.avg_voltage_pu - prev_snap.avg_voltage_pu, color="#AA00FF", icon="⚡", unit="pu")
with c6:
    status_label = stability_indicator(snap.is_secure, snap.num_voltage_violations, snap.num_line_overloads)
    metric_card("Status", status_label, color={"SECURE":"#00C853","ALERT":"#FFD600","EMERGENCY":"#FF9100","CRITICAL":"#FF1744"}.get(status_label,"#888"), icon="🛡️")

st.markdown("---")

# Landing page also shows topology + power balance
from streamlit_app.components.topology_graph import build_topology_figure
from streamlit_app.components.time_series_charts import power_balance_chart, frequency_chart

col_topo, col_charts = st.columns([3, 2])
with col_topo:
    st.markdown("#### Grid Topology")
    convergence_dot(snap.converged)
    fig_topo = build_topology_figure(snap, height=520)
    st.plotly_chart(fig_topo, use_container_width=True, key="landing_topo")

with col_charts:
    sim_clock(snap.sim_hour, snap.step)
    hist = get_history(50)
    st.plotly_chart(power_balance_chart(hist, height=240), use_container_width=True, key="landing_pb")
    st.plotly_chart(frequency_chart(hist, height=220), use_container_width=True, key="landing_freq")

st.markdown("")
st.caption("Navigate to detailed views using the sidebar pages →")
