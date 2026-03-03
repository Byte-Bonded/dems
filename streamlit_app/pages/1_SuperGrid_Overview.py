"""
Page 1 – SuperGrid Overview
Full 117-bus topology with global metrics, area summaries, and tie-line overview.
"""

import sys, os
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _root not in sys.path:
    sys.path.insert(0, _root)

import streamlit as st
import plotly.graph_objects as go

from streamlit_app.simulation.state_manager import init_session, get_current, get_history, get_engine
from streamlit_app.components.metrics_cards import metric_card, status_badge, small_stat
from streamlit_app.components.topology_graph import build_topology_figure, AREA_COLORS, AREA_NAMES
from streamlit_app.components.time_series_charts import power_balance_chart, frequency_chart, voltage_profile_chart, tie_line_flow_chart
from streamlit_app.components.status_indicators import stability_indicator, convergence_dot, sim_clock
from streamlit_app.components.der_cards import der_summary_row
from streamlit_app.components.sidebar import render_sidebar

render_sidebar()

st.markdown("## 🔌 SuperGrid Overview")
st.markdown("*117-bus Tri-Area System — 3 × IEEE 39-bus (New England)*")

snap = get_current()
hist = get_history(80)
prev = hist[-2] if len(hist) >= 2 else snap

# ── Row 1: KPI cards ─────────────────────────────────────────────────────
c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
    metric_card("Generation", f"{snap.total_generation_mw:.1f}", snap.total_generation_mw - prev.total_generation_mw, "#00C853", "🔌", unit="MW")
with c2:
    metric_card("Load", f"{snap.total_load_mw:.1f}", snap.total_load_mw - prev.total_load_mw, "#448AFF", "🏭", unit="MW")
with c3:
    metric_card("Losses", f"{snap.total_losses_mw:.1f}", snap.total_losses_mw - prev.total_losses_mw, "#FF9100", "🔥", unit="MW", invert_delta=True)
with c4:
    metric_card("Frequency", f"{snap.system_frequency_hz:.3f}", snap.system_frequency_hz - prev.system_frequency_hz, "#00E5FF", "〰️", unit="Hz")
with c5:
    metric_card("Avg Voltage", f"{snap.avg_voltage_pu:.4f}", snap.avg_voltage_pu - prev.avg_voltage_pu, "#AA00FF", "⚡", unit="pu")
with c6:
    sl = stability_indicator(snap.is_secure, snap.num_voltage_violations, snap.num_line_overloads)
    metric_card("Status", sl, color={"SECURE":"#00C853","ALERT":"#FFD600","EMERGENCY":"#FF9100","CRITICAL":"#FF1744"}.get(sl,"#888"), icon="🛡️")

st.markdown("")

# ── Row 2: Topology + time-series charts ─────────────────────────────────
col_left, col_right = st.columns([3, 2])

with col_left:
    st.markdown("#### Interactive Grid Topology")
    show_labels = st.checkbox("Show bus labels", value=False, key="sg_labels")
    convergence_dot(snap.converged)
    fig = build_topology_figure(snap, height=550, show_labels=show_labels)
    st.plotly_chart(fig, use_container_width=True, key="sg_topo")

with col_right:
    sim_clock(snap.sim_hour, snap.step)
    tab1, tab2, tab3 = st.tabs(["Power Balance", "Frequency", "Tie-Lines"])
    with tab1:
        st.plotly_chart(power_balance_chart(hist, 260), use_container_width=True, key="sg_pb")
    with tab2:
        st.plotly_chart(frequency_chart(hist, 260), use_container_width=True, key="sg_freq")
    with tab3:
        st.plotly_chart(tie_line_flow_chart(hist, 260), use_container_width=True, key="sg_tl")

# ── Row 3: Area summary cards ────────────────────────────────────────────
st.markdown("#### Area Summaries")
area_cols = st.columns(3)

for idx, area_key in enumerate(["A", "B", "C"]):
    adata = snap.area_states.get(area_key, {})
    color = AREA_COLORS[area_key]
    with area_cols[idx]:
        gen_mw = adata.get("total_generation_mw", 0)
        load_mw = adata.get("total_load_mw", 0)
        net = adata.get("net_interchange_mw", gen_mw - load_mw)
        vmin = adata.get("min_voltage_pu", 0)
        vmax = adata.get("max_voltage_pu", 0)
        n_gen = adata.get("num_generators", 0)
        n_load = adata.get("num_loads", 0)

        # Count DERs in this area
        area_ders = [d for d in snap.der_states if d.get("name", "").endswith(f"_{area_key}1") or f"_{area_key}" in d.get("name","")]
        der_count = len(area_ders)
        der_cap = sum(d.get("capacity_mw", 0) for d in area_ders)

        st.markdown(f"""
        <div style="background:linear-gradient(135deg, #1a1a2e, {color}15);
            border-top:4px solid {color}; border-radius:12px; padding:16px;">
            <h4 style="color:{color};margin:0 0 8px 0;">{AREA_NAMES[area_key]}</h4>
            <table style="width:100%;color:#ccc;font-size:0.85rem;">
                <tr><td>Generation</td><td style="text-align:right;font-weight:600;">{gen_mw:.1f} MW</td></tr>
                <tr><td>Load</td><td style="text-align:right;font-weight:600;">{load_mw:.1f} MW</td></tr>
                <tr><td>Net Export</td><td style="text-align:right;font-weight:600;color:{'#00C853' if net>=0 else '#FF5252'};">{net:+.1f} MW</td></tr>
                <tr><td>Voltage Range</td><td style="text-align:right;">{vmin:.3f} – {vmax:.3f} pu</td></tr>
                <tr><td>Generators</td><td style="text-align:right;">{n_gen}</td></tr>
                <tr><td>Loads</td><td style="text-align:right;">{n_load}</td></tr>
                <tr><td>DER Units</td><td style="text-align:right;">{der_count} ({der_cap:.0f} MW)</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

# ── Row 4: DER summary bar ──────────────────────────────────────────────
st.markdown("#### DER Overview")
if snap.der_summary:
    der_summary_row(snap.der_summary)

# ── Row 5: Voltage profile ──────────────────────────────────────────────
st.markdown("#### Bus Voltage Profile")
st.plotly_chart(voltage_profile_chart(snap, height=300), use_container_width=True, key="sg_volt")
