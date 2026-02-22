"""
Page 5 – Microgrid View
Standalone 5-bus microgrid with custom Newton-Raphson solver and DER models.
"""

import sys, os
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _root not in sys.path:
    sys.path.insert(0, _root)

import streamlit as st
import plotly.graph_objects as go
import numpy as np

from streamlit_app.simulation.state_manager import init_session

init_session()

st.markdown("## 🏘️ Microgrid View")
st.markdown("*Standalone 5-bus system — PyPower-compatible Newton-Raphson solver*")

# ── Lazy import & cache the microgrid ────────────────────────────────────
@st.cache_resource
def _build_microgrid():
    from src.simulation.microgrid import (
        create_example_microgrid, PowerFlowSolver,
        MicrogridController, SolarPV, WindTurbine, BatteryESS, DieselGenerator,
    )
    case, ders = create_example_microgrid()
    solver = PowerFlowSolver(case)
    controller = MicrogridController(ders, case)
    return case, ders, solver, controller

try:
    case, ders, solver, controller = _build_microgrid()
    mg_available = True
except Exception as exc:
    mg_available = False
    st.error(f"Failed to build microgrid: {exc}")

if not mg_available:
    st.stop()

# ── Controls ─────────────────────────────────────────────────────────────
col_ctrl, col_viz = st.columns([1, 3])

with col_ctrl:
    st.markdown("#### Controls")
    time_step = st.slider("Time step (h)", 0, 23, 12, key="mg_ts")
    droop_gain = st.slider("Droop gain", 0.01, 0.10, 0.05, 0.01, key="mg_droop")
    v_setpoint = st.slider("Voltage setpoint (pu)", 0.95, 1.05, 1.0, 0.01, key="mg_vsp")

    run_pf_btn = st.button("Run Power Flow", type="primary", key="mg_run")
    run_sim_btn = st.button("Run 24h Simulation", key="mg_sim24")

# ── Run power flow ───────────────────────────────────────────────────────
if "mg_pf_result" not in st.session_state:
    st.session_state["mg_pf_result"] = None

if run_pf_btn:
    # Update DER outputs for current time step
    for d in ders:
        try:
            p, q = d.get_output(time_step)
        except:
            pass
    try:
        controller.update_generation(time_step)
    except:
        pass
    result = solver.run()
    st.session_state["mg_pf_result"] = result

pf = st.session_state.get("mg_pf_result")

# ── Topology visualisation ───────────────────────────────────────────────
with col_viz:
    st.markdown("#### 5-Bus Microgrid Topology")

    # Fixed layout for 5-bus
    positions = {
        0: (0, 2),    # Slack / Grid connection
        1: (-1.5, 0), # PV bus
        2: (1.5, 0),  # Wind bus
        3: (0, -1.5), # Load + BESS
        4: (-2, -2),  # Diesel
    }
    bus_labels = {0: "Slack", 1: "Solar PV", 2: "Wind", 3: "Load+BESS", 4: "Diesel"}
    bus_colors = ["#FF6B6B", "#FFD700", "#00BFFF", "#00C853", "#FF9100"]

    # Branches from case
    fig = go.Figure()
    branch = case.branch
    for br_idx in range(case.branch_count):
        fb = int(branch[br_idx, 0])
        tb = int(branch[br_idx, 1])
        x0, y0 = positions.get(fb, (0, 0))
        x1, y1 = positions.get(tb, (0, 0))
        fig.add_trace(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode="lines", line=dict(color="#555", width=3),
            hoverinfo="text",
            text=f"Branch {fb}→{tb} | R={branch[br_idx,2]:.4f} X={branch[br_idx,3]:.4f}",
            showlegend=False,
        ))

    # Bus nodes
    for bus_id, (x, y) in positions.items():
        vm = 1.0
        va = 0.0
        if pf and pf.get("converged"):
            V = pf.get("V")
            if V is not None and bus_id < len(V):
                vm = abs(V[bus_id])
                va = np.angle(V[bus_id], deg=True)
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers+text",
            marker=dict(size=28, color=bus_colors[bus_id], line=dict(width=2, color="#222")),
            text=bus_labels.get(bus_id, str(bus_id)),
            textposition="top center",
            textfont=dict(size=11, color="#eee"),
            hoverinfo="text",
            hovertext=f"Bus {bus_id}: {bus_labels.get(bus_id,'')}<br>V={vm:.4f} pu | θ={va:.1f}°",
            showlegend=False,
        ))

    fig.update_layout(
        height=380, margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        hoverlabel=dict(bgcolor="#1e1e2e", font_size=12, font_color="#eee"),
    )
    st.plotly_chart(fig, use_container_width=True, key="mg_topo")

# ── Power flow results ───────────────────────────────────────────────────
if pf:
    st.markdown("#### Power Flow Results")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Converged", "✅ Yes" if pf.get("converged") else "❌ No")
    with c2:
        st.metric("Iterations", pf.get("iterations", "?"))
    with c3:
        total_gen = sum(pf.get("gen", [[0,0]])[i][1] for i in range(len(pf.get("gen", [])))) if pf.get("gen") is not None else 0
        st.metric("Total Gen (MW)", f"{total_gen:.2f}")

    # Bus results
    if pf.get("bus") is not None:
        st.markdown("**Bus Voltages**")
        import pandas as pd
        V = pf.get("V", [])
        bus_data = []
        for i in range(min(len(V), case.bus_count)):
            bus_data.append({
                "Bus": i,
                "Label": bus_labels.get(i, ""),
                "|V| (pu)": round(abs(V[i]), 4),
                "θ (°)": round(np.angle(V[i], deg=True), 2),
            })
        st.dataframe(pd.DataFrame(bus_data), use_container_width=True)

# ── DER outputs ──────────────────────────────────────────────────────────
st.markdown("#### DER Component Outputs")
der_cols = st.columns(len(ders))
for i, d in enumerate(ders):
    with der_cols[i]:
        try:
            p, q = d.get_output(time_step)
        except:
            p, q = 0, 0
        icon = {"SolarPV": "☀️", "WindTurbine": "🌬️", "BatteryESS": "🔋", "DieselGenerator": "⛽"}.get(type(d).__name__, "⚙️")
        color = {"SolarPV": "#FFD700", "WindTurbine": "#00BFFF", "BatteryESS": "#00C853", "DieselGenerator": "#FF9100"}.get(type(d).__name__, "#888")
        soc_str = ""
        if hasattr(d, "soc"):
            soc_str = f"<br>SOC: {d.soc:.1%}"
        st.markdown(f"""
        <div style="background:#1a1a2e;border-top:3px solid {color};border-radius:10px;padding:12px;text-align:center;">
            <div style="font-size:1.3rem;">{icon}</div>
            <div style="color:#eee;font-size:0.85rem;font-weight:600;">{d.name}</div>
            <div style="color:{color};font-size:1.1rem;font-weight:700;">P={p:.2f} MW</div>
            <div style="color:#aaa;font-size:0.75rem;">Q={q:.2f} Mvar{soc_str}</div>
        </div>
        """, unsafe_allow_html=True)

# ── 24-hour simulation ───────────────────────────────────────────────────
if run_sim_btn:
    st.markdown("#### 24-Hour Simulation")
    p_solar, p_wind, p_bess, p_diesel = [], [], [], []
    v_profiles = {i: [] for i in range(case.bus_count)}

    progress = st.progress(0)
    for h in range(24):
        for d in ders:
            try:
                d.get_output(h)
            except:
                pass
        try:
            controller.update_generation(h)
        except:
            pass
        result = solver.run()
        if result.get("converged") and result.get("V") is not None:
            for bus_id in range(case.bus_count):
                v_profiles[bus_id].append(abs(result["V"][bus_id]))

        # Collect DER outputs
        for d in ders:
            try:
                p, _ = d.get_output(h)
            except:
                p = 0
            tname = type(d).__name__
            if tname == "SolarPV":
                p_solar.append(p)
            elif tname == "WindTurbine":
                p_wind.append(p)
            elif tname == "BatteryESS":
                p_bess.append(p)
            elif tname == "DieselGenerator":
                p_diesel.append(p)
        progress.progress((h + 1) / 24)

    hours = list(range(24))

    # DER output chart
    fig_der = go.Figure()
    if p_solar:
        fig_der.add_trace(go.Scatter(x=hours, y=p_solar, mode="lines", name="Solar",
                                      line=dict(color="#FFD700", width=2), fill="tozeroy", fillcolor="rgba(255,215,0,0.1)"))
    if p_wind:
        fig_der.add_trace(go.Scatter(x=hours, y=p_wind, mode="lines", name="Wind",
                                      line=dict(color="#00BFFF", width=2), fill="tozeroy", fillcolor="rgba(0,191,255,0.1)"))
    if p_bess:
        fig_der.add_trace(go.Scatter(x=hours, y=p_bess, mode="lines", name="Battery",
                                      line=dict(color="#00C853", width=2)))
    if p_diesel:
        fig_der.add_trace(go.Scatter(x=hours, y=p_diesel, mode="lines", name="Diesel",
                                      line=dict(color="#FF9100", width=2)))
    fig_der.update_layout(height=280, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(14,17,23,0.6)",
                          font=dict(color="#ccc"), margin=dict(l=50, r=20, t=30, b=30),
                          title="DER Output over 24 Hours", xaxis_title="Hour", yaxis_title="Power (MW)")
    st.plotly_chart(fig_der, use_container_width=True, key="mg_24h_der")

    # Voltage profiles
    fig_v = go.Figure()
    colors_v = ["#FF6B6B", "#FFD700", "#00BFFF", "#00C853", "#FF9100"]
    for bus_id, vals in v_profiles.items():
        if vals:
            fig_v.add_trace(go.Scatter(x=hours[:len(vals)], y=vals, mode="lines",
                                        name=f"Bus {bus_id}", line=dict(color=colors_v[bus_id % 5], width=2)))
    fig_v.add_hline(y=0.95, line_dash="dash", line_color="#FF5252")
    fig_v.add_hline(y=1.05, line_dash="dash", line_color="#FF5252")
    fig_v.update_layout(height=260, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(14,17,23,0.6)",
                        font=dict(color="#ccc"), margin=dict(l=50, r=20, t=30, b=30),
                        title="Bus Voltages over 24 Hours", xaxis_title="Hour", yaxis_title="V (pu)")
    st.plotly_chart(fig_v, use_container_width=True, key="mg_24h_v")
