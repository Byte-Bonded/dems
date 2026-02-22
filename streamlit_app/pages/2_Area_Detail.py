"""
Page 2 – Area Detail
Deep dive into a single area: zoomed topology, bus table, gen table, line loading.
"""

import sys, os
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _root not in sys.path:
    sys.path.insert(0, _root)

import streamlit as st
import pandas as pd

from streamlit_app.simulation.state_manager import init_session, get_current, get_history
from streamlit_app.components.topology_graph import build_topology_figure, AREA_COLORS, AREA_NAMES, AREA_OFFSETS
from streamlit_app.components.time_series_charts import voltage_profile_chart
from streamlit_app.components.metrics_cards import metric_card
from streamlit_app.components.status_indicators import convergence_dot

init_session()

st.markdown("## 🗺️ Area Detail")

area = st.selectbox("Select Area", ["A", "B", "C"], format_func=lambda a: AREA_NAMES[a], key="area_sel")
color = AREA_COLORS[area]
snap = get_current()
adata = snap.area_states.get(area, {})

# ── Area KPIs ────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    metric_card("Generation", f"{adata.get('total_generation_mw',0):.1f}", color="#00C853", icon="🔌", unit="MW")
with c2:
    metric_card("Load", f"{adata.get('total_load_mw',0):.1f}", color="#448AFF", icon="🏭", unit="MW")
with c3:
    net = adata.get("net_interchange_mw", 0)
    metric_card("Net Export", f"{net:+.1f}", color="#4ECDC4" if net >= 0 else "#FF5252", icon="↔", unit="MW")
with c4:
    metric_card("Avg Voltage", f"{adata.get('avg_voltage_pu',1.0):.4f}", color="#AA00FF", icon="⚡", unit="pu")

# ── Zoomed topology ─────────────────────────────────────────────────────
st.markdown(f"#### {AREA_NAMES[area]} — Topology")
convergence_dot(snap.converged)
fig = build_topology_figure(snap, height=480, selected_area=area, show_labels=True)
st.plotly_chart(fig, use_container_width=True, key="area_topo")

# ── Data tables ──────────────────────────────────────────────────────────
lo, hi = AREA_OFFSETS[area], AREA_OFFSETS[area] + 38

tab_buses, tab_gens, tab_lines = st.tabs(["🔌 Buses", "⚙ Generators", "🔗 Lines"])

with tab_buses:
    bus_rows = []
    for bus in range(lo, hi + 1):
        vm = snap.bus_voltages.get(bus, None)
        va = snap.bus_angles.get(bus, None)
        if vm is None:
            continue
        status = "⚠️ Violation" if vm < 0.95 or vm > 1.05 else "✅"
        bus_rows.append({"Bus": bus, "V (pu)": round(vm, 4), "θ (°)": round(va, 2), "Status": status})
    if bus_rows:
        df = pd.DataFrame(bus_rows)
        st.dataframe(df, use_container_width=True, height=400,
                      column_config={"V (pu)": st.column_config.NumberColumn(format="%.4f"),
                                     "θ (°)": st.column_config.NumberColumn(format="%.2f")})
    else:
        st.info("No bus data available — run a simulation step.")

with tab_gens:
    gen_rows = []
    for g in snap.gen_results:
        bus = g["bus"]
        if lo <= bus <= hi:
            gen_rows.append({
                "Index": g["index"],
                "Bus": bus,
                "P (MW)": round(g["p_mw"], 2),
                "Q (Mvar)": round(g["q_mvar"], 2),
                "Vm setpoint (pu)": round(g["vm_pu"], 3),
            })
    if gen_rows:
        st.dataframe(pd.DataFrame(gen_rows), use_container_width=True, height=350)
    else:
        st.info("No generator data.")

with tab_lines:
    line_rows = []
    for lr in snap.line_results:
        fb, tb = lr["from_bus"], lr["to_bus"]
        if (lo <= fb <= hi) or (lo <= tb <= hi):
            loading = lr.get("loading_percent", 0)
            status = "🔴 Overload" if loading > 100 else ("🟡 High" if loading > 80 else "🟢")
            line_rows.append({
                "From": fb, "To": tb,
                "Name": lr.get("name", ""),
                "P (MW)": round(lr.get("p_from_mw", 0), 2),
                "Loading (%)": round(loading, 1),
                "Status": status,
            })
    if line_rows:
        st.dataframe(pd.DataFrame(line_rows), use_container_width=True, height=400)
    else:
        st.info("No line data.")

# ── Voltage profile ────────────────────────────────────────────────────
st.markdown(f"#### Voltage Profile — {AREA_NAMES[area]}")
st.plotly_chart(voltage_profile_chart(snap, area=area, height=280), use_container_width=True, key="area_vp")
