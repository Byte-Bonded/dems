"""
Page 6 – Tie Lines
Inter-area transfer analysis, Sankey diagram, N-1 contingency table.
"""

import sys, os
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _root not in sys.path:
    sys.path.insert(0, _root)

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from streamlit_app.simulation.state_manager import init_session, get_current, get_history
from streamlit_app.components.time_series_charts import tie_line_flow_chart
from streamlit_app.components.metrics_cards import metric_card
from streamlit_app.components.topology_graph import AREA_COLORS
from streamlit_app.components.sidebar import render_sidebar

render_sidebar()

st.markdown("## 🔗 Tie Lines & Inter-Area Transfers")
st.markdown("*8 tie-lines connecting Areas A, B, C in mesh topology for N-1 security*")

snap = get_current()
hist = get_history(80)
tl = snap.tie_line_flows

if not tl:
    st.warning("No tie-line data available. Run a simulation step first.")
    st.stop()

# ── Tie-line KPIs ────────────────────────────────────────────────────────
total_flow = sum(abs(t.get("p_from_mw", 0)) for t in tl)
max_loading = max(t.get("loading_percent", 0) for t in tl)
n_overloaded = sum(1 for t in tl if t.get("is_overloaded", False))
n_critical = sum(1 for t in tl if t.get("loading_percent", 0) > 80)

c1, c2, c3, c4 = st.columns(4)
with c1:
    metric_card("Total Transfer", f"{total_flow:.1f}", color="#4ECDC4", icon="↔", unit="MW")
with c2:
    metric_card("Max Loading", f"{max_loading:.1f}", color="#FFD600" if max_loading < 80 else "#FF5252", icon="📊", unit="%")
with c3:
    metric_card("Critical Lines", str(n_critical), color="#FF9100" if n_critical > 0 else "#00C853", icon="⚠️")
with c4:
    metric_card("Overloaded", str(n_overloaded), color="#FF1744" if n_overloaded > 0 else "#00C853", icon="🔴")

# ── Sankey diagram ───────────────────────────────────────────────────────
st.markdown("#### Power Flow Sankey")

# Compute net flows between area pairs
pair_flows = {"A→B": 0, "B→A": 0, "B→C": 0, "C→B": 0, "A→C": 0, "C→A": 0}
for t in tl:
    name = t.get("name", "")
    p = t.get("p_from_mw", 0)
    if "AB" in name:
        if p >= 0:
            pair_flows["A→B"] += p
        else:
            pair_flows["B→A"] += abs(p)
    elif "BC" in name:
        if p >= 0:
            pair_flows["B→C"] += p
        else:
            pair_flows["C→B"] += abs(p)
    elif "AC" in name:
        if p >= 0:
            pair_flows["A→C"] += p
        else:
            pair_flows["C→A"] += abs(p)

# Sankey: nodes = A(0), B(1), C(2)
labels = ["Area A", "Area B", "Area C"]
node_colors = [AREA_COLORS["A"], AREA_COLORS["B"], AREA_COLORS["C"]]

sources, targets, values, link_colors = [], [], [], []
flow_map = {
    "A→B": (0, 1, AREA_COLORS["A"]),
    "B→A": (1, 0, AREA_COLORS["B"]),
    "B→C": (1, 2, AREA_COLORS["B"]),
    "C→B": (2, 1, AREA_COLORS["C"]),
    "A→C": (0, 2, AREA_COLORS["A"]),
    "C→A": (2, 0, AREA_COLORS["C"]),
}
for key, (s, t, c) in flow_map.items():
    v = pair_flows.get(key, 0)
    if v > 0.1:
        sources.append(s)
        targets.append(t)
        values.append(v)
        link_colors.append(c + "66")  # semi-transparent

fig_sankey = go.Figure(go.Sankey(
    node=dict(pad=25, thickness=25, line=dict(color="#333", width=1),
              label=labels, color=node_colors),
    link=dict(source=sources, target=targets, value=values, color=link_colors),
))
fig_sankey.update_layout(height=320, margin=dict(l=20, r=20, t=30, b=20),
                         paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#ccc", size=12),
                         title="Inter-Area MW Transfers")
st.plotly_chart(fig_sankey, use_container_width=True, key="tl_sankey")

# ── Tie-line table ───────────────────────────────────────────────────────
st.markdown("#### Tie-Line Detail Table")
rows = []
for t in tl:
    loading = t.get("loading_percent", 0)
    status = "🔴 Overload" if t.get("is_overloaded") else ("🟡 Critical" if loading > 80 else "🟢 Normal")
    rows.append({
        "Name": t.get("name", ""),
        "From Bus": t.get("from_bus", ""),
        "To Bus": t.get("to_bus", ""),
        "P (MW)": round(t.get("p_from_mw", 0), 2),
        "Q (Mvar)": round(t.get("q_from_mvar", 0), 2),
        "Loading (%)": round(loading, 1),
        "Status": status,
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True, height=350)

# ── Tie-line flow time-series ────────────────────────────────────────────
st.markdown("#### Tie-Line Flow Trends")
st.plotly_chart(tie_line_flow_chart(hist, 300), use_container_width=True, key="tl_ts")

# ── Net area interchange summary ─────────────────────────────────────────
st.markdown("#### Net Area Interchange")
interchange_cols = st.columns(3)
for idx, area_key in enumerate(["A", "B", "C"]):
    adata = snap.area_states.get(area_key, {})
    net = adata.get("net_interchange_mw", 0)
    color = AREA_COLORS[area_key]
    with interchange_cols[idx]:
        direction = "Exporting ↗" if net >= 0 else "Importing ↙"
        st.markdown(f"""
        <div style="background:#1a1a2e;border-top:3px solid {color};border-radius:10px;padding:14px;text-align:center;">
            <div style="color:{color};font-weight:700;font-size:1rem;">Area {area_key}</div>
            <div style="color:#eee;font-size:1.4rem;font-weight:700;">{net:+.1f} MW</div>
            <div style="color:#888;font-size:0.8rem;">{direction}</div>
        </div>
        """, unsafe_allow_html=True)
