"""
Page 4 – Dynamics Monitor
Generator frequency/angle, AGC, protection relay status.
"""

import sys, os
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _root not in sys.path:
    sys.path.insert(0, _root)

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from streamlit_app.simulation.state_manager import init_session, get_current, get_history
from streamlit_app.components.time_series_charts import frequency_chart
from streamlit_app.components.metrics_cards import metric_card
from streamlit_app.components.sidebar import render_sidebar

render_sidebar()

st.markdown("## 📊 Dynamics Monitor")
st.markdown("*Generator swing dynamics, AGC, exciter/governor, protection relays*")

snap = get_current()
hist = get_history(80)

# ── System frequency gauge ───────────────────────────────────────────────
c1, c2, c3 = st.columns([2, 3, 2])

with c1:
    freq = snap.system_frequency_hz
    freq_dev = freq - 50.0
    dev_color = "#00C853" if abs(freq_dev) < 0.1 else ("#FFD600" if abs(freq_dev) < 0.3 else "#FF5252")
    metric_card("System Frequency", f"{freq:.4f}", delta=freq_dev, color="#00E5FF", icon="〰️", unit="Hz")
    st.markdown(f"""
    <div style="text-align:center;margin-top:8px;">
        <span style="color:{dev_color};font-size:1.8rem;font-weight:700;">{freq_dev:+.4f} Hz</span>
        <br/><span style="color:#777;font-size:0.7rem;">deviation from 50 Hz</span>
    </div>
    """, unsafe_allow_html=True)

with c2:
    # Frequency gauge
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=freq,
        number=dict(suffix=" Hz", font=dict(color="#eee", size=28)),
        gauge=dict(
            axis=dict(range=[49.0, 51.0], tickcolor="#555", dtick=0.25),
            bar=dict(color="#00E5FF"),
            bgcolor="#1a1a2e",
            bordercolor="#333",
            steps=[
                dict(range=[49.0, 49.5], color="rgba(255,82,82,0.2)"),
                dict(range=[49.5, 49.7], color="rgba(255,214,0,0.12)"),
                dict(range=[49.7, 50.3], color="rgba(0,200,83,0.08)"),
                dict(range=[50.3, 50.5], color="rgba(255,214,0,0.12)"),
                dict(range=[50.5, 51.0], color="rgba(255,82,82,0.2)"),
            ],
            threshold=dict(line=dict(color="#FF1744", width=3), thickness=0.8, value=50.0),
        ),
    ))
    fig_gauge.update_layout(height=200, margin=dict(l=30, r=30, t=30, b=10),
                            paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#ccc"))
    st.plotly_chart(fig_gauge, use_container_width=True, key="dyn_gauge")

with c3:
    # Quick stats
    n_gens = len(snap.generator_frequencies) if snap.generator_frequencies else 0
    metric_card("Generators Tracked", str(n_gens), color="#4ECDC4", icon="⚙️")
    n_agc = len(snap.agc_adjustments) if snap.agc_adjustments else 0
    metric_card("AGC Controllers", str(n_agc), color="#FFE66D", icon="🎛️")

# ── Frequency time-series ────────────────────────────────────────────────
st.markdown("#### Frequency Trend")
st.plotly_chart(frequency_chart(hist, 260), use_container_width=True, key="dyn_freq_ts")

# ── Generator frequencies (multi-line) ───────────────────────────────────
st.markdown("#### Generator Frequencies")
if snap.generator_frequencies:
    area_colors_map = {}
    for gid in snap.generator_frequencies:
        parts = gid.split("_")
        area = parts[1] if len(parts) >= 2 else "?"
        area_colors_map[gid] = {"A": "#FF6B6B", "B": "#4ECDC4", "C": "#FFE66D"}.get(area, "#888")

    fig_gf = go.Figure()
    # Collect per-generator traces across history
    gen_ids = sorted(snap.generator_frequencies.keys())
    steps = [s.step for s in hist]
    for gid in gen_ids[:15]:  # limit to avoid clutter
        vals = [s.generator_frequencies.get(gid, 50.0) for s in hist]
        fig_gf.add_trace(go.Scatter(x=steps, y=vals, mode="lines", name=gid,
                                     line=dict(color=area_colors_map.get(gid, "#888"), width=1.5)))
    fig_gf.add_hline(y=50.0, line_dash="dash", line_color="#555")
    fig_gf.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(14,17,23,0.6)",
                         font=dict(color="#ccc"), margin=dict(l=50, r=20, t=30, b=30),
                         legend=dict(font=dict(size=8, color="#aaa"), bgcolor="rgba(0,0,0,0)"),
                         yaxis=dict(gridcolor="#222", title="Frequency (Hz)"),
                         xaxis=dict(gridcolor="#222", title="Step"))
    st.plotly_chart(fig_gf, use_container_width=True, key="dyn_gen_freq")
else:
    st.info("Generator frequency data not yet available. Dynamic simulation must be running.")

# ── Generator rotor angles ───────────────────────────────────────────────
st.markdown("#### Rotor Angles")
if snap.generator_angles:
    angle_data = []
    for gid, angle in sorted(snap.generator_angles.items()):
        parts = gid.split("_")
        area = parts[1] if len(parts) >= 2 else "?"
        angle_data.append({"Generator": gid, "Area": area, "δ (deg)": round(angle, 2)})
    df_angles = pd.DataFrame(angle_data)

    fig_angle = go.Figure()
    for area, color in [("A", "#FF6B6B"), ("B", "#4ECDC4"), ("C", "#FFE66D")]:
        mask = df_angles["Area"] == area
        subset = df_angles[mask]
        if not subset.empty:
            fig_angle.add_trace(go.Bar(x=subset["Generator"], y=subset["δ (deg)"],
                                        marker_color=color, name=f"Area {area}"))
    fig_angle.update_layout(height=280, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(14,17,23,0.6)",
                            font=dict(color="#ccc"), margin=dict(l=50, r=20, t=30, b=30),
                            barmode="group", yaxis_title="Rotor Angle (°)", xaxis_tickangle=-45)
    st.plotly_chart(fig_angle, use_container_width=True, key="dyn_angles")
else:
    st.info("Rotor angle data not yet available.")

# ── AGC Panel ────────────────────────────────────────────────────────────
st.markdown("#### AGC (Automatic Generation Control)")
if snap.agc_adjustments:
    for area_id, adjustments in snap.agc_adjustments.items():
        with st.expander(f"Area {area_id} AGC", expanded=True):
            if isinstance(adjustments, dict):
                adj_rows = [{"Generator": k, "ΔP (MW)": round(v, 3)} for k, v in adjustments.items()]
                if adj_rows:
                    st.dataframe(pd.DataFrame(adj_rows), use_container_width=True, height=200)
                else:
                    st.write("No adjustments")
            else:
                st.write(adjustments)
else:
    st.info("AGC data not available. Dynamics must be initialised.")

# ── Protection Relays ────────────────────────────────────────────────────
st.markdown("#### Protection Relay Status")
if snap.protection_status:
    prot_rows = []
    for relay_id, status in snap.protection_status.items():
        tripped = status.get("tripped", False)
        reason = status.get("trip_reason", "—")
        prot_rows.append({
            "Relay": relay_id,
            "Status": "🔴 TRIPPED" if tripped else "🟢 Normal",
            "Reason": reason,
        })
    if prot_rows:
        st.dataframe(pd.DataFrame(prot_rows), use_container_width=True, height=300)
else:
    st.info("Protection relay data not available.")
