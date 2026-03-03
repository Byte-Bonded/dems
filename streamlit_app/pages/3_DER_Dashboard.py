"""
Page 3 – DER Dashboard
All 23 DER units grouped by type with interactive controls.
"""

import sys, os
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _root not in sys.path:
    sys.path.insert(0, _root)

import streamlit as st
import plotly.graph_objects as go

from streamlit_app.simulation.state_manager import init_session, get_current, get_history, get_engine
from streamlit_app.components.der_cards import der_state_card, der_summary_row, DER_ICONS, DER_COLORS, DER_LABELS
from streamlit_app.components.time_series_charts import der_output_chart, soc_gauge
from streamlit_app.components.metrics_cards import metric_card
from streamlit_app.components.sidebar import render_sidebar

render_sidebar()

st.markdown("## 🔋 DER Dashboard")
st.markdown("*Distributed Energy Resources — 23 units across 3 areas*")

snap = get_current()
hist = get_history(60)
engine = get_engine()

# ── Summary row ──────────────────────────────────────────────────────────
if snap.der_summary:
    der_summary_row(snap.der_summary)
st.markdown("")

# ── DER output stacked chart ────────────────────────────────────────────
st.plotly_chart(der_output_chart(hist, 260), use_container_width=True, key="der_out")

st.markdown("---")

# ── Group DERs by type ──────────────────────────────────────────────────
ders_by_type = {}
for d in snap.der_states:
    dt = d.get("der_type", "unknown").lower()
    ders_by_type.setdefault(dt, []).append(d)

# ── Tabs per type ────────────────────────────────────────────────────────
tab_order = ["solar_pv", "wind", "bess", "ev_charging", "demand_response"]
tab_labels = [f"{DER_ICONS.get(t,'⚙️')} {DER_LABELS.get(t, t.title())}" for t in tab_order]
tabs = st.tabs(tab_labels)

for tab, dt in zip(tabs, tab_order):
    with tab:
        units = ders_by_type.get(dt, [])
        if not units:
            st.info(f"No {DER_LABELS.get(dt, dt)} units in the grid.")
            continue

        # ── Cards row ──
        cols = st.columns(min(len(units), 4))
        for i, ds in enumerate(units):
            with cols[i % len(cols)]:
                der_state_card(ds)

        # ── Type-specific controls ──
        if dt == "solar_pv":
            st.markdown("##### ☀️ Solar Controls")
            irr = st.slider("Irradiance override (W/m²)", 0, 1200, 600, 50, key="solar_irr")
            if st.button("Apply irradiance", key="apply_irr"):
                engine.set_irradiance(irr)
                st.success(f"Irradiance set to {irr} W/m²")

        elif dt == "wind":
            st.markdown("##### 🌬️ Wind Controls")
            ws = st.slider("Wind speed override (m/s)", 0.0, 30.0, 8.0, 0.5, key="wind_ws")
            if st.button("Apply wind speed", key="apply_ws"):
                engine.set_wind_speed(ws)
                st.success(f"Wind speed set to {ws} m/s")

            # Show IEC power curve
            with st.expander("IEC 61400 Power Curve"):
                import numpy as np
                speeds = np.linspace(0, 30, 200)
                power = []
                for v in speeds:
                    if v < 3:
                        power.append(0)
                    elif v <= 12:
                        power.append(((v - 3) / (12 - 3)) ** 3)
                    elif v <= 25:
                        power.append(1.0)
                    else:
                        power.append(0)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=speeds, y=power, mode="lines",
                                         line=dict(color="#00BFFF", width=2)))
                fig.update_layout(height=200, margin=dict(l=40,r=20,t=30,b=30),
                                  paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(14,17,23,0.6)",
                                  xaxis_title="Wind Speed (m/s)", yaxis_title="Power (pu)",
                                  font=dict(color="#ccc"), title="IEC 61400 Cubic Power Curve")
                st.plotly_chart(fig, use_container_width=True, key="wind_curve")

        elif dt == "bess":
            st.markdown("##### 🔋 Battery Controls")
            # SOC gauges
            gauge_cols = st.columns(len(units))
            for i, ds in enumerate(units):
                soc = ds.get("soc")
                soc_pct = (soc * 100) if soc and soc <= 1.0 else (soc or 0)
                with gauge_cols[i]:
                    st.plotly_chart(soc_gauge(soc_pct, ds.get("name", "BESS")),
                                    use_container_width=True, key=f"soc_{ds.get('name','')}")

            # Manual charge/discharge
            st.markdown("**Manual Command**")
            bess_name = st.selectbox("Select BESS", [d.get("name") for d in units], key="bess_sel")
            power = st.slider("Power MW (+discharge / −charge)", -50.0, 50.0, 0.0, 1.0, key="bess_pwr")
            if st.button("Apply", key="apply_bess"):
                engine.set_battery_power(bess_name, power)
                st.success(f"{bess_name}: {power:+.1f} MW")

        elif dt == "ev_charging":
            st.markdown("##### ⚡ EV Charging Stations")
            for ds in units:
                nc = ds.get("num_active_chargers")
                util = ds.get("utilization")
                if nc is not None or util is not None:
                    st.markdown(f"**{ds.get('name','')}**: {nc or '?'} active chargers, utilisation {util:.0%}" if util else f"**{ds.get('name','')}**")

        elif dt == "demand_response":
            st.markdown("##### 📉 Demand Response Controls")
            dr_name = st.selectbox("Select DR program", [d.get("name") for d in units], key="dr_sel")
            curt = st.slider("Curtailment level", 0.0, 1.0, 0.0, 0.05, key="dr_curt")
            if st.button("Apply curtailment", key="apply_dr"):
                engine.set_dr_curtailment(dr_name, curt)
                st.success(f"{dr_name}: curtailment {curt:.0%}")
