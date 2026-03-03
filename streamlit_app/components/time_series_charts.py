"""
Real-time scrolling time-series chart builders using Plotly.
"""

from typing import Dict, List, Optional

import plotly.graph_objects as go
from streamlit_app.simulation.sim_engine import SimSnapshot


# ---------------------------------------------------------------------------
# Common layout defaults
# ---------------------------------------------------------------------------
_DARK_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(14,17,23,0.6)",
    font=dict(color="#ccc", size=11),
    margin=dict(l=50, r=20, t=35, b=35),
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
        font=dict(size=10, color="#bbb"), bgcolor="rgba(0,0,0,0)",
    ),
    xaxis=dict(gridcolor="#222", zerolinecolor="#333"),
    yaxis=dict(gridcolor="#222", zerolinecolor="#333"),
)


def _apply_dark(fig: go.Figure, **kw) -> go.Figure:
    layout = {**_DARK_LAYOUT, **kw}
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# Public chart builders
# ---------------------------------------------------------------------------

def power_balance_chart(history: List[SimSnapshot], height: int = 280) -> go.Figure:
    """Stacked area: generation, load, losses over time."""
    steps = [s.step for s in history]
    gen = [s.total_generation_mw for s in history]
    load = [s.total_load_mw for s in history]
    losses = [s.total_losses_mw for s in history]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=steps, y=gen, mode="lines", name="Generation",
                             line=dict(color="#00C853", width=2), fill="tozeroy",
                             fillcolor="rgba(0,200,83,0.15)"))
    fig.add_trace(go.Scatter(x=steps, y=load, mode="lines", name="Load",
                             line=dict(color="#448AFF", width=2), fill="tozeroy",
                             fillcolor="rgba(68,138,255,0.10)"))
    fig.add_trace(go.Scatter(x=steps, y=losses, mode="lines", name="Losses",
                             line=dict(color="#FF9100", width=1.5, dash="dot")))
    return _apply_dark(fig, height=height, title="Power Balance (MW)")


def frequency_chart(history: List[SimSnapshot], height: int = 260) -> go.Figure:
    """System frequency with IEGC limit bands."""
    steps = [s.step for s in history]
    freq = [s.system_frequency_hz for s in history]

    fig = go.Figure()
    # limit bands
    fig.add_hrect(y0=49.5, y1=49.7, fillcolor="rgba(255,0,0,0.08)", line_width=0)
    fig.add_hrect(y0=50.3, y1=50.5, fillcolor="rgba(255,0,0,0.08)", line_width=0)
    fig.add_hrect(y0=49.7, y1=50.3, fillcolor="rgba(0,200,100,0.05)", line_width=0)
    fig.add_hline(y=50.0, line_dash="dash", line_color="#555", line_width=1)
    fig.add_hline(y=49.5, line_dash="dot", line_color="#FF5252", line_width=1, annotation_text="49.5 Hz")
    fig.add_hline(y=50.5, line_dash="dot", line_color="#FF5252", line_width=1, annotation_text="50.5 Hz")

    fig.add_trace(go.Scatter(x=steps, y=freq, mode="lines+markers",
                             line=dict(color="#00E5FF", width=2),
                             marker=dict(size=4), name="Frequency"))
    return _apply_dark(fig, height=height, title="System Frequency (Hz)",
                       yaxis=dict(gridcolor="#222", range=[49.3, 50.7]))


def voltage_profile_chart(snap: SimSnapshot, area: Optional[str] = None, height: int = 280) -> go.Figure:
    """Bar chart of bus voltages for an area or entire grid."""
    buses = sorted(snap.bus_voltages.keys())
    if area:
        lo, hi = {"A": (0, 38), "B": (39, 77), "C": (78, 116)}[area]
        buses = [b for b in buses if lo <= b <= hi]

    vm = [snap.bus_voltages.get(b, 1.0) for b in buses]
    colors = ["#FF5252" if v < 0.95 or v > 1.05 else "#4ECDC4" for v in vm]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=[str(b) for b in buses], y=vm, marker_color=colors, name="Vm (pu)"))
    fig.add_hline(y=0.95, line_dash="dash", line_color="#FF5252", line_width=1)
    fig.add_hline(y=1.05, line_dash="dash", line_color="#FF5252", line_width=1)
    fig.add_hline(y=1.0, line_dash="dot", line_color="#555", line_width=1)
    title = f"Voltage Profile – Area {area}" if area else "Voltage Profile – All Buses"
    return _apply_dark(fig, height=height, title=title,
                       yaxis=dict(gridcolor="#222", range=[0.90, 1.12]),
                       xaxis=dict(title="Bus", tickangle=-45, dtick=2))


def der_output_chart(history: List[SimSnapshot], height: int = 280) -> go.Figure:
    """Stacked bar of DER output by type over time."""
    steps = [s.step for s in history]

    # Aggregate outputs by type per step
    type_totals: Dict[str, List[float]] = {
        "solar_pv": [], "wind": [], "bess": [], "ev_charging": [], "demand_response": [],
    }
    for s in history:
        sums: Dict[str, float] = {k: 0.0 for k in type_totals}
        for d in s.der_states:
            dt = d.get("der_type", "").lower()
            if dt in sums:
                sums[dt] += d.get("current_output_mw", 0)
        for k in type_totals:
            type_totals[k].append(sums[k])

    der_colors = {"solar_pv": "#FFD700", "wind": "#00BFFF", "bess": "#00C853",
                  "ev_charging": "#AA00FF", "demand_response": "#FF6D00"}
    der_labels = {"solar_pv": "Solar", "wind": "Wind", "bess": "Battery",
                  "ev_charging": "EV", "demand_response": "DR"}

    fig = go.Figure()
    for dt in type_totals:
        fig.add_trace(go.Bar(x=steps, y=type_totals[dt], name=der_labels[dt],
                             marker_color=der_colors[dt]))
    fig.update_layout(barmode="stack")
    return _apply_dark(fig, height=height, title="DER Output by Type (MW)")


def tie_line_flow_chart(history: List[SimSnapshot], height: int = 280) -> go.Figure:
    """Multi-line time series of tie-line MW flows."""
    fig = go.Figure()
    if not history or not history[0].tie_line_flows:
        return _apply_dark(fig, height=height, title="Tie-Line Flows (MW)")

    tl_names = [tl["name"] for tl in history[0].tie_line_flows]
    steps = [s.step for s in history]
    palette = ["#FF6B6B", "#4ECDC4", "#FFE66D", "#AA00FF", "#00C853", "#FF9100", "#448AFF", "#FF1744"]

    for i, name in enumerate(tl_names):
        vals = []
        for s in history:
            matched = [tl for tl in s.tie_line_flows if tl["name"] == name]
            vals.append(matched[0]["p_from_mw"] if matched else 0)
        fig.add_trace(go.Scatter(x=steps, y=vals, mode="lines", name=name,
                                 line=dict(color=palette[i % len(palette)], width=2)))
    return _apply_dark(fig, height=height, title="Tie-Line Flows (MW)")


def soc_gauge(soc_pct: float, name: str = "BESS") -> go.Figure:
    """Circular gauge for battery SOC."""
    if soc_pct < 20:
        bar_color = "#FF5252"
    elif soc_pct < 40:
        bar_color = "#FFD600"
    elif soc_pct < 80:
        bar_color = "#00C853"
    else:
        bar_color = "#FFD600"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=soc_pct,
        title=dict(text=name, font=dict(size=13, color="#ccc")),
        number=dict(suffix="%", font=dict(color="#eee", size=22)),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor="#555"),
            bar=dict(color=bar_color),
            bgcolor="#1a1a2e",
            bordercolor="#333",
            steps=[
                dict(range=[0, 20], color="rgba(255,82,82,0.15)"),
                dict(range=[20, 80], color="rgba(0,200,83,0.08)"),
                dict(range=[80, 100], color="rgba(255,214,0,0.12)"),
            ],
        ),
    ))
    fig.update_layout(
        height=180, margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#ccc"),
    )
    return fig
