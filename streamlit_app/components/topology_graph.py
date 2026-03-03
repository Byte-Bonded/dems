"""
Interactive topology graph builder using Plotly.
Renders the 117-bus SuperGrid as an interactive scatter-network graph
with area-coloured nodes, DER markers, and loading-coloured edges.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import plotly.graph_objects as go

from streamlit_app.simulation.sim_engine import SimSnapshot

# ---------------------------------------------------------------------------
# Area colours (consistent with SuperGrid definition)
# ---------------------------------------------------------------------------
AREA_COLORS = {"A": "#FF6B6B", "B": "#4ECDC4", "C": "#FFE66D"}
AREA_OFFSETS = {"A": 0, "B": 39, "C": 78}
AREA_NAMES = {"A": "Area A (North)", "B": "Area B (West)", "C": "Area C (East)"}

# DER marker config  (symbol, colour, label_prefix)
DER_MARKERS = {
    "solar_pv":         ("circle-open",    "#FFD700", "☀"),
    "wind":             ("triangle-up",    "#00BFFF", "🌬"),
    "bess":             ("square",         "#00C853", "🔋"),
    "ev_charging":      ("diamond",        "#AA00FF", "⚡"),
    "demand_response":  ("pentagon",       "#FF6D00", "📉"),
}

SLACK_BUSES = {30, 69, 108}


def _loading_color(pct: float) -> str:
    if pct < 50:
        return "rgba(0,200,100,0.35)"
    if pct < 80:
        return "rgba(255,200,0,0.50)"
    return "rgba(255,60,60,0.70)"


def _bus_area(bus: int) -> str:
    if bus < 39:
        return "A"
    if bus < 78:
        return "B"
    return "C"


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------
def _build_networkx_graph(snap: SimSnapshot) -> nx.Graph:
    """Create a networkx graph from the snapshot's line results."""
    G = nx.Graph()
    # Add all 117 buses
    for bus in range(117):
        G.add_node(bus)
    # Add edges from line results
    for lr in snap.line_results:
        G.add_edge(lr["from_bus"], lr["to_bus"], **lr)
    return G


def _compute_layout(G: nx.Graph) -> Dict[int, Tuple[float, float]]:
    """
    Compute a grouped layout:
    - Area A: top-center cluster
    - Area B: bottom-left cluster
    - Area C: bottom-right cluster
    Then refine with spring layout for local separation.
    """
    # Initial positions seeded by area
    init_pos = {}
    # Area centres
    centres = {"A": (0, 3), "B": (-3, -1), "C": (3, -1)}
    import random
    rng = random.Random(42)
    for node in G.nodes:
        area = _bus_area(node)
        cx, cy = centres[area]
        init_pos[node] = (cx + rng.uniform(-1.5, 1.5), cy + rng.uniform(-1.5, 1.5))

    # Spring layout refines while keeping area clusters
    pos = nx.spring_layout(G, pos=init_pos, k=0.8, iterations=60, seed=42)
    return pos


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def build_topology_figure(
    snap: SimSnapshot,
    *,
    height: int = 600,
    selected_area: Optional[str] = None,
    show_der: bool = True,
    show_labels: bool = False,
) -> go.Figure:
    """
    Build a Plotly Figure of the grid topology from a SimSnapshot.

    Parameters
    ----------
    snap : SimSnapshot
    height : figure height in pixels
    selected_area : if set, only show buses in this area + tie-line stubs
    show_der : overlay DER markers
    show_labels : show bus number labels
    """
    G = _build_networkx_graph(snap)
    pos = _compute_layout(G)

    fig = go.Figure()

    # --- filter buses if area selected ---
    if selected_area:
        lo, hi = AREA_OFFSETS[selected_area], AREA_OFFSETS[selected_area] + 38
        visible_buses = set(range(lo, hi + 1))
        # also include tie-line neighbours
        for tl in snap.tie_line_flows:
            fb, tb = tl["from_bus"], tl["to_bus"]
            if fb in visible_buses or tb in visible_buses:
                visible_buses.add(fb)
                visible_buses.add(tb)
    else:
        visible_buses = set(range(117))

    # ---- edges ----
    for lr in snap.line_results:
        fb, tb = lr["from_bus"], lr["to_bus"]
        if fb not in visible_buses and tb not in visible_buses:
            continue
        x0, y0 = pos.get(fb, (0, 0))
        x1, y1 = pos.get(tb, (0, 0))
        loading = lr.get("loading_percent", 0)
        color = _loading_color(loading)

        # Tie-lines are dashed & thicker
        is_tie = lr.get("name", "").startswith("TL_")
        dash = "dash" if is_tie else "solid"
        width = 3 if is_tie else 1.2

        fig.add_trace(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode="lines",
            line=dict(color=color, width=width, dash=dash),
            hoverinfo="text",
            text=f"{'[TIE] ' if is_tie else ''}Bus {fb}→{tb} | {lr.get('p_from_mw',0):.1f} MW | {loading:.0f}%",
            showlegend=False,
        ))

    # ---- bus nodes ----
    for area_key, area_color in AREA_COLORS.items():
        lo = AREA_OFFSETS[area_key]
        hi = lo + 38
        bx, by, btext, bsize = [], [], [], []
        for bus in range(lo, hi + 1):
            if bus not in visible_buses:
                continue
            x, y = pos.get(bus, (0, 0))
            bx.append(x)
            by.append(y)
            vm = snap.bus_voltages.get(bus, 1.0)
            va = snap.bus_angles.get(bus, 0.0)
            btext.append(f"Bus {bus} ({AREA_NAMES[area_key]})<br>V={vm:.4f} pu | θ={va:.1f}°")
            base = 7
            dev = abs(vm - 1.0) * 120  # bigger if voltage deviation
            bsize.append(base + dev)

        fig.add_trace(go.Scatter(
            x=bx, y=by,
            mode="markers+text" if show_labels else "markers",
            text=[str(lo + i) for i in range(len(bx))] if show_labels else None,
            textposition="top center",
            textfont=dict(size=7, color="#ccc"),
            marker=dict(
                size=bsize,
                color=area_color,
                line=dict(width=0.5, color="#333"),
                opacity=0.9,
            ),
            hovertext=btext,
            hoverinfo="text",
            name=AREA_NAMES[area_key],
        ))

    # ---- slack bus rings ----
    for sb in SLACK_BUSES:
        if sb not in visible_buses:
            continue
        x, y = pos.get(sb, (0, 0))
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode="markers",
            marker=dict(size=16, color="rgba(0,0,0,0)", line=dict(width=2, color="#FFFFFF")),
            hoverinfo="text",
            hovertext=f"Slack Bus {sb}",
            showlegend=False,
        ))

    # ---- DER markers ----
    if show_der and snap.der_states:
        for ds in snap.der_states:
            bus = ds.get("bus")
            if bus is None or bus not in visible_buses:
                continue
            dt = ds.get("der_type", "").lower()
            sym, col, pfx = DER_MARKERS.get(dt, ("circle", "#ccc", "?"))
            x, y = pos.get(bus, (0, 0))
            # offset slightly so DER doesn't overlap bus dot
            x += 0.08
            y += 0.08
            hover_parts = [
                f"<b>{ds.get('name','')}</b> ({dt})",
                f"Bus {bus}",
                f"Output: {ds.get('current_output_mw',0):.2f} MW",
                f"Capacity: {ds.get('capacity_mw',0):.1f} MW",
            ]
            soc = ds.get("soc")
            if soc is not None:
                hover_parts.append(f"SOC: {soc:.1%}")
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode="markers",
                marker=dict(size=10, color=col, symbol=sym, line=dict(width=1, color="#222")),
                hoverinfo="text",
                hovertext="<br>".join(hover_parts),
                showlegend=False,
            ))

    # ---- layout ----
    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="center", x=0.5,
            font=dict(color="#ddd", size=11),
            bgcolor="rgba(0,0,0,0)",
        ),
        hoverlabel=dict(bgcolor="#1e1e2e", font_size=12, font_color="#eee"),
    )
    return fig
