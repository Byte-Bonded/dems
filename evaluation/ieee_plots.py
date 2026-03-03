"""
IEEE Publication-Ready Plotting Module for DEMS Evaluation

Generates figures conforming to IEEE two-column format:
    - Single column width:  3.5 in (88.9 mm)
    - Double column width:  7.16 in (181.9 mm)
    - Font: Times New Roman, 8pt labels, 9pt axis titles
    - Resolution: 300 DPI minimum
    - Accessible colour palette

Plots:
    1.  Bode plots (AVR, Governor, PSS, SMIB composite)
    2.  Nyquist diagram
    3.  Eigenvalue map (σ-jω plane)
    4.  Participation factor bar chart
    5.  Root locus (PSS gain sweep)
    6.  Frequency response to load step
    7.  Voltage profile heatmap
    8.  Rotor angle trajectories
    9.  Tie-line power flows
    10. Area Control Error (ACE)
    11. LMP heatmap
    12. Operating cost comparison
    13. DR price-response curve
    14. Renewable curtailment Pareto
    15. Training convergence curves
    16. Performance radar chart
    17. Action distribution heatmap
    18. Box plots comparison
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server/CI
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import rcParams
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


# ======================== IEEE FORMAT CONFIGURATION ======================== #

def setup_ieee_style():
    """Configure matplotlib for IEEE two-column publication format."""
    rcParams.update({
        # Font
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': 8,
        'axes.titlesize': 9,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 7,
        'figure.titlesize': 10,
        # Figure
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        # Lines
        'lines.linewidth': 1.0,
        'lines.markersize': 4,
        # Grid
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
        # Legend
        'legend.framealpha': 0.8,
        'legend.edgecolor': '0.8',
        # LaTeX
        'text.usetex': False,  # Set True if LaTeX is available
        'mathtext.fontset': 'stix',
    })


# IEEE accessible colour palette
IEEE_COLORS = {
    'rl': '#1f77b4',        # Blue
    'droop': '#ff7f0e',     # Orange
    'merit_order': '#2ca02c',  # Green
    'no_control': '#d62728',   # Red
    'pi_agc': '#9467bd',    # Purple
    'reference': '#7f7f7f', # Grey
}

# Column widths
SINGLE_COL = 3.5   # inches
DOUBLE_COL = 7.16   # inches


class IEEEPlotter:
    """Generator for all IEEE publication-quality figures."""

    def __init__(self, output_dir: str = "results/ieee_eval/figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        setup_ieee_style()

    def _save(self, fig, name: str, tight: bool = True):
        """Save figure in both PDF (vector) and PNG (raster) formats."""
        if tight:
            fig.tight_layout()
        fig.savefig(self.output_dir / f"{name}.pdf", format='pdf')
        fig.savefig(self.output_dir / f"{name}.png", format='png')
        plt.close(fig)
        logger.info(f"Saved figure: {name}")

    # ================================================================
    # 1. BODE PLOTS
    # ================================================================

    def plot_bode(
        self,
        bode_data_dict: Dict[str, Dict],
        margins_dict: Optional[Dict[str, Any]] = None,
        title: str = "Bode Diagram",
        filename: str = "fig01_bode",
    ):
        """Generate Bode plot (magnitude + phase) for multiple transfer functions.

        Args:
            bode_data_dict: {label: {'omega', 'magnitude_db', 'phase_deg'}}
            margins_dict:   {label: StabilityMargins} for margin annotations
        """
        fig, (ax_mag, ax_phase) = plt.subplots(
            2, 1, figsize=(SINGLE_COL, 3.5), sharex=True,
            gridspec_kw={'height_ratios': [1.2, 1]})

        colors = list(IEEE_COLORS.values())
        for i, (label, data) in enumerate(bode_data_dict.items()):
            c = colors[i % len(colors)]
            omega = data['omega']
            ax_mag.semilogx(omega, data['magnitude_db'], label=label, color=c)
            ax_phase.semilogx(omega, data['phase_deg'], color=c)

        # Magnitude plot
        ax_mag.set_ylabel('Magnitude (dB)')
        ax_mag.axhline(y=0, color='k', linewidth=0.5, linestyle='--')
        ax_mag.legend(loc='best')
        ax_mag.set_title(title)

        # Phase plot
        ax_phase.set_ylabel('Phase (deg)')
        ax_phase.set_xlabel('Frequency (rad/s)')
        ax_phase.axhline(y=-180, color='k', linewidth=0.5, linestyle='--')

        # Annotate margins
        if margins_dict:
            for label, margins in margins_dict.items():
                if hasattr(margins, 'gain_margin_db'):
                    gm = margins.gain_margin_db
                    pm = margins.phase_margin_deg
                    if np.isfinite(gm) and np.isfinite(pm):
                        ax_mag.annotate(
                            f'GM={gm:.1f}dB',
                            xy=(margins.phase_crossover_freq_rad, 0),
                            fontsize=6, color='red',
                            xytext=(5, 5), textcoords='offset points')
                        ax_phase.annotate(
                            f'PM={pm:.1f}°',
                            xy=(margins.gain_crossover_freq_rad, -180),
                            fontsize=6, color='red',
                            xytext=(5, 5), textcoords='offset points')

        self._save(fig, filename)

    # ================================================================
    # 2. NYQUIST DIAGRAM
    # ================================================================

    def plot_nyquist(
        self,
        nyquist_data_dict: Dict[str, Dict],
        title: str = "Nyquist Diagram",
        filename: str = "fig02_nyquist",
    ):
        """Generate Nyquist plot with unit circle."""
        fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL))

        colors = list(IEEE_COLORS.values())
        for i, (label, data) in enumerate(nyquist_data_dict.items()):
            c = colors[i % len(colors)]
            ax.plot(data['real'], data['imag'], label=label, color=c)
            # Mirror (negative frequencies)
            ax.plot(data['real'], -data['imag'], color=c, alpha=0.3, linestyle='--')

        # Unit circle
        circle = Circle((-1, 0), 0.02, fill=True, color='red', zorder=5)
        ax.add_patch(circle)
        ax.annotate('(-1, 0)', xy=(-1, 0), fontsize=6, color='red',
                     xytext=(-15, 10), textcoords='offset points')

        ax.set_xlabel('Real')
        ax.set_ylabel('Imaginary')
        ax.set_title(title)
        ax.legend(loc='best')
        ax.set_aspect('equal', adjustable='datalim')
        ax.axhline(y=0, color='k', linewidth=0.3)
        ax.axvline(x=0, color='k', linewidth=0.3)

        self._save(fig, filename)

    # ================================================================
    # 3. EIGENVALUE MAP
    # ================================================================

    def plot_eigenvalue_map(
        self,
        eigenvalues: np.ndarray,
        modes: Optional[List] = None,
        title: str = "Eigenvalue Map",
        filename: str = "fig03_eigenvalues",
    ):
        """Plot eigenvalues on σ-jω plane with damping ratio contours."""
        fig, ax = plt.subplots(figsize=(SINGLE_COL, 3.0))

        sigma = eigenvalues.real
        omega = eigenvalues.imag

        # Colour by mode category if available
        if modes:
            cat_colors = {
                'local_plant': '#1f77b4',
                'inter_area': '#d62728',
                'control': '#2ca02c',
                'overdamped': '#7f7f7f',
            }
            for mode in modes:
                ev = mode.eigenvalue
                c = cat_colors.get(mode.category, '#7f7f7f')
                ax.plot(ev.real, ev.imag, 'o', color=c, markersize=5, alpha=0.7)
                if ev.imag > 0:
                    ax.plot(ev.real, -ev.imag, 'o', color=c, markersize=5, alpha=0.3)

            # Legend for categories
            for cat, colour in cat_colors.items():
                ax.plot([], [], 'o', color=colour, label=cat.replace('_', ' ').title())
        else:
            ax.plot(sigma, omega, 'bx', markersize=5)

        # Damping ratio contours (ζ = 0.05, 0.10, 0.20)
        for zeta in [0.05, 0.10, 0.20]:
            theta = np.arccos(zeta)
            r = np.linspace(0, max(abs(omega.max()) if len(omega) > 0 else 10, 10), 100)
            sigma_line = -r * np.cos(theta)
            omega_line = r * np.sin(theta)
            ax.plot(sigma_line, omega_line, 'k--', linewidth=0.5, alpha=0.4)
            ax.plot(sigma_line, -omega_line, 'k--', linewidth=0.5, alpha=0.4)
            ax.annotate(f'ζ={zeta}', xy=(sigma_line[50], omega_line[50]),
                        fontsize=6, alpha=0.5)

        ax.axvline(x=0, color='k', linewidth=0.5)
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.set_xlabel('Real (σ) [1/s]')
        ax.set_ylabel('Imaginary (jω) [rad/s]')
        ax.set_title(title)
        ax.legend(loc='best', fontsize=6)

        self._save(fig, filename)

    # ================================================================
    # 4. PARTICIPATION FACTOR BAR CHART
    # ================================================================

    def plot_participation_factors(
        self,
        mode,
        state_names: List[str],
        participation_col: np.ndarray,
        title: str = "Participation Factors — Dominant Mode",
        filename: str = "fig04_participation",
    ):
        """Bar chart of participation factors for a single mode."""
        fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.5))

        # Show only top 10 participating states
        idx = np.argsort(participation_col)[::-1][:10]
        names = [state_names[i] for i in idx]
        values = participation_col[idx]

        bars = ax.barh(range(len(names)), values, color='#1f77b4', height=0.6)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=6)
        ax.set_xlabel('Participation Factor')
        ax.set_title(f"{title}\n(f={mode.frequency_hz:.3f} Hz, ζ={mode.damping_ratio:.3f})")
        ax.invert_yaxis()

        self._save(fig, filename)

    # ================================================================
    # 5. ROOT LOCUS (PSS GAIN SWEEP)
    # ================================================================

    def plot_root_locus_pss(
        self,
        gain_values: np.ndarray,
        eigenvalue_trajectories: List[np.ndarray],
        title: str = "Root Locus — PSS Gain Variation",
        filename: str = "fig05_root_locus",
    ):
        """Root locus as PSS gain varies from 0 to K_max."""
        fig, ax = plt.subplots(figsize=(SINGLE_COL, 3.0))

        cmap = plt.cm.viridis
        for i, (gain, evs) in enumerate(zip(gain_values, eigenvalue_trajectories)):
            color = cmap(i / max(len(gain_values) - 1, 1))
            ax.plot(evs.real, evs.imag, '.', color=color, markersize=2, alpha=0.6)

        # Colorbar for gain
        sm = plt.cm.ScalarMappable(
            cmap=cmap,
            norm=plt.Normalize(gain_values[0], gain_values[-1]))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label='PSS Gain (K)')
        cbar.ax.tick_params(labelsize=6)

        ax.axvline(x=0, color='k', linewidth=0.5)
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.set_xlabel('Real [1/s]')
        ax.set_ylabel('Imaginary [rad/s]')
        ax.set_title(title)

        self._save(fig, filename)

    # ================================================================
    # 6. FREQUENCY RESPONSE COMPARISON
    # ================================================================

    def plot_frequency_response(
        self,
        results: Dict[str, List],
        dt_s: float = 300.0,
        title: str = "System Frequency Response",
        filename: str = "fig06_frequency",
    ):
        """Plot system frequency over time for multiple controllers.

        Args:
            results: {controller_name: [TimestepMetrics, ...]}
        """
        fig, ax = plt.subplots(figsize=(DOUBLE_COL, 2.5))

        for ctrl, timesteps in results.items():
            t = np.array([ts.step * dt_s / 3600.0 for ts in timesteps])
            f = np.array([ts.frequency_hz for ts in timesteps])
            color = IEEE_COLORS.get(ctrl, '#333333')
            ax.plot(t, f, label=ctrl.replace('_', ' ').title(), color=color)

        # Operating band
        ax.axhline(y=50.0, color='k', linewidth=0.5, linestyle='-')
        ax.axhline(y=49.5, color='red', linewidth=0.5, linestyle='--', alpha=0.5)
        ax.axhline(y=50.5, color='red', linewidth=0.5, linestyle='--', alpha=0.5)
        ax.fill_between(ax.get_xlim(), 49.5, 50.5, alpha=0.05, color='green')

        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title(title)
        ax.legend(loc='best')

        self._save(fig, filename)

    # ================================================================
    # 7. VOLTAGE PROFILE HEATMAP
    # ================================================================

    def plot_voltage_heatmap(
        self,
        voltage_data: np.ndarray,
        bus_labels: Optional[List[str]] = None,
        dt_s: float = 300.0,
        title: str = "Bus Voltage Profile (24h)",
        filename: str = "fig07_voltage_heatmap",
    ):
        """Heatmap of bus voltages over time.

        Args:
            voltage_data: Shape (n_buses, n_steps) of voltage magnitudes (pu).
        """
        fig, ax = plt.subplots(figsize=(DOUBLE_COL, 3.0))

        n_buses, n_steps = voltage_data.shape
        hours = np.arange(n_steps) * dt_s / 3600.0

        im = ax.imshow(voltage_data, aspect='auto', cmap='RdYlGn',
                        vmin=0.90, vmax=1.10,
                        extent=[0, hours[-1], n_buses, 0])

        cbar = plt.colorbar(im, ax=ax, label='Voltage (pu)')
        cbar.ax.tick_params(labelsize=6)

        # Threshold lines
        # (drawn as contour if data allows)
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Bus Index')
        ax.set_title(title)

        self._save(fig, filename)

    # ================================================================
    # 8. ROTOR ANGLE TRAJECTORIES
    # ================================================================

    def plot_rotor_angles(
        self,
        angle_data: Dict[str, np.ndarray],
        time_s: np.ndarray,
        title: str = "Generator Rotor Angles — N-1 Contingency",
        filename: str = "fig08_rotor_angles",
    ):
        """Plot generator rotor angle trajectories."""
        fig, ax = plt.subplots(figsize=(DOUBLE_COL, 2.5))

        for gen_id, angles in angle_data.items():
            ax.plot(time_s, angles, label=gen_id, linewidth=0.8)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Rotor Angle (degrees)')
        ax.set_title(title)
        ax.legend(loc='best', ncol=3, fontsize=5)

        self._save(fig, filename)

    # ================================================================
    # 9. TIE-LINE POWER FLOWS
    # ================================================================

    def plot_tie_lines(
        self,
        results: Dict[str, List],
        dt_s: float = 300.0,
        title: str = "Inter-Area Tie-Line Flows",
        filename: str = "fig09_tie_lines",
    ):
        """Plot tie-line flows for RL vs baseline."""
        fig, axes = plt.subplots(3, 1, figsize=(DOUBLE_COL, 4.0), sharex=True)

        tie_names = ['AB', 'BC', 'AC']
        tie_keys = ['tie_ab_flow_mw', 'tie_bc_flow_mw', 'tie_ac_flow_mw']

        for ax, tie_name, key in zip(axes, tie_names, tie_keys):
            for ctrl, timesteps in results.items():
                t = np.array([ts.step * dt_s / 3600.0 for ts in timesteps])
                flows = np.array([getattr(ts, key, 0.0) for ts in timesteps])
                color = IEEE_COLORS.get(ctrl, '#333333')
                ax.plot(t, flows, label=ctrl.replace('_', ' ').title(),
                        color=color, linewidth=0.8)
            ax.set_ylabel(f'{tie_name} (MW)')
            if ax == axes[0]:
                ax.legend(loc='best', fontsize=6)

        axes[-1].set_xlabel('Time (hours)')
        axes[0].set_title(title)

        self._save(fig, filename)

    # ================================================================
    # 10. AREA CONTROL ERROR
    # ================================================================

    def plot_ace(
        self,
        results: Dict[str, List],
        dt_s: float = 300.0,
        title: str = "Area Control Error (ACE)",
        filename: str = "fig10_ace",
    ):
        """Plot ACE = Δf_dev × β for each controller."""
        fig, ax = plt.subplots(figsize=(DOUBLE_COL, 2.0))

        beta = 1000.0  # MW/Hz
        for ctrl, timesteps in results.items():
            t = np.array([ts.step * dt_s / 3600.0 for ts in timesteps])
            ace = np.array([(ts.frequency_hz - 50.0) * beta for ts in timesteps])
            color = IEEE_COLORS.get(ctrl, '#333333')
            ax.plot(t, ace, label=ctrl.replace('_', ' ').title(),
                    color=color, linewidth=0.8)

        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('ACE (MW)')
        ax.set_title(title)
        ax.legend(loc='best')

        self._save(fig, filename)

    # ================================================================
    # 11. LMP HEATMAP
    # ================================================================

    def plot_lmp_heatmap(
        self,
        lmp_data: Dict[str, Dict[int, float]],
        title: str = "Locational Marginal Prices",
        filename: str = "fig11_lmp",
    ):
        """Plot LMP across buses for peak/off-peak.

        Args:
            lmp_data: {'peak': {bus: lmp}, 'off_peak': {bus: lmp}}
        """
        fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.5), sharey=True)

        for ax, (period, lmps) in zip(axes, lmp_data.items()):
            buses = sorted(lmps.keys())
            prices = [lmps[b] for b in buses]
            ax.barh(range(len(buses)), prices, color='#1f77b4', height=0.8)
            ax.set_xlabel('LMP ($/MWh)')
            ax.set_title(period.replace('_', ' ').title())
            if ax == axes[0]:
                ax.set_ylabel('Bus Index')

        fig.suptitle(title, fontsize=9)
        self._save(fig, filename)

    # ================================================================
    # 12. OPERATING COST COMPARISON
    # ================================================================

    def plot_cost_comparison(
        self,
        results: Dict[str, List[float]],
        title: str = "Total Operating Cost Comparison",
        filename: str = "fig12_cost",
    ):
        """Bar chart with error bars: cost per controller.

        Args:
            results: {controller_name: [cost_per_episode, ...]}
        """
        fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.5))

        names = list(results.keys())
        means = [np.mean(v) for v in results.values()]
        stds = [np.std(v) for v in results.values()]
        colors = [IEEE_COLORS.get(n, '#333333') for n in names]

        bars = ax.bar(range(len(names)), means, yerr=stds, capsize=3,
                       color=colors, edgecolor='black', linewidth=0.5)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.replace('_', ' ').title() for n in names],
                           rotation=30, ha='right', fontsize=7)
        ax.set_ylabel('Operating Cost ($/day)')
        ax.set_title(title)

        # Value labels
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'${mean:.0f}', ha='center', va='bottom', fontsize=6)

        self._save(fig, filename)

    # ================================================================
    # 13. DR PRICE–RESPONSE CURVE
    # ================================================================

    def plot_dr_price_response(
        self,
        prices: np.ndarray,
        load_reductions: np.ndarray,
        title: str = "Demand Response: Price vs Load Reduction",
        filename: str = "fig13_dr_price",
    ):
        """Plot how load reduction responds to price signals."""
        fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.5))

        ax.plot(prices, load_reductions, 'o-', color='#1f77b4', markersize=4)
        ax.set_xlabel('Electricity Price ($/MWh)')
        ax.set_ylabel('Load Reduction (MW)')
        ax.set_title(title)

        self._save(fig, filename)

    # ================================================================
    # 14. RENEWABLE CURTAILMENT PARETO
    # ================================================================

    def plot_pareto_frontier(
        self,
        costs: np.ndarray,
        curtailments: np.ndarray,
        labels: Optional[List[str]] = None,
        title: str = "Cost vs Renewable Curtailment Pareto Frontier",
        filename: str = "fig14_pareto",
    ):
        """Pareto frontier: operating cost vs renewable curtailment."""
        fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.5))

        ax.scatter(curtailments, costs, c='#1f77b4', s=20, alpha=0.6)

        # Highlight Pareto-optimal points
        pareto = _pareto_front(curtailments, costs)
        ax.plot(curtailments[pareto], costs[pareto], 'r-o',
                markersize=4, linewidth=1.0, label='Pareto front')

        if labels:
            for i, lbl in enumerate(labels):
                ax.annotate(lbl, (curtailments[i], costs[i]), fontsize=5,
                            xytext=(3, 3), textcoords='offset points')

        ax.set_xlabel('Renewable Curtailment (MWh)')
        ax.set_ylabel('Operating Cost ($/day)')
        ax.set_title(title)
        ax.legend(loc='best')

        self._save(fig, filename)

    # ================================================================
    # 15. TRAINING CONVERGENCE
    # ================================================================

    def plot_training_convergence(
        self,
        rewards: Dict[str, np.ndarray],
        window: int = 20,
        title: str = "Training Convergence — 13-Agent Hierarchy",
        filename: str = "fig15_convergence",
    ):
        """Multi-agent reward curves with smoothing and confidence bands.

        Args:
            rewards: {agent_name: episode_rewards_array}
        """
        fig, ax = plt.subplots(figsize=(DOUBLE_COL, 3.0))

        for agent_name, r in rewards.items():
            if len(r) < window:
                ax.plot(r, label=agent_name, linewidth=0.8)
                continue

            # Smoothed mean
            smooth = np.convolve(r, np.ones(window) / window, mode='valid')
            episodes = np.arange(len(smooth))

            # 95% confidence band (rolling std)
            rolling_std = np.array([
                np.std(r[max(0, i):i + window])
                for i in range(len(smooth))
            ])
            ci = 1.96 * rolling_std / np.sqrt(window)

            color = None  # Let matplotlib auto-assign
            line, = ax.plot(episodes, smooth, label=agent_name, linewidth=0.8)
            ax.fill_between(episodes, smooth - ci, smooth + ci,
                            alpha=0.15, color=line.get_color())

        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title(title)
        ax.legend(loc='best', ncol=3, fontsize=5)

        self._save(fig, filename)

    # ================================================================
    # 16. PERFORMANCE RADAR CHART
    # ================================================================

    def plot_radar_chart(
        self,
        metrics: Dict[str, Dict[str, float]],
        title: str = "Performance Comparison — Radar Chart",
        filename: str = "fig16_radar",
    ):
        """Radar/spider chart comparing controllers across metrics.

        Args:
            metrics: {controller: {metric_name: normalised_value_0_to_1}}
        """
        categories = list(next(iter(metrics.values())).keys())
        N = len(categories)

        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL),
                                subplot_kw=dict(polar=True))

        for ctrl, vals in metrics.items():
            values = [vals[cat] for cat in categories]
            values += values[:1]
            color = IEEE_COLORS.get(ctrl, '#333333')
            ax.plot(angles, values, 'o-', label=ctrl.replace('_', ' ').title(),
                    color=color, linewidth=1.0, markersize=3)
            ax.fill(angles, values, alpha=0.1, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=6)
        ax.set_ylim(0, 1)
        ax.set_title(title, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=6)

        self._save(fig, filename, tight=False)

    # ================================================================
    # 17. ACTION DISTRIBUTION HEATMAP
    # ================================================================

    def plot_action_heatmap(
        self,
        actions: np.ndarray,
        agent_labels: Optional[List[str]] = None,
        dt_s: float = 300.0,
        title: str = "Agent Action Distribution (24h)",
        filename: str = "fig17_actions",
    ):
        """Heatmap of agent actions over time.

        Args:
            actions: Shape (n_agents, n_steps) of actions in [0, 1].
        """
        fig, ax = plt.subplots(figsize=(DOUBLE_COL, 2.5))

        n_agents, n_steps = actions.shape
        hours = np.arange(n_steps) * dt_s / 3600.0

        im = ax.imshow(actions, aspect='auto', cmap='viridis',
                        vmin=0, vmax=1,
                        extent=[0, hours[-1], n_agents, 0])

        cbar = plt.colorbar(im, ax=ax, label='Action Value')
        cbar.ax.tick_params(labelsize=6)

        if agent_labels:
            ax.set_yticks(np.arange(n_agents) + 0.5)
            ax.set_yticklabels(agent_labels, fontsize=5)

        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Agent')
        ax.set_title(title)

        self._save(fig, filename)

    # ================================================================
    # 18. BOX PLOT COMPARISON
    # ================================================================

    def plot_box_comparison(
        self,
        data: Dict[str, Dict[str, List[float]]],
        title: str = "Performance Distribution Comparison",
        filename: str = "fig18_boxplots",
    ):
        """Box plots of key metrics across controllers.

        Args:
            data: {metric_name: {controller_name: [values]}}
        """
        metrics = list(data.keys())
        n_metrics = len(metrics)
        controllers = list(next(iter(data.values())).keys())

        fig, axes = plt.subplots(1, n_metrics, figsize=(DOUBLE_COL, 2.5))
        if n_metrics == 1:
            axes = [axes]

        for ax, metric_name in zip(axes, metrics):
            box_data = [data[metric_name].get(c, []) for c in controllers]
            colors = [IEEE_COLORS.get(c, '#333333') for c in controllers]

            bp = ax.boxplot(box_data, patch_artist=True, widths=0.6)
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            ax.set_xticklabels([c.replace('_', ' ').title() for c in controllers],
                               rotation=45, ha='right', fontsize=5)
            ax.set_title(metric_name, fontsize=7)

        fig.suptitle(title, fontsize=9)
        self._save(fig, filename)

    # ================================================================
    # STEP RESPONSE PLOT
    # ================================================================

    def plot_step_response(
        self,
        response_data: Dict[str, Dict],
        title: str = "Step Response",
        filename: str = "fig_step_response",
    ):
        """Plot step response with transient metric annotations.

        Args:
            response_data: {label: {'time', 'response', 'rise_time',
                           'settling_time', 'overshoot_pct', 'steady_state'}}
        """
        fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.5))

        for label, data in response_data.items():
            ax.plot(data['time'], data['response'], label=label, linewidth=1.0)

            # Annotate transient metrics for first entry
            if label == list(response_data.keys())[0]:
                ss = data['steady_state']
                if abs(ss) > 1e-10:
                    ax.axhline(y=ss, color='k', linewidth=0.3, linestyle='--')
                    ax.axhline(y=ss * 1.02, color='red', linewidth=0.3, linestyle=':')
                    ax.axhline(y=ss * 0.98, color='red', linewidth=0.3, linestyle=':')
                    ax.annotate(f'ts={data["settling_time"]:.2f}s',
                                xy=(data['settling_time'], ss),
                                fontsize=6, color='red')

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Response')
        ax.set_title(title)
        ax.legend(loc='best')

        self._save(fig, filename)

    # ================================================================
    # IEEE TABLE I GENERATION
    # ================================================================

    def generate_comparison_table_latex(
        self,
        table: Dict[str, Dict[str, str]],
        caption: str = "Performance Comparison Under Base Case Scenario",
        label: str = "tab:comparison",
        filename: str = "table_I_comparison",
    ) -> str:
        """Generate LaTeX table for IEEE paper.

        Args:
            table: {metric: {controller: "mean ± std"}}

        Returns:
            LaTeX string.
        """
        controllers = list(next(iter(table.values())).keys())
        metrics = list(table.keys())

        # Build LaTeX
        cols = 'l' + 'c' * len(controllers)
        header = ' & '.join(['Metric'] + [c.replace('_', ' ').title() for c in controllers])

        lines = [
            '\\begin{table}[!t]',
            '\\caption{' + caption + '}',
            '\\label{' + label + '}',
            '\\centering',
            '\\begin{tabular}{' + cols + '}',
            '\\hline',
            header + ' \\\\',
            '\\hline',
        ]

        for metric in metrics:
            row = [metric]
            for ctrl in controllers:
                row.append(table[metric].get(ctrl, '--'))
            lines.append(' & '.join(row) + ' \\\\')

        lines.extend([
            '\\hline',
            '\\end{tabular}',
            '\\end{table}',
        ])

        latex = '\n'.join(lines)

        # Save to file
        output_path = self.output_dir / f"{filename}.tex"
        with open(output_path, 'w') as f:
            f.write(latex)

        logger.info(f"Saved LaTeX table: {output_path}")
        return latex


# ======================== UTILITIES ======================== #

def _pareto_front(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Find indices of Pareto-optimal points (minimise both x and y)."""
    is_pareto = np.ones(len(x), dtype=bool)
    for i in range(len(x)):
        for j in range(len(x)):
            if i != j:
                if x[j] <= x[i] and y[j] <= y[i] and (x[j] < x[i] or y[j] < y[i]):
                    is_pareto[i] = False
                    break
    return np.where(is_pareto)[0]
