#!/usr/bin/env python3
"""
IEEE-Compliant Evaluation Runner for DEMS SuperGrid

Orchestrates the complete evaluation pipeline:
1. Loads trained RL model or instantiates baseline controllers
2. Runs N episodes across K scenarios
3. Collects per-timestep metrics via MetricsCollector
4. Generates all IEEE publication figures via IEEEPlotter
5. Produces LaTeX comparison tables (IEEE Table I format)
6. Runs small-signal stability and transfer-function analysis
7. Outputs consolidated JSON results

Usage:
    # Evaluate trained RL model
    python -m evaluation.run_eval --model logs/rl/best --scenarios all --episodes 10

    # Evaluate specific baseline
    python -m evaluation.run_eval --baseline droop --scenarios base_case n1_generator

    # Full comparative evaluation (RL + all baselines)
    python -m evaluation.run_eval --model logs/rl/best --compare-all --episodes 50

    # Only stability analysis (Bode/Nyquist/eigenvalue, no episodes)
    python -m evaluation.run_eval --stability-only

    # Quick smoke test
    python -m evaluation.run_eval --model logs/rl/best --scenarios base_case --episodes 1
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Project imports ──────────────────────────────────────────────────────
# Add project root to path so `src.*` and `evaluation.*` resolve
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from evaluation.metrics_collector import MetricsCollector, EpisodeMetrics, compute_comparison_table
from evaluation.scenarios import (
    ScenarioDefinition,
    get_all_scenarios,
    get_scenario_by_name,
    apply_scenario_event,
)
from evaluation.baselines import (
    BaselineController,
    NoControlBaseline,
    DroopController,
    MeritOrderController,
    PIAGCController,
)
from evaluation.ieee_plots import IEEEPlotter

logger = logging.getLogger("evaluation")


# ═══════════════════════════════════════════════════════════════════════════
# Controller wrappers — unified interface for RL agents and baselines
# ═══════════════════════════════════════════════════════════════════════════

class RLControllerWrapper:
    """Wraps the trained hierarchical RL agents for evaluation."""

    def __init__(self, model_dir: str, device: str = "cpu"):
        self.name = "rl_ppo"
        self.model_dir = Path(model_dir)
        self.device = device
        self.agents: Dict = {}
        self._loaded = False

    def load(self, coordinator):
        """Load all trained agents from the model directory."""
        from src.agent.ppo_agents import PPOAgentWrapper

        # Try to load each agent
        agent_names = coordinator.all_env_names
        for agent_name in agent_names:
            model_path = self.model_dir / agent_name
            zip_path = model_path.with_suffix(".zip")
            if zip_path.exists():
                env = coordinator.get_env(agent_name)
                try:
                    agent = PPOAgentWrapper.from_pretrained(
                        str(model_path), env=env, name=agent_name, device=self.device
                    )
                    self.agents[agent_name] = agent
                    logger.info(f"  Loaded RL agent: {agent_name}")
                except Exception as e:
                    logger.warning(f"  Failed to load {agent_name}: {e}")
            else:
                logger.debug(f"  No checkpoint for {agent_name} at {zip_path}")

        self._loaded = len(self.agents) > 0
        if not self._loaded:
            logger.warning(
                f"No RL agents loaded from {self.model_dir}. "
                "Falling back to random actions."
            )
        return self._loaded

    def select_actions(self, coordinator, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Select deterministic actions from all loaded agents."""
        actions = {}
        for agent_name, ob in obs.items():
            if agent_name in self.agents:
                actions[agent_name] = self.agents[agent_name].predict(ob, deterministic=True)
            else:
                # Random for missing agents
                env = coordinator.get_env(agent_name)
                actions[agent_name] = env.action_space.sample()
        return actions

    def reset(self):
        pass


class BaselineControllerWrapper:
    """Adapts a BaselineController to the same interface as RLControllerWrapper."""

    def __init__(self, baseline: BaselineController):
        self.name = baseline.name
        self.baseline = baseline

    def load(self, coordinator):
        return True  # Baselines don't need loading

    def select_actions(self, coordinator, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Generate actions from the baseline controller."""
        engine = coordinator.physics

        # Build area_states from physics engine
        area_states = {}
        for area_id in coordinator.AREA_IDS:
            try:
                area_states[area_id.value] = engine.get_area_state(area_id)
            except Exception:
                area_states[area_id.value] = {}

        # DER status
        try:
            der_status = engine.get_der_status()
        except Exception:
            der_status = {}

        # Tie-line flows
        try:
            tie_line_flows = engine.get_tie_line_flows()
        except Exception:
            tie_line_flows = []

        step = coordinator._step_count

        actions = self.baseline.select_actions(
            engine=engine,
            step=step,
            area_states=area_states,
            der_status=der_status,
            tie_line_flows=tie_line_flows,
        )

        # Ensure all agent names are present (sub-agents get zeros)
        for agent_name in coordinator.all_env_names:
            if agent_name not in actions:
                env = coordinator.get_env(agent_name)
                actions[agent_name] = np.zeros(env.action_space.shape[0])

        return actions

    def reset(self):
        self.baseline.reset()


# ═══════════════════════════════════════════════════════════════════════════
# Episode runner
# ═══════════════════════════════════════════════════════════════════════════

def run_episode(
    coordinator,
    controller,
    scenario: ScenarioDefinition,
    collector: MetricsCollector,
    episode_id: int = 0,
) -> EpisodeMetrics:
    """Run a single evaluation episode.

    Args:
        coordinator: MultiAgentStepCoordinator instance.
        controller: RLControllerWrapper or BaselineControllerWrapper.
        scenario: ScenarioDefinition describing the test case.
        collector: MetricsCollector for this episode.
        episode_id: Episode number.

    Returns:
        EpisodeMetrics with full timestep data.
    """
    # Reset
    collector.reset()
    controller.reset()
    obs = coordinator.reset()

    engine = coordinator.physics
    n_steps = scenario.episode_length_steps

    for step in range(1, n_steps + 1):
        # Check for scenario events at this step
        for event in scenario.events:
            if event.get('step') == step:
                try:
                    apply_scenario_event(engine, event)
                    logger.debug(f"  Applied event at step {step}: {event.get('type')}")
                except Exception as e:
                    logger.warning(f"  Failed to apply event at step {step}: {e}")

        # Select actions
        actions = controller.select_actions(coordinator, obs)

        # Step the environment
        obs, rewards, done, info = coordinator.step(actions)

        # Compute hour of day
        hour = (step * scenario.dt_control_s / 3600.0) % 24.0

        # Extract state information for metrics
        area_states = {}
        for area_id in coordinator.AREA_IDS:
            try:
                area_states[area_id.value] = engine.get_area_state(area_id)
            except Exception:
                area_states[area_id.value] = {}

        try:
            der_status = engine.get_der_status()
        except Exception:
            der_status = {}

        try:
            tie_line_flows = engine.get_tie_line_flows()
        except Exception:
            tie_line_flows = []

        # Build step_info from physics
        step_info = info.get('physics', {})
        step_info['frequency_hz'] = getattr(engine, 'system_frequency_hz', 50.0)

        # Flatten action vector for smoothness computation
        all_actions = np.concatenate([
            actions.get(name, np.array([0.0]))
            for name in sorted(actions.keys())
        ])

        collector.collect_step(
            step=step,
            step_info=step_info,
            area_states=area_states,
            der_status=der_status,
            tie_line_flows=tie_line_flows,
            rewards=rewards,
            actions=all_actions,
            hour=hour,
        )

        if done:
            break

    return collector.finalize_episode(
        scenario_name=scenario.name,
        controller_name=controller.name,
        episode_id=episode_id,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Stability analysis (Bode, Nyquist, eigenvalue, step response)
# ═══════════════════════════════════════════════════════════════════════════

def run_stability_analysis(plotter: IEEEPlotter, output_dir: Path):
    """Run transfer-function and small-signal stability analysis.

    Generates IEEE figures for:
    - Bode plots (AVR, Governor, PSS, SMIB composite)
    - Nyquist diagram (SMIB)
    - Eigenvalue map (multi-machine)
    - Participation factors
    - Root locus (PSS gain sweep)
    - Step response

    No simulation needed — uses analytical models.
    """
    logger.info("=" * 60)
    logger.info("Running stability analysis (transfer functions + eigenvalues)")
    logger.info("=" * 60)

    from src.simulation.transfer_functions import (
        avr_transfer_function,
        governor_transfer_function,
        pss_transfer_function,
        agc_transfer_function,
        smib_composite_model,
        bode_data,
        nyquist_data,
        step_response_data,
    )
    from src.simulation.small_signal import SmallSignalAnalyzer, quick_eigenvalue_analysis

    results = {}

    # ── 1. Transfer function Bode plots ──────────────────────────────

    logger.info("Computing transfer functions ...")
    avr_tfs = avr_transfer_function()
    gov_tfs = governor_transfer_function()
    pss_tf = pss_transfer_function()
    agc_tf = agc_transfer_function()
    smib = smib_composite_model()

    # Bode: individual loops
    bode_dict = {}
    margins_dict = {}

    # AVR loop gain
    avr_loop = avr_tfs['loop']
    bode_dict["AVR Open-Loop"] = bode_data(avr_loop)

    # Governor open-loop
    gov_ol = gov_tfs['open_loop']
    bode_dict["Governor"] = bode_data(gov_ol)

    # PSS
    bode_dict["PSS"] = bode_data(pss_tf)

    # SMIB governor-swing composite
    if 'gov_swing_loop' in smib:
        bode_dict["Gov-Swing Composite"] = bode_data(smib['gov_swing_loop'])

    # Margins from SMIB model
    if 'margins' in smib:
        smib_margins = smib['margins']
        margins_dict["AVR Open-Loop"] = smib_margins.get('avr')
        margins_dict["Governor"] = smib_margins.get('governor')
        margins_dict["Gov-Swing Composite"] = smib_margins.get('gov_swing')

    plotter.plot_bode(bode_dict, margins_dict,
                      title="Control Loop Bode Diagrams",
                      filename="fig01_bode_all_loops")

    # Separate SMIB Gov-Swing Bode with margins annotated
    if 'gov_swing_loop' in smib:
        gs_bode = {"Gov-Swing": bode_data(smib['gov_swing_loop'])}
        gs_margins = {"Gov-Swing": smib['margins'].get('gov_swing')} if 'margins' in smib else {}
        plotter.plot_bode(gs_bode, gs_margins,
                          title="SMIB Gov-Swing Composite Bode Diagram",
                          filename="fig01b_bode_smib")

    results['transfer_functions'] = {
        'avr_keys': list(avr_tfs.keys()),
        'governor_keys': list(gov_tfs.keys()),
    }
    if 'margins' in smib:
        sm = smib['margins']
        results['transfer_functions']['margins'] = {
            sub: {
                'gain_margin_db': float(sm[sub].gain_margin_db),
                'phase_margin_deg': float(sm[sub].phase_margin_deg),
                'ieee_compliant': bool(sm[sub].ieee_compliant),
            }
            for sub in sm
        }

    # ── 2. Nyquist diagram ───────────────────────────────────────────

    logger.info("Generating Nyquist diagram ...")
    nyquist_dict = {}
    # Use SMIB gov-swing loop and AVR loop
    if 'gov_swing_loop' in smib:
        nd = nyquist_data(smib['gov_swing_loop'])
        nyquist_dict["Gov-Swing"] = nd
    nd_avr = nyquist_data(avr_tfs['loop'])
    nyquist_dict["AVR"] = nd_avr

    plotter.plot_nyquist(nyquist_dict,
                         title="Nyquist Stability Diagram",
                         filename="fig02_nyquist")

    # ── 3. Step response ─────────────────────────────────────────────

    logger.info("Generating step responses ...")
    step_dict = {}
    # AVR closed-loop step response
    if 'closed_loop' in avr_tfs:
        step_dict["AVR"] = step_response_data(avr_tfs['closed_loop'], t_max=10.0)
    # Governor open-loop step response
    if 'open_loop' in gov_tfs:
        step_dict["Governor"] = step_response_data(gov_tfs['open_loop'], t_max=10.0)
    # Swing equation step response
    if 'swing' in smib:
        step_dict["Swing"] = step_response_data(smib['swing'], t_max=10.0)

    plotter.plot_step_response(step_dict,
                               title="Closed-Loop Step Response",
                               filename="fig18_step_response")

    # ── 4. Eigenvalue analysis ───────────────────────────────────────

    logger.info("Running eigenvalue analysis ...")
    try:
        ssa_result = quick_eigenvalue_analysis(n_generators=30)

        # Eigenvalue map
        eig_data = {
            'eigenvalues': ssa_result.eigenvalues,
            'modes': ssa_result.modes,
        }
        plotter.plot_eigenvalue_map(
            eig_data,
            title="Small-Signal Eigenvalue Map (30-gen SuperGrid)",
            filename="fig03_eigenvalue_map",
        )

        # Participation factors for poorly damped modes
        poorly_damped = [m for m in ssa_result.modes if m.damping_ratio < 0.1 and m.damping_ratio > 0]
        if poorly_damped:
            # Pick the most critical mode
            critical_mode = min(poorly_damped, key=lambda m: m.damping_ratio)
            plotter.plot_participation_factors(
                critical_mode.participation_factors,
                mode_label=f"{critical_mode.frequency_hz:.2f} Hz (ζ={critical_mode.damping_ratio:.3f})",
                title="Participation Factors — Critical Mode",
                filename="fig04_participation_factors",
            )

        # Root locus for PSS gain sweep
        logger.info("Computing PSS root locus ...")
        pss_gains = np.linspace(0.5, 20.0, 20)
        root_locus_data = []
        for K_pss in pss_gains:
            try:
                result_k = quick_eigenvalue_analysis(
                    n_generators=30,
                    K_pss=K_pss,
                )
                root_locus_data.append({
                    'gain': K_pss,
                    'eigenvalues': result_k.eigenvalues,
                })
            except Exception:
                continue

        if root_locus_data:
            plotter.plot_root_locus_pss(
                root_locus_data,
                title="Root Locus — PSS Gain Sweep",
                filename="fig05_root_locus_pss",
            )

        results['eigenvalue_analysis'] = {
            'n_eigenvalues': len(ssa_result.eigenvalues),
            'n_unstable': ssa_result.n_unstable,
            'n_poorly_damped': ssa_result.n_poorly_damped,
            'min_damping_ratio': float(ssa_result.min_damping_ratio),
            'dominant_inter_area_freq_hz': float(ssa_result.dominant_inter_area_freq_hz),
            'stable': ssa_result.stable,
            'modes': [
                {
                    'freq_hz': m.frequency_hz,
                    'damping': m.damping_ratio,
                    'category': m.category,
                    'stable': m.stable,
                }
                for m in ssa_result.modes[:20]  # Top 20 modes
            ],
        }

    except Exception as e:
        logger.error(f"Eigenvalue analysis failed: {e}", exc_info=True)
        results['eigenvalue_analysis'] = {'error': str(e)}

    logger.info("Stability analysis complete.")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Episode-based evaluation loop
# ═══════════════════════════════════════════════════════════════════════════

def run_evaluation(
    controllers: List,
    scenarios: List[ScenarioDefinition],
    n_episodes: int = 10,
    output_dir: str = "results/ieee_eval",
    episode_length: int = 288,
    seed: int = 42,
) -> Dict[str, Dict[str, List[EpisodeMetrics]]]:
    """Run full comparative evaluation across controllers and scenarios.

    Args:
        controllers: List of RLControllerWrapper / BaselineControllerWrapper.
        scenarios: List of ScenarioDefinition.
        n_episodes: Number of episodes per (controller, scenario) pair.
        output_dir: Root output directory.
        episode_length: Default episode length if not overridden by scenario.
        seed: Random seed.

    Returns:
        Nested dict: {controller_name: {scenario_name: [EpisodeMetrics]}}.
    """
    from src.agent.coordinator import MultiAgentStepCoordinator
    from src.simulation.orchestrator import ScenarioConfig
    from src.simulation.economics import EconomicsEngine

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics_dir = out / "metrics"
    metrics_dir.mkdir(exist_ok=True)

    all_results: Dict[str, Dict[str, List[EpisodeMetrics]]] = {}

    for ctrl in controllers:
        ctrl_name = ctrl.name
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating controller: {ctrl_name}")
        logger.info(f"{'='*60}")

        all_results[ctrl_name] = {}

        for scenario in scenarios:
            logger.info(f"\n  Scenario: {scenario.name} ({scenario.description})")
            logger.info(f"  IEEE Ref: {scenario.ieee_reference}")

            episode_metrics_list: List[EpisodeMetrics] = []

            for ep in range(n_episodes):
                ep_seed = seed + ep * 1000
                t0 = time.time()

                # Create a fresh coordinator for each episode
                sc = ScenarioConfig(
                    episode_length_steps=scenario.episode_length_steps,
                    dt_control_s=scenario.dt_control_s,
                )
                coordinator = MultiAgentStepCoordinator(
                    scenario=sc,
                    seed=ep_seed,
                    enable_sub_agents=True,
                )

                # Load controller if needed
                if isinstance(ctrl, RLControllerWrapper) and not ctrl._loaded:
                    ctrl.load(coordinator)

                # Create economics engine
                try:
                    econ = EconomicsEngine(net=coordinator.physics._net)
                except Exception:
                    econ = EconomicsEngine()

                collector = MetricsCollector(economics_engine=econ)

                # Run episode
                ep_metrics = run_episode(
                    coordinator=coordinator,
                    controller=ctrl,
                    scenario=scenario,
                    collector=collector,
                    episode_id=ep,
                )

                episode_metrics_list.append(ep_metrics)
                elapsed = time.time() - t0

                logger.info(
                    f"    Episode {ep+1}/{n_episodes}: "
                    f"cost=${ep_metrics.total_operating_cost_usd:.0f}, "
                    f"freq_dev={ep_metrics.freq_mean_deviation_hz*1000:.1f}mHz, "
                    f"v_viol={ep_metrics.total_voltage_violations}, "
                    f"CO₂={ep_metrics.total_carbon_tco2:.1f}t "
                    f"({elapsed:.1f}s)"
                )

                # Save raw metrics
                collector.save_episode(ep_metrics, str(metrics_dir / ctrl_name))

            all_results[ctrl_name][scenario.name] = episode_metrics_list

    return all_results


# ═══════════════════════════════════════════════════════════════════════════
# Figure generation from episode results
# ═══════════════════════════════════════════════════════════════════════════

def generate_all_figures(
    all_results: Dict[str, Dict[str, List[EpisodeMetrics]]],
    plotter: IEEEPlotter,
    output_dir: Path,
):
    """Generate all episode-based IEEE figures from evaluation results.

    Produces:
    - Frequency response plots per scenario
    - Voltage heatmaps
    - Cost comparison bar charts
    - Performance radar charts
    - Box plot comparisons
    - comparison tables (LaTeX + JSON)
    """

    controllers = list(all_results.keys())
    scenarios = set()
    for ctrl_results in all_results.values():
        scenarios.update(ctrl_results.keys())

    # ── Per-scenario time-series figures ──────────────────────────────

    for scenario_name in sorted(scenarios):
        logger.info(f"\nGenerating figures for scenario: {scenario_name}")

        # Collect time-series data across controllers
        freq_traces = {}
        voltage_traces = {}
        cost_traces = {}
        tie_traces = {}

        for ctrl_name in controllers:
            episodes = all_results[ctrl_name].get(scenario_name, [])
            if not episodes:
                continue
            # Use last episode for time-series (or average)
            ep = episodes[-1]
            if not ep.timesteps:
                continue

            hours = [t.hour for t in ep.timesteps]
            freq_traces[ctrl_name] = {
                'hours': hours,
                'frequency': [t.frequency_hz for t in ep.timesteps],
                'deviation': [t.frequency_deviation_hz for t in ep.timesteps],
            }
            voltage_traces[ctrl_name] = {
                'hours': hours,
                'v_min': [t.min_voltage_pu for t in ep.timesteps],
                'v_max': [t.max_voltage_pu for t in ep.timesteps],
                'v_avg': [t.avg_voltage_pu for t in ep.timesteps],
            }
            cost_traces[ctrl_name] = {
                'hours': hours,
                'cost': [t.operating_cost_usd_h for t in ep.timesteps],
                'smc': [t.system_marginal_cost_usd_mwh for t in ep.timesteps],
            }
            tie_traces[ctrl_name] = {
                'hours': hours,
                'tie_ab': [t.tie_ab_flow_mw for t in ep.timesteps],
                'tie_bc': [t.tie_bc_flow_mw for t in ep.timesteps],
                'tie_ac': [t.tie_ac_flow_mw for t in ep.timesteps],
            }

        # Frequency response
        if freq_traces:
            plotter.plot_frequency_response(
                freq_traces,
                title=f"Frequency Response — {scenario_name}",
                filename=f"fig06_freq_{scenario_name}",
            )

        # Voltage heatmap (using RL controller if available)
        for ctrl_name in ['rl_ppo', controllers[0]]:
            if ctrl_name in voltage_traces:
                episodes = all_results[ctrl_name].get(scenario_name, [])
                if episodes and episodes[-1].timesteps:
                    ep = episodes[-1]
                    plotter.plot_voltage_heatmap(
                        bus_voltages={
                            f"Step {t.step}": {
                                'min': t.min_voltage_pu,
                                'max': t.max_voltage_pu,
                                'avg': t.avg_voltage_pu,
                            }
                            for t in ep.timesteps[::6]  # Every 30 min
                        },
                        title=f"Voltage Profile — {scenario_name} ({ctrl_name})",
                        filename=f"fig07_voltage_{scenario_name}_{ctrl_name}",
                    )
                break

        # Tie-line flows
        if tie_traces:
            plotter.plot_tie_lines(
                tie_traces,
                title=f"Tie-Line Flows — {scenario_name}",
                filename=f"fig09_tieline_{scenario_name}",
            )

    # ── Cross-scenario comparison figures ────────────────────────────

    # Cost comparison bar chart
    logger.info("\nGenerating cross-scenario comparison figures ...")

    cost_data = {}
    for ctrl_name in controllers:
        ctrl_costs = {}
        for sc_name, episodes in all_results[ctrl_name].items():
            costs = [ep.total_operating_cost_usd for ep in episodes]
            ctrl_costs[sc_name] = {'mean': np.mean(costs), 'std': np.std(costs)}
        cost_data[ctrl_name] = ctrl_costs

    if cost_data:
        plotter.plot_cost_comparison(
            cost_data,
            title="Operating Cost Comparison",
            filename="fig12_cost_comparison",
        )

    # Performance radar chart (base_case)
    radar_data = {}
    for ctrl_name in controllers:
        base_eps = all_results[ctrl_name].get('base_case', [])
        if base_eps:
            ep = base_eps[0]
            # Normalise to [0, 1] where 1 is best
            radar_data[ctrl_name] = {
                'Freq Stability': max(0, 1 - ep.freq_mean_deviation_hz * 10),
                'Voltage Quality': max(0, 1 - ep.voltage_violation_rate_pct / 100),
                'Cost Efficiency': max(0, 1 - ep.total_operating_cost_usd / 500000),
                'RE Integration': ep.renewable_energy_fraction_pct / 100.0,
                'Reliability': max(0, 1 - ep.lolp_pct / 10),
                'Low Carbon': max(0, 1 - ep.avg_carbon_intensity_kgco2_mwh / 500),
            }

    if radar_data:
        plotter.plot_radar_chart(
            radar_data,
            title="Performance Radar — Base Case",
            filename="fig16_radar_base",
        )

    # Box plot comparison across all base_case episodes
    box_data = {}
    for ctrl_name in controllers:
        base_eps = all_results[ctrl_name].get('base_case', [])
        if base_eps:
            box_data[ctrl_name] = {
                'freq_dev_mhz': [ep.freq_mean_deviation_hz * 1000 for ep in base_eps],
                'v_violations': [ep.total_voltage_violations for ep in base_eps],
                'cost_kusd': [ep.total_operating_cost_usd / 1000 for ep in base_eps],
                'carbon_tco2': [ep.total_carbon_tco2 for ep in base_eps],
            }

    if box_data:
        plotter.plot_box_comparison(
            box_data,
            title="Performance Distribution — Base Case",
            filename="fig17_box_comparison",
        )

    # ── Comparison tables ────────────────────────────────────────────

    logger.info("\nGenerating IEEE comparison tables ...")
    for scenario_name in sorted(scenarios):
        scenario_results = {}
        for ctrl_name in controllers:
            eps = all_results[ctrl_name].get(scenario_name, [])
            if eps:
                scenario_results[ctrl_name] = eps

        if len(scenario_results) >= 2:
            table = compute_comparison_table(scenario_results)

            # Save as JSON
            table_path = output_dir / "tables" / f"table_{scenario_name}.json"
            table_path.parent.mkdir(parents=True, exist_ok=True)
            with open(table_path, 'w') as f:
                json.dump(table, f, indent=2)

            # Generate LaTeX
            latex = plotter.generate_comparison_table_latex(
                table,
                caption=f"Performance Comparison — {scenario_name.replace('_', ' ').title()}",
                label=f"tab:{scenario_name}",
            )
            latex_path = output_dir / "tables" / f"table_{scenario_name}.tex"
            with open(latex_path, 'w') as f:
                f.write(latex)

            logger.info(f"  Table: {scenario_name} ({len(scenario_results)} controllers)")


# ═══════════════════════════════════════════════════════════════════════════
# Training convergence plots
# ═══════════════════════════════════════════════════════════════════════════

def plot_training_curves(plotter: IEEEPlotter, log_dir: str = "logs/rl"):
    """Plot training convergence from saved training history."""
    history_path = Path(log_dir) / "training_history.json"
    if not history_path.exists():
        logger.warning(f"No training history at {history_path}")
        return

    with open(history_path) as f:
        history = json.load(f)

    if isinstance(history, dict):
        training_data = history
    elif isinstance(history, list):
        # Convert list of records to dict of lists
        training_data = {}
        for record in history:
            for key, val in record.items():
                training_data.setdefault(key, []).append(val)
    else:
        return

    plotter.plot_training_convergence(
        training_data,
        title="Training Convergence",
        filename="fig15_training_convergence",
    )
    logger.info("Plotted training convergence curves.")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def build_controllers(args) -> List:
    """Build list of controllers based on CLI arguments."""
    controllers = []

    # RL controller
    if args.model:
        rl = RLControllerWrapper(model_dir=args.model, device=args.device)
        controllers.append(rl)

    # Specific baseline
    if args.baseline:
        baseline_map = {
            'no_control': NoControlBaseline,
            'droop': DroopController,
            'merit_order': MeritOrderController,
            'pi_agc': PIAGCController,
        }
        if args.baseline in baseline_map:
            controllers.append(BaselineControllerWrapper(baseline_map[args.baseline]()))
        else:
            logger.error(f"Unknown baseline: {args.baseline}")
            sys.exit(1)

    # All baselines for comparison
    if args.compare_all:
        for name, cls in [
            ('no_control', NoControlBaseline),
            ('droop', DroopController),
            ('merit_order', MeritOrderController),
            ('pi_agc', PIAGCController),
        ]:
            # Avoid duplicates
            existing_names = [c.name for c in controllers]
            if name not in existing_names:
                controllers.append(BaselineControllerWrapper(cls()))

    # If no controller specified, default to RL model if available
    if not controllers:
        default_model = _project_root / "logs" / "rl" / "best"
        if default_model.exists():
            controllers.append(RLControllerWrapper(str(default_model), device=args.device))
            logger.info(f"Auto-detected model at {default_model}")
        else:
            controllers.append(BaselineControllerWrapper(DroopController()))
            logger.info("No model found — using droop baseline as default.")

    return controllers


def build_scenarios(args) -> List[ScenarioDefinition]:
    """Build list of scenarios based on CLI arguments."""
    if args.stability_only:
        return []

    if args.scenarios is None or 'all' in args.scenarios:
        return get_all_scenarios()

    scenarios = []
    for name in args.scenarios:
        s = get_scenario_by_name(name)
        if s is not None:
            scenarios.append(s)
        else:
            logger.warning(f"Unknown scenario: {name}")
    return scenarios


def main():
    parser = argparse.ArgumentParser(
        description="IEEE-Compliant Evaluation for DEMS SuperGrid",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --model logs/rl/best --scenarios all --episodes 10
  %(prog)s --baseline droop --scenarios base_case n1_generator
  %(prog)s --model logs/rl/best --compare-all --episodes 50
  %(prog)s --stability-only
  %(prog)s --model logs/rl/best --scenarios base_case --episodes 1
        """,
    )

    # Controller selection
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained RL model directory")
    parser.add_argument("--baseline", type=str, default=None,
                        choices=["no_control", "droop", "merit_order", "pi_agc"],
                        help="Run a specific baseline controller")
    parser.add_argument("--compare-all", action="store_true",
                        help="Include all baselines for comparison")

    # Scenario selection
    parser.add_argument("--scenarios", nargs="+", default=None,
                        help="Scenario names (or 'all')")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of episodes per (controller, scenario) pair")
    parser.add_argument("--episode-length", type=int, default=288,
                        help="Default episode length in steps")

    # Analysis options
    parser.add_argument("--stability-only", action="store_true",
                        help="Run only stability analysis (Bode/Nyquist/eigenvalue)")
    parser.add_argument("--skip-stability", action="store_true",
                        help="Skip stability analysis")
    parser.add_argument("--skip-figures", action="store_true",
                        help="Skip figure generation (metrics only)")
    parser.add_argument("--plot-training", action="store_true",
                        help="Also plot training convergence curves")

    # Output
    parser.add_argument("--output", type=str, default="results/ieee_eval",
                        help="Output directory for results")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["auto", "cuda", "mps", "cpu"],
                        help="PyTorch device for RL model inference")

    # Misc
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(name)-30s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    # Suppress noisy loggers
    for noisy in ("pandapower", "numba", "matplotlib", "PIL",
                  "src.simulation.der", "src.simulation.dynamics",
                  "src.simulation.power_flow", "src.simulation.supergrid"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("DEMS SuperGrid — IEEE-Compliant Evaluation Pipeline")
    logger.info("=" * 70)
    logger.info(f"Output directory:  {output_dir.resolve()}")
    logger.info(f"Device:            {args.device}")
    logger.info(f"Seed:              {args.seed}")

    # Plotter
    plotter = IEEEPlotter(output_dir=str(output_dir / "figures"))

    # ── Phase 1: Stability analysis ──────────────────────────────────

    stability_results = {}
    if not args.skip_stability:
        try:
            stability_results = run_stability_analysis(plotter, output_dir)
        except Exception as e:
            logger.error(f"Stability analysis failed: {e}", exc_info=True)
            stability_results = {'error': str(e)}

    if args.stability_only:
        # Save and exit
        with open(output_dir / "stability_results.json", 'w') as f:
            json.dump(stability_results, f, indent=2, default=str)
        logger.info(f"\nStability results saved to {output_dir / 'stability_results.json'}")
        logger.info("Done (stability-only mode).")
        return

    # ── Phase 2: Build controllers and scenarios ─────────────────────

    controllers = build_controllers(args)
    scenarios = build_scenarios(args)

    logger.info(f"\nControllers:  {[c.name for c in controllers]}")
    logger.info(f"Scenarios:    {[s.name for s in scenarios]}")
    logger.info(f"Episodes:     {args.episodes}")
    logger.info(f"Total runs:   {len(controllers) * len(scenarios) * args.episodes}")

    if not scenarios:
        logger.error("No scenarios selected. Use --scenarios <name> or --scenarios all.")
        sys.exit(1)

    # ── Phase 3: Run evaluation episodes ─────────────────────────────

    all_results = run_evaluation(
        controllers=controllers,
        scenarios=scenarios,
        n_episodes=args.episodes,
        output_dir=str(output_dir),
        episode_length=args.episode_length,
        seed=args.seed,
    )

    # ── Phase 4: Generate figures and tables ─────────────────────────

    if not args.skip_figures:
        generate_all_figures(all_results, plotter, output_dir)

    # ── Phase 5: Training convergence (optional) ─────────────────────

    if args.plot_training:
        log_dir = str(_project_root / "logs" / "rl")
        plot_training_curves(plotter, log_dir=log_dir)

    # ── Phase 6: Save consolidated results ───────────────────────────

    logger.info("\nSaving consolidated results ...")

    # Build summary
    summary = {
        'metadata': {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'controllers': [c.name for c in controllers],
            'scenarios': [s.name for s in scenarios],
            'episodes_per_pair': args.episodes,
            'seed': args.seed,
        },
        'stability': stability_results,
        'episode_summaries': {},
    }

    for ctrl_name, ctrl_results in all_results.items():
        summary['episode_summaries'][ctrl_name] = {}
        for sc_name, episodes in ctrl_results.items():
            ep_summary = []
            for ep in episodes:
                ep_summary.append({
                    'ep_id': ep.episode_id,
                    'n_steps': ep.n_steps,
                    'freq_mean_dev_hz': ep.freq_mean_deviation_hz,
                    'freq_nadir_hz': ep.freq_nadir_hz,
                    'v_violations': ep.total_voltage_violations,
                    'cost_usd': ep.total_operating_cost_usd,
                    'carbon_tco2': ep.total_carbon_tco2,
                    're_frac_pct': ep.renewable_energy_fraction_pct,
                    'lolp_pct': ep.lolp_pct,
                    'pf_conv_pct': ep.pf_convergence_rate_pct,
                })
            summary['episode_summaries'][ctrl_name][sc_name] = ep_summary

    results_path = output_dir / "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"\n{'='*70}")
    logger.info("Evaluation Complete")
    logger.info(f"{'='*70}")
    logger.info(f"Results:    {results_path}")
    logger.info(f"Figures:    {output_dir / 'figures'}")
    logger.info(f"Tables:     {output_dir / 'tables'}")
    logger.info(f"Metrics:    {output_dir / 'metrics'}")

    # Print quick summary table
    if len(all_results) > 1:
        logger.info("\n── Quick Summary (base_case, mean over episodes) ──")
        header = f"{'Controller':<20} {'Freq Dev (mHz)':>15} {'V Viol':>8} {'Cost ($k)':>10} {'CO₂ (t)':>8}"
        logger.info(header)
        logger.info("-" * len(header))
        for ctrl_name in all_results:
            base_eps = all_results[ctrl_name].get('base_case', [])
            if base_eps:
                fd = np.mean([ep.freq_mean_deviation_hz * 1000 for ep in base_eps])
                vv = np.mean([ep.total_voltage_violations for ep in base_eps])
                co = np.mean([ep.total_operating_cost_usd / 1000 for ep in base_eps])
                ca = np.mean([ep.total_carbon_tco2 for ep in base_eps])
                logger.info(f"{ctrl_name:<20} {fd:>15.2f} {vv:>8.0f} {co:>10.1f} {ca:>8.1f}")


if __name__ == "__main__":
    main()
