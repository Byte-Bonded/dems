"""
Per-Timestep Metrics Collector for IEEE Evaluation

Collects comprehensive metrics during evaluation episodes, grouped by:
- Frequency stability (IEEE Std 1547-2018 §6.5)
- Voltage quality (IEEE Std 1547-2018 §6.4)
- Small-signal stability (IEEE Std 421.5-2016)
- Economic performance
- Reliability (IEEE Std 1366)
- Environmental impact
- DER utilisation
- Inter-area performance (NERC BAL-001)
- Control quality
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TimestepMetrics:
    """Metrics captured at every control timestep (5-min interval)."""
    step: int
    hour: float

    # Frequency (IEEE Std 1547-2018 §6.5)
    frequency_hz: float = 50.0
    frequency_deviation_hz: float = 0.0
    rocof_hz_s: float = 0.0          # Rate of Change of Frequency

    # Voltage (IEEE Std 1547-2018 §6.4)
    min_voltage_pu: float = 1.0
    max_voltage_pu: float = 1.0
    avg_voltage_pu: float = 1.0
    voltage_violations: int = 0
    voltage_deviation_mean: float = 0.0

    # Power balance
    total_generation_mw: float = 0.0
    total_load_mw: float = 0.0
    total_losses_mw: float = 0.0
    loss_percentage: float = 0.0
    power_balance_error_mw: float = 0.0

    # Economics
    operating_cost_usd_h: float = 0.0
    system_marginal_cost_usd_mwh: float = 0.0
    tou_price_usd_mwh: float = 0.0
    rtp_price_usd_mwh: float = 0.0

    # Carbon
    carbon_emissions_tco2_h: float = 0.0
    carbon_intensity_kgco2_mwh: float = 0.0

    # DER
    solar_output_mw: float = 0.0
    wind_output_mw: float = 0.0
    renewable_fraction: float = 0.0
    solar_curtailment_frac: float = 0.0
    wind_curtailment_frac: float = 0.0
    battery_soc: float = 0.5

    # Protection & reliability
    protection_trips: int = 0
    line_overloads: int = 0
    load_served_mw: float = 0.0
    load_curtailed_mw: float = 0.0

    # Tie-lines
    tie_ab_flow_mw: float = 0.0
    tie_bc_flow_mw: float = 0.0
    tie_ac_flow_mw: float = 0.0
    total_tie_flow_mw: float = 0.0

    # RL
    reward_central: float = 0.0
    reward_mg_a: float = 0.0
    reward_mg_b: float = 0.0
    reward_mg_c: float = 0.0
    action_smoothness: float = 0.0

    # Power flow
    pf_converged: bool = True
    pf_iterations: int = 0


@dataclass
class EpisodeMetrics:
    """Aggregated metrics for a complete evaluation episode."""
    scenario_name: str
    controller_name: str
    episode_id: int
    n_steps: int

    # Timestep data
    timesteps: List[TimestepMetrics] = field(default_factory=list)

    # ─── Aggregated frequency metrics ───
    freq_mean_deviation_hz: float = 0.0
    freq_max_deviation_hz: float = 0.0
    freq_nadir_hz: float = 50.0
    freq_zenith_hz: float = 50.0
    rocof_max_hz_s: float = 0.0
    freq_settling_time_s: float = 0.0     # Time to return within ±0.1 Hz
    freq_time_within_band_pct: float = 0.0  # % time in [49.5, 50.5] Hz

    # ─── Aggregated voltage metrics ───
    voltage_mean_deviation_pu: float = 0.0
    voltage_max_deviation_pu: float = 0.0
    total_voltage_violations: int = 0
    voltage_violation_rate_pct: float = 0.0

    # ─── Aggregated economics ───
    total_operating_cost_usd: float = 0.0
    avg_marginal_cost_usd_mwh: float = 0.0
    total_energy_mwh: float = 0.0

    # ─── Aggregated environmental ───
    total_carbon_tco2: float = 0.0
    avg_carbon_intensity_kgco2_mwh: float = 0.0
    renewable_energy_fraction_pct: float = 0.0
    total_renewable_curtailment_mwh: float = 0.0

    # ─── Reliability ───
    lolp_pct: float = 0.0            # Loss of Load Probability
    eens_mwh: float = 0.0            # Expected Energy Not Served
    total_protection_trips: int = 0
    saifi: float = 0.0               # System Average Interruption Frequency Index

    # ─── Control quality ───
    avg_action_smoothness: float = 0.0
    total_reward_central: float = 0.0
    total_reward_mg: float = 0.0
    pf_convergence_rate_pct: float = 0.0

    # ─── Tie-line ───
    avg_tie_flow_mw: float = 0.0
    max_tie_flow_mw: float = 0.0


class MetricsCollector:
    """
    Collects per-timestep metrics during evaluation episodes and computes
    aggregated statistics per IEEE evaluation standards.
    """

    def __init__(self, economics_engine=None):
        """
        Args:
            economics_engine: Optional EconomicsEngine for cost computation.
        """
        self.economics = economics_engine
        self._timesteps: List[TimestepMetrics] = []
        self._prev_frequency: float = 50.0
        self._prev_actions: Optional[np.ndarray] = None
        self._dt_control_s: float = 300.0  # 5-min steps

    def reset(self):
        """Reset for a new episode."""
        self._timesteps = []
        self._prev_frequency = 50.0
        self._prev_actions = None

    def collect_step(
        self,
        step: int,
        step_info: Dict,
        area_states: Dict[str, Dict],
        der_status: Dict,
        tie_line_flows: List[Dict],
        rewards: Optional[Dict[str, float]] = None,
        actions: Optional[np.ndarray] = None,
        hour: Optional[float] = None,
    ) -> TimestepMetrics:
        """Collect metrics for a single timestep.

        Args:
            step: Current step number.
            step_info: Info dict from PhysicsEngine.step().
            area_states: Per-area state dicts from PhysicsEngine.
            der_status: DER status dict.
            tie_line_flows: List of tie-line flow dicts.
            rewards: Optional dict of agent rewards.
            actions: Optional current action vector.
            hour: Optional hour of day.
        """
        freq_hz = step_info.get('frequency_hz', 50.0)
        h = hour if hour is not None else (step * self._dt_control_s / 3600.0) % 24.0

        # ROCOF
        rocof = abs(freq_hz - self._prev_frequency) / self._dt_control_s
        self._prev_frequency = freq_hz

        # Voltage aggregation across areas
        v_mins = []
        v_maxs = []
        v_avgs = []
        total_gen = step_info.get('total_gen_mw', 0)
        total_load = step_info.get('total_load_mw', 0)
        total_losses = step_info.get('losses_mw', 0)

        for area_id, state in area_states.items():
            v_mins.append(state.get('min_voltage_pu', 1.0))
            v_maxs.append(state.get('max_voltage_pu', 1.0))
            v_avgs.append(state.get('avg_voltage_pu', 1.0))

        v_min = min(v_mins) if v_mins else 1.0
        v_max = max(v_maxs) if v_maxs else 1.0
        v_avg = np.mean(v_avgs) if v_avgs else 1.0

        # DER metrics
        solar_mw = der_status.get('solar', {}).get('current_output_mw', 0)
        wind_mw = der_status.get('wind', {}).get('current_output_mw', 0)
        renewable_mw = solar_mw + wind_mw
        renewable_frac = renewable_mw / max(total_gen, 1.0) if total_gen > 0 else 0.0

        # Economics
        cost = 0.0
        smc = 0.0
        carbon = 0.0
        ci = 0.0
        tou = 0.0
        rtp = 0.0
        if self.economics is not None:
            try:
                cost = step_info.get('operating_cost_usd_h', 0.0)
                smc = step_info.get('system_marginal_cost', 0.0)
                tou = self.economics.tou_price(h)
                rtp = self.economics.rtp_price(h, total_load, total_gen)
                carbon = step_info.get('carbon_tco2_h', 0.0)
                ci = step_info.get('carbon_intensity', 0.0)
            except Exception:
                pass

        # Tie-line flows
        tie_ab = 0.0
        tie_bc = 0.0
        tie_ac = 0.0
        for tl in tie_line_flows:
            pair = tl.get('pair', '')
            flow = abs(tl.get('flow_mw', 0))
            if 'AB' in pair or 'A_B' in pair:
                tie_ab += flow
            elif 'BC' in pair or 'B_C' in pair:
                tie_bc += flow
            elif 'AC' in pair or 'A_C' in pair:
                tie_ac += flow

        # Action smoothness
        smoothness = 1.0
        if actions is not None and self._prev_actions is not None:
            if len(actions) == len(self._prev_actions):
                delta = float(np.mean(np.abs(actions - self._prev_actions)))
                smoothness = 1.0 - min(delta / 0.3, 1.0)
        if actions is not None:
            self._prev_actions = actions.copy()

        # Rewards
        r = rewards or {}

        tm = TimestepMetrics(
            step=step,
            hour=h,
            frequency_hz=freq_hz,
            frequency_deviation_hz=abs(freq_hz - 50.0),
            rocof_hz_s=rocof,
            min_voltage_pu=v_min,
            max_voltage_pu=v_max,
            avg_voltage_pu=v_avg,
            voltage_violations=step_info.get('voltage_violations', 0),
            voltage_deviation_mean=abs(v_avg - 1.0),
            total_generation_mw=total_gen,
            total_load_mw=total_load,
            total_losses_mw=total_losses,
            loss_percentage=(total_losses / max(total_gen, 1.0)) * 100.0 if total_gen > 0 else 0,
            power_balance_error_mw=abs(total_gen - total_load - total_losses),
            operating_cost_usd_h=cost,
            system_marginal_cost_usd_mwh=smc,
            tou_price_usd_mwh=tou,
            rtp_price_usd_mwh=rtp,
            carbon_emissions_tco2_h=carbon,
            carbon_intensity_kgco2_mwh=ci,
            solar_output_mw=solar_mw,
            wind_output_mw=wind_mw,
            renewable_fraction=renewable_frac,
            protection_trips=step_info.get('protection_trips', 0),
            line_overloads=step_info.get('line_overloads', 0),
            load_served_mw=total_load,
            tie_ab_flow_mw=tie_ab,
            tie_bc_flow_mw=tie_bc,
            tie_ac_flow_mw=tie_ac,
            total_tie_flow_mw=tie_ab + tie_bc + tie_ac,
            reward_central=r.get('central', 0.0),
            reward_mg_a=r.get('mg_A', 0.0),
            reward_mg_b=r.get('mg_B', 0.0),
            reward_mg_c=r.get('mg_C', 0.0),
            action_smoothness=smoothness,
            pf_converged=step_info.get('pf_converged', True),
        )

        self._timesteps.append(tm)
        return tm

    def finalize_episode(
        self,
        scenario_name: str = "base_case",
        controller_name: str = "rl",
        episode_id: int = 0,
    ) -> EpisodeMetrics:
        """Compute aggregated metrics from all collected timesteps.

        Returns:
            EpisodeMetrics with full per-timestep data and summaries.
        """
        ts = self._timesteps
        n = len(ts)
        if n == 0:
            return EpisodeMetrics(
                scenario_name=scenario_name,
                controller_name=controller_name,
                episode_id=episode_id,
                n_steps=0,
            )

        dt_h = self._dt_control_s / 3600.0  # hours per step

        # Frequency metrics
        freqs = [t.frequency_hz for t in ts]
        f_devs = [t.frequency_deviation_hz for t in ts]
        rocofs = [t.rocof_hz_s for t in ts]

        freq_mean_dev = np.mean(f_devs)
        freq_max_dev = max(f_devs)
        freq_nadir = min(freqs)
        freq_zenith = max(freqs)
        rocof_max = max(rocofs)

        # Time within frequency band [49.5, 50.5]
        in_band = sum(1 for f in freqs if 49.5 <= f <= 50.5)
        freq_in_band_pct = (in_band / n) * 100.0

        # Settling time: find last step where |f-50| > 0.1 Hz
        settling_step = 0
        for i, f in enumerate(freqs):
            if abs(f - 50.0) > 0.1:
                settling_step = i
        freq_settling = settling_step * self._dt_control_s

        # Voltage metrics
        v_devs = [t.voltage_deviation_mean for t in ts]
        v_violations = [t.voltage_violations for t in ts]
        total_v_violations = sum(v_violations)
        v_violation_rate = (sum(1 for v in v_violations if v > 0) / n) * 100.0

        # Economics
        total_cost = sum(t.operating_cost_usd_h * dt_h for t in ts)
        smcs = [t.system_marginal_cost_usd_mwh for t in ts]
        avg_smc = np.mean(smcs) if smcs else 0.0
        total_energy = sum(t.total_generation_mw * dt_h for t in ts)

        # Environmental
        total_carbon = sum(t.carbon_emissions_tco2_h * dt_h for t in ts)
        cis = [t.carbon_intensity_kgco2_mwh for t in ts]
        avg_ci = np.mean(cis) if cis else 0.0
        re_mwh = sum((t.solar_output_mw + t.wind_output_mw) * dt_h for t in ts)
        re_frac = (re_mwh / max(total_energy, 1.0)) * 100.0

        # Renewable curtailment
        curtailment_mwh = sum(
            (t.solar_curtailment_frac * t.solar_output_mw +
             t.wind_curtailment_frac * t.wind_output_mw) * dt_h
            for t in ts
        )

        # Reliability
        total_trips = sum(t.protection_trips for t in ts)
        load_curtailed_sum = sum(t.load_curtailed_mw * dt_h for t in ts)
        lolp = (sum(1 for t in ts if t.load_curtailed_mw > 0) / n) * 100.0

        # Control quality
        avg_smooth = np.mean([t.action_smoothness for t in ts])
        total_r_central = sum(t.reward_central for t in ts)
        total_r_mg = sum(t.reward_mg_a + t.reward_mg_b + t.reward_mg_c for t in ts)
        pf_conv_rate = (sum(1 for t in ts if t.pf_converged) / n) * 100.0

        # Tie-line
        tie_flows = [t.total_tie_flow_mw for t in ts]
        avg_tie = np.mean(tie_flows)
        max_tie = max(tie_flows) if tie_flows else 0.0

        return EpisodeMetrics(
            scenario_name=scenario_name,
            controller_name=controller_name,
            episode_id=episode_id,
            n_steps=n,
            timesteps=ts,
            freq_mean_deviation_hz=float(freq_mean_dev),
            freq_max_deviation_hz=float(freq_max_dev),
            freq_nadir_hz=float(freq_nadir),
            freq_zenith_hz=float(freq_zenith),
            rocof_max_hz_s=float(rocof_max),
            freq_settling_time_s=float(freq_settling),
            freq_time_within_band_pct=float(freq_in_band_pct),
            voltage_mean_deviation_pu=float(np.mean(v_devs)),
            voltage_max_deviation_pu=float(max(v_devs)),
            total_voltage_violations=total_v_violations,
            voltage_violation_rate_pct=float(v_violation_rate),
            total_operating_cost_usd=float(total_cost),
            avg_marginal_cost_usd_mwh=float(avg_smc),
            total_energy_mwh=float(total_energy),
            total_carbon_tco2=float(total_carbon),
            avg_carbon_intensity_kgco2_mwh=float(avg_ci),
            renewable_energy_fraction_pct=float(re_frac),
            total_renewable_curtailment_mwh=float(curtailment_mwh),
            lolp_pct=float(lolp),
            eens_mwh=float(load_curtailed_sum),
            total_protection_trips=total_trips,
            avg_action_smoothness=float(avg_smooth),
            total_reward_central=float(total_r_central),
            total_reward_mg=float(total_r_mg),
            pf_convergence_rate_pct=float(pf_conv_rate),
            avg_tie_flow_mw=float(avg_tie),
            max_tie_flow_mw=float(max_tie),
        )

    def save_episode(self, metrics: EpisodeMetrics, output_dir: str) -> str:
        """Save episode metrics to JSON file.

        Returns path to the saved file.
        """
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)

        filename = f"{metrics.scenario_name}_{metrics.controller_name}_ep{metrics.episode_id}.json"
        filepath = path / filename

        # Convert timesteps to minimal dicts
        ts_data = []
        for t in metrics.timesteps:
            ts_data.append({
                'step': t.step, 'hour': t.hour,
                'freq_hz': t.frequency_hz, 'freq_dev': t.frequency_deviation_hz,
                'rocof': t.rocof_hz_s,
                'v_min': t.min_voltage_pu, 'v_max': t.max_voltage_pu,
                'v_avg': t.avg_voltage_pu, 'v_violations': t.voltage_violations,
                'gen_mw': t.total_generation_mw, 'load_mw': t.total_load_mw,
                'losses_mw': t.total_losses_mw,
                'cost_usd_h': t.operating_cost_usd_h,
                'carbon_tco2_h': t.carbon_emissions_tco2_h,
                'solar_mw': t.solar_output_mw, 'wind_mw': t.wind_output_mw,
                're_frac': t.renewable_fraction,
                'trips': t.protection_trips,
                'tie_total_mw': t.total_tie_flow_mw,
                'reward_central': t.reward_central,
                'pf_ok': t.pf_converged,
            })

        data = {
            'scenario': metrics.scenario_name,
            'controller': metrics.controller_name,
            'episode_id': metrics.episode_id,
            'n_steps': metrics.n_steps,
            'summary': {
                'freq_mean_dev_hz': metrics.freq_mean_deviation_hz,
                'freq_max_dev_hz': metrics.freq_max_deviation_hz,
                'freq_nadir_hz': metrics.freq_nadir_hz,
                'freq_in_band_pct': metrics.freq_time_within_band_pct,
                'rocof_max': metrics.rocof_max_hz_s,
                'v_violations_total': metrics.total_voltage_violations,
                'v_violation_rate_pct': metrics.voltage_violation_rate_pct,
                'total_cost_usd': metrics.total_operating_cost_usd,
                'total_carbon_tco2': metrics.total_carbon_tco2,
                're_fraction_pct': metrics.renewable_energy_fraction_pct,
                'protection_trips': metrics.total_protection_trips,
                'lolp_pct': metrics.lolp_pct,
                'eens_mwh': metrics.eens_mwh,
                'pf_convergence_pct': metrics.pf_convergence_rate_pct,
            },
            'timesteps': ts_data,
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved metrics to {filepath}")
        return str(filepath)


def compute_comparison_table(
    results: Dict[str, List[EpisodeMetrics]],
) -> Dict[str, Dict[str, str]]:
    """Compute IEEE comparison table from multi-controller results.

    Args:
        results: {controller_name: [EpisodeMetrics, ...]} for each controller.

    Returns:
        Table as {metric_name: {controller_name: "mean ± std"}} for IEEE Table I.
    """
    table = {}

    metric_extractors = {
        'Freq. Deviation (mHz)': lambda m: m.freq_mean_deviation_hz * 1000,
        'Freq. Nadir (Hz)': lambda m: m.freq_nadir_hz,
        'ROCOF Max (Hz/s)': lambda m: m.rocof_max_hz_s,
        'Freq. in Band (%)': lambda m: m.freq_time_within_band_pct,
        'V Violation Rate (%)': lambda m: m.voltage_violation_rate_pct,
        'V Mean Dev. (pu)': lambda m: m.voltage_mean_deviation_pu,
        'Operating Cost ($/day)': lambda m: m.total_operating_cost_usd,
        'SMC ($/MWh)': lambda m: m.avg_marginal_cost_usd_mwh,
        'CO₂ Emissions (tCO₂/day)': lambda m: m.total_carbon_tco2,
        'Carbon Intensity (kgCO₂/MWh)': lambda m: m.avg_carbon_intensity_kgco2_mwh,
        'Renewable Fraction (%)': lambda m: m.renewable_energy_fraction_pct,
        'RE Curtailment (MWh)': lambda m: m.total_renewable_curtailment_mwh,
        'LOLP (%)': lambda m: m.lolp_pct,
        'EENS (MWh)': lambda m: m.eens_mwh,
        'Protection Trips': lambda m: m.total_protection_trips,
        'PF Convergence (%)': lambda m: m.pf_convergence_rate_pct,
        'Losses (%)': lambda m: np.mean([t.loss_percentage for t in m.timesteps]) if m.timesteps else 0,
        'Avg Tie Flow (MW)': lambda m: m.avg_tie_flow_mw,
    }

    for metric_name, extractor in metric_extractors.items():
        table[metric_name] = {}
        for ctrl_name, episodes in results.items():
            values = [extractor(ep) for ep in episodes]
            mean = np.mean(values)
            std = np.std(values)
            table[metric_name][ctrl_name] = f"{mean:.2f} ± {std:.2f}"

    return table
