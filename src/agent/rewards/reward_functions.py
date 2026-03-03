"""
Reward function implementations for each hierarchy level.

All sub-rewards are designed to be in [-1, 1] range and combined via
weighted sum so the total reward stays bounded.

Microgrid Reward:
    - Voltage profile quality
    - Economic generation cost (loss-based proxy)
    - DER utilisation
    - Constraint violation penalty

Central Reward:
    - Frequency regulation (primary objective)
    - Tie-line flow balance
    - Global stability
    - Protection trip penalty

Sub-Agent Rewards:
    - Inverter: voltage support + power tracking
    - Renewable: maximise generation - curtailment
    - Load: DR effectiveness + voltage support
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field

from ..constraints.grid_constraints import (
    ConstraintReport,
    IEGC_FREQ_NOMINAL_HZ,
    IEGC_FREQ_MIN_HZ,
    IEGC_FREQ_MAX_HZ,
    IEEE_V_MIN_PU,
    IEEE_V_MAX_PU,
)


@dataclass
class RewardWeights:
    """Configurable weights for reward components."""
    # Microgrid weights
    mg_voltage: float = 0.30
    mg_economy: float = 0.25
    mg_der_util: float = 0.15
    mg_constraint: float = 0.20
    mg_smoothness: float = 0.10

    # Central weights
    central_frequency: float = 0.35
    central_tie_line: float = 0.25
    central_stability: float = 0.20
    central_protection: float = 0.20

    # Sub-agent weights  (simple, usually just 2-3 terms)
    sub_primary: float = 0.7
    sub_secondary: float = 0.3


class MicrogridReward:
    """
    Reward function for microgrid-level PPO agent.

    Objectives:
    1. Voltage profile quality (all buses in [0.95, 1.05] pu)
    2. Economic operation (minimise losses)
    3. DER utilisation (maximise renewable fraction)
    4. Constraint violation penalty
    5. Action smoothness
    """

    def __init__(self, weights: Optional[RewardWeights] = None):
        self.w = weights or RewardWeights()
        self._prev_action: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._prev_action = None

    def compute(
        self,
        area_state: Dict,
        constraint_report: ConstraintReport,
        action: np.ndarray,
        der_status: Dict,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute microgrid reward.

        Returns (total_reward, breakdown_dict).
        """
        # 1. Voltage profile
        v_min = area_state.get("min_voltage_pu", 1.0)
        v_max = area_state.get("max_voltage_pu", 1.0)
        v_avg = area_state.get("avg_voltage_pu", 1.0)
        v_ok_min = max(0, 1.0 - abs(v_min - 1.0) / 0.05) if v_min >= IEEE_V_MIN_PU else -abs(v_min - IEEE_V_MIN_PU) / 0.1
        v_ok_max = max(0, 1.0 - abs(v_max - 1.0) / 0.05) if v_max <= IEEE_V_MAX_PU else -abs(v_max - IEEE_V_MAX_PU) / 0.1
        r_voltage = (v_ok_min + v_ok_max) / 2.0

        # 2. Economic (loss minimisation proxy)
        gen_mw = area_state.get("total_generation_mw", 0)
        load_mw = area_state.get("total_load_mw", 0)
        loss_mw = max(gen_mw - load_mw, 0) if gen_mw > 0 else 0
        loss_frac = loss_mw / max(gen_mw, 1.0)
        r_economy = 1.0 - min(loss_frac / 0.03, 1.0)  # 0% → 1.0, ≥3% → 0.0

        # 3. DER utilisation (renewable fraction of total gen)
        solar_mw = der_status.get("solar", {}).get("current_output_mw", 0)
        wind_mw = der_status.get("wind", {}).get("current_output_mw", 0)
        renewable_mw = solar_mw + wind_mw
        r_der = min(renewable_mw / max(gen_mw, 1.0), 1.0) if gen_mw > 0 else 0.0

        # 4. Constraint penalty
        n_violations = constraint_report.total_violations
        severity = constraint_report.total_severity
        r_constraint = 1.0 - min(severity, 5.0) / 5.0

        # 5. Action smoothness
        if self._prev_action is not None and len(action) == len(self._prev_action):
            delta = float(np.mean(np.abs(action - self._prev_action)))
            r_smooth = 1.0 - min(delta / 0.3, 1.0)
        else:
            r_smooth = 0.5
        self._prev_action = action.copy()

        # Weighted sum
        total = (
            self.w.mg_voltage * r_voltage
            + self.w.mg_economy * r_economy
            + self.w.mg_der_util * r_der
            + self.w.mg_constraint * r_constraint
            + self.w.mg_smoothness * r_smooth
        )

        breakdown = {
            "r_voltage": float(r_voltage),
            "r_economy": float(r_economy),
            "r_der_util": float(r_der),
            "r_constraint": float(r_constraint),
            "r_smoothness": float(r_smooth),
            "total": float(total),
        }
        # Guard against NaN propagating from non-converged PF
        if np.isnan(total):
            total = -1.0
        return float(total), breakdown


class CentralReward:
    """
    Reward function for the central coordination PPO agent.

    Objectives:
    1. Frequency regulation (IEGC 49.5–50.5 Hz)
    2. Tie-line flow balance (minimise unscheduled interchange)
    3. Global stability (no protection trips)
    4. Action smoothness
    """

    def __init__(self, weights: Optional[RewardWeights] = None):
        self.w = weights or RewardWeights()
        self._prev_action: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._prev_action = None

    def compute(
        self,
        frequency_hz: float,
        tie_line_flows: list,
        protection_trips: int,
        constraint_report: ConstraintReport,
        action: np.ndarray,
        pf_converged: bool = True,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute central coordination reward.

        Returns (total_reward, breakdown_dict).
        """
        # 1. Frequency
        f_dev = abs(frequency_hz - IEGC_FREQ_NOMINAL_HZ)
        half_band = (IEGC_FREQ_MAX_HZ - IEGC_FREQ_MIN_HZ) / 2.0
        r_frequency = 1.0 - min((f_dev / half_band) ** 2, 1.0)
        if not pf_converged:
            r_frequency = -1.0

        # 2. Tie-line balance (penalise absolute flow imbalance)
        total_abs_flow = sum(abs(tl.get("flow_mw", 0)) for tl in tie_line_flows)
        r_tie_line = 1.0 - min(total_abs_flow / 1000.0, 1.0)

        # 3. Stability (protection trips)
        r_stability = 1.0 if protection_trips == 0 else max(-1.0, 1.0 - protection_trips * 0.5)

        # 4. Action smoothness
        if self._prev_action is not None and len(action) == len(self._prev_action):
            delta = float(np.mean(np.abs(action - self._prev_action)))
            r_smooth = 1.0 - min(delta / 0.3, 1.0)
        else:
            r_smooth = 0.5
        self._prev_action = action.copy()

        total = (
            self.w.central_frequency * r_frequency
            + self.w.central_tie_line * r_tie_line
            + self.w.central_stability * r_stability
            + (1.0 - self.w.central_frequency - self.w.central_tie_line - self.w.central_stability) * r_smooth
        )

        breakdown = {
            "r_frequency": float(r_frequency),
            "r_tie_line": float(r_tie_line),
            "r_stability": float(r_stability),
            "r_smoothness": float(r_smooth),
            "total": float(total),
        }
        if np.isnan(total):
            total = -1.0
        return float(total), breakdown


class InverterSubReward:
    """Sub-agent reward for inverter (generator setpoint) control."""

    def compute(
        self,
        v_local_pu: float,
        p_actual_mw: float,
        p_target_mw: float,
        constraint_report: ConstraintReport,
    ) -> Tuple[float, Dict[str, float]]:
        # Voltage support
        v_dev = abs(v_local_pu - 1.0)
        r_voltage = 1.0 - min(v_dev / 0.05, 1.0)

        # Power tracking
        p_err = abs(p_actual_mw - p_target_mw) / max(abs(p_target_mw), 1.0)
        r_tracking = 1.0 - min(p_err, 1.0)

        total = 0.5 * r_voltage + 0.5 * r_tracking
        if np.isnan(total):
            total = -1.0
        return float(total), {"r_voltage": float(r_voltage), "r_tracking": float(r_tracking)}


class RenewableSubReward:
    """Sub-agent reward for renewable (solar/wind) curtailment control."""

    def compute(
        self,
        output_mw: float,
        rated_mw: float,
        curtailment_frac: float,
        v_local_pu: float,
    ) -> Tuple[float, Dict[str, float]]:
        # Maximise output (penalise curtailment)
        utilisation = output_mw / max(rated_mw, 1e-3)
        r_output = min(utilisation, 1.0)

        # Voltage safety (penalise if curtailment was needed for voltage)
        v_ok = 1.0 if IEEE_V_MIN_PU <= v_local_pu <= IEEE_V_MAX_PU else 0.0

        total = 0.7 * r_output + 0.3 * v_ok
        return float(total), {"r_output": float(r_output), "r_voltage_ok": float(v_ok)}


class LoadSubReward:
    """Sub-agent reward for load (EV + DR) management."""

    def compute(
        self,
        dr_curtailment_frac: float,
        ev_utilisation_frac: float,
        v_local_pu: float,
        frequency_hz: float,
    ) -> Tuple[float, Dict[str, float]]:
        # Minimise DR curtailment (customer satisfaction)
        r_comfort = 1.0 - dr_curtailment_frac

        # Voltage support
        v_ok = 1.0 if IEEE_V_MIN_PU <= v_local_pu <= IEEE_V_MAX_PU else 0.0

        # Frequency support (reduce load when frequency drops)
        f_dev = frequency_hz - 50.0
        if f_dev < -0.2:
            # Good if curtailing during under-frequency
            r_freq_support = min(dr_curtailment_frac * 2.0, 1.0)
        else:
            r_freq_support = r_comfort

        total = 0.4 * r_comfort + 0.3 * v_ok + 0.3 * r_freq_support
        return float(total), {
            "r_comfort": float(r_comfort),
            "r_voltage_ok": float(v_ok),
            "r_freq_support": float(r_freq_support),
        }
