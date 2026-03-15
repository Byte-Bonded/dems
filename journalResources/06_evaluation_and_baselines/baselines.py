"""
Baseline Controllers for IEEE Comparison

Provides non-RL controllers to serve as comparison baselines for the
hierarchical RL agent. Each baseline implements the same interface:
    select_actions(engine, step) -> Dict[str, np.ndarray]

Baselines:
    1. NoControlBaseline:     Fixed generator setpoints, no DER management
    2. DroopController:       Proportional frequency droop (R=5%) on all gens
    3. MeritOrderController:  Economic dispatch + droop frequency response
    4. PIAGCController:       AGC-only using existing PI controller
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class BaselineController(ABC):
    """Abstract base class for baseline controllers."""

    def __init__(self, name: str):
        self.name = name
        self._prev_actions: Optional[Dict[str, np.ndarray]] = None

    @abstractmethod
    def select_actions(
        self,
        engine,
        step: int,
        area_states: Dict[str, Dict],
        der_status: Dict,
        tie_line_flows: List[Dict],
    ) -> Dict[str, np.ndarray]:
        """Select control actions for all agents (returns same format as RL).

        Returns:
            Dict mapping agent names to action arrays in [0, 1] or [-1, 1].
            Keys: 'central', 'mg_A', 'mg_B', 'mg_C', plus sub-agents.
        """
        ...

    def reset(self):
        """Reset controller state."""
        self._prev_actions = None

    def apply_actions(self, engine, actions: Dict[str, np.ndarray], step: int):
        """Apply selected actions to the PhysicsEngine.

        This is a helper that translates the action dict into PhysicsEngine
        method calls, mimicking what MultiAgentStepCoordinator does.
        """
        # Apply generator setpoints from microgrid actions
        for area_id in ['A', 'B', 'C']:
            mg_key = f'mg_{area_id}'
            if mg_key not in actions:
                continue
            mg_action = actions[mg_key]
            try:
                gen_indices = engine.get_area_generators(
                    _get_area_id(area_id))
                # First N actions are generator setpoints (fraction of Pmax)
                for i, gen_idx in enumerate(gen_indices):
                    if i < len(mg_action):
                        p_min, p_max = engine.get_generator_limits(gen_idx)
                        p_mw = p_min + mg_action[i] * (p_max - p_min)
                        engine.apply_generator_setpoint(gen_idx, p_mw)
            except Exception as e:
                logger.debug(f"Failed to apply mg actions for area {area_id}: {e}")


def _get_area_id(area_str: str):
    """Convert area string to AreaID enum."""
    from src.simulation.supergrid import AreaID
    return AreaID(area_str)


class NoControlBaseline(BaselineController):
    """
    No active control — generators stay at initial setpoints.

    This represents the worst-case scenario where the RL agent provides
    no value. DER output follows stochastic profiles without curtailment.
    """

    def __init__(self):
        super().__init__("no_control")

    def select_actions(
        self,
        engine,
        step: int,
        area_states: Dict[str, Dict],
        der_status: Dict,
        tie_line_flows: List[Dict],
    ) -> Dict[str, np.ndarray]:
        # No action changes — return mid-point actions (0.5 = keep current)
        actions = {
            'central': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        }
        for area_id in ['A', 'B', 'C']:
            try:
                gen_indices = engine.get_area_generators(_get_area_id(area_id))
                n_gens = len(gen_indices)
                # Fixed 50% setpoint + no curtailment + no DR
                actions[f'mg_{area_id}'] = np.full(n_gens + 4, 0.5)
            except Exception:
                actions[f'mg_{area_id}'] = np.full(10, 0.5)

        return actions


class DroopController(BaselineController):
    """
    Proportional frequency droop control on all generators.

    ΔP = -(1/R) × Δf/f0

    R = 5% droop (standard), f0 = 50 Hz.
    No economic optimisation — purely frequency-driven.
    """

    def __init__(self, R: float = 0.05, f0: float = 50.0):
        super().__init__("droop")
        self.R = R
        self.f0 = f0

    def select_actions(
        self,
        engine,
        step: int,
        area_states: Dict[str, Dict],
        der_status: Dict,
        tie_line_flows: List[Dict],
    ) -> Dict[str, np.ndarray]:
        freq = engine.system_frequency_hz
        delta_f = (freq - self.f0) / self.f0

        # Droop response: ΔP_pu = -(1/R) × Δf_pu
        droop_response = -(1.0 / self.R) * delta_f  # pu change

        actions = {
            'central': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        }

        for area_id in ['A', 'B', 'C']:
            try:
                gen_indices = engine.get_area_generators(_get_area_id(area_id))
                n_gens = len(gen_indices)
                gen_actions = np.full(n_gens, 0.5)

                # Apply droop: move each generator proportionally
                for i, gen_idx in enumerate(gen_indices):
                    try:
                        p_min, p_max = engine.get_generator_limits(gen_idx)
                        p_range = p_max - p_min
                        if p_range > 0:
                            # Current output as fraction
                            current_frac = 0.5
                            # Add droop adjustment (clipped to [0, 1])
                            new_frac = np.clip(
                                current_frac + droop_response * 0.5, 0.0, 1.0)
                            gen_actions[i] = new_frac
                    except Exception:
                        pass

                # DER actions: no curtailment (1.0 = max output), EV at 0.5
                der_actions = np.array([0.0, 0.0, 0.5, 0.5])  # solar_curt, wind_curt, ev, dr
                actions[f'mg_{area_id}'] = np.concatenate([gen_actions, der_actions])
            except Exception:
                actions[f'mg_{area_id}'] = np.full(10, 0.5)

        return actions


class MeritOrderController(BaselineController):
    """
    Merit-order economic dispatch with frequency droop overlay.

    1. Dispatch generators cheapest-first using EconomicsEngine
    2. Apply droop correction for frequency regulation
    3. No DER curtailment unless voltage/frequency issues

    This is the strong baseline — economic + frequency without RL.
    """

    def __init__(
        self,
        economics_engine=None,
        R: float = 0.05,
        f0: float = 50.0,
    ):
        super().__init__("merit_order")
        self.economics = economics_engine
        self.R = R
        self.f0 = f0

    def select_actions(
        self,
        engine,
        step: int,
        area_states: Dict[str, Dict],
        der_status: Dict,
        tie_line_flows: List[Dict],
    ) -> Dict[str, np.ndarray]:
        freq = engine.system_frequency_hz
        delta_f = (freq - self.f0) / self.f0
        droop_adj = -(1.0 / self.R) * delta_f

        # Get total demand
        total_load = sum(
            s.get('total_load_mw', 0)
            for s in area_states.values()
        )

        # Merit-order dispatch if economics engine available
        merit_dispatch = None
        if self.economics is not None:
            try:
                merit_dispatch = self.economics.merit_order_dispatch(
                    total_demand_mw=total_load * 1.03,  # 3% margin for losses
                )
            except Exception as e:
                logger.debug(f"Merit-order dispatch failed: {e}")

        actions = {
            'central': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        }

        for area_id in ['A', 'B', 'C']:
            try:
                gen_indices = engine.get_area_generators(_get_area_id(area_id))
                n_gens = len(gen_indices)
                gen_actions = np.full(n_gens, 0.5)

                for i, gen_idx in enumerate(gen_indices):
                    try:
                        p_min, p_max = engine.get_generator_limits(gen_idx)
                        p_range = p_max - p_min

                        if merit_dispatch and gen_idx in merit_dispatch:
                            # Merit-order setpoint
                            p_merit = merit_dispatch[gen_idx]
                            base_frac = (p_merit - p_min) / max(p_range, 1.0)
                        else:
                            base_frac = 0.5

                        # Add droop overlay
                        new_frac = np.clip(base_frac + droop_adj * 0.3, 0.0, 1.0)
                        gen_actions[i] = new_frac
                    except Exception:
                        pass

                # Voltage-based DER curtailment
                v_max = area_states.get(area_id, {}).get('max_voltage_pu', 1.0)
                solar_curt = max(0, (v_max - 1.05) * 10.0) if v_max > 1.05 else 0.0
                wind_curt = max(0, (v_max - 1.06) * 10.0) if v_max > 1.06 else 0.0

                der_actions = np.array([
                    np.clip(solar_curt, 0, 1),
                    np.clip(wind_curt, 0, 1),
                    0.5,  # EV charging (moderate)
                    0.0,  # DR curtailment (none unless needed)
                ])
                actions[f'mg_{area_id}'] = np.concatenate([gen_actions, der_actions])
            except Exception:
                actions[f'mg_{area_id}'] = np.full(10, 0.5)

        return actions


class PIAGCController(BaselineController):
    """
    PI AGC controller using the existing AGC implementation.

    Uses the DynamicsCoordinator's built-in AGC for secondary frequency
    control, with droop for primary response. No economic optimisation.
    """

    def __init__(self, f0: float = 50.0, R: float = 0.05):
        super().__init__("pi_agc")
        self.f0 = f0
        self.R = R

    def select_actions(
        self,
        engine,
        step: int,
        area_states: Dict[str, Dict],
        der_status: Dict,
        tie_line_flows: List[Dict],
    ) -> Dict[str, np.ndarray]:
        freq = engine.system_frequency_hz
        delta_f = (freq - self.f0) / self.f0
        droop_adj = -(1.0 / self.R) * delta_f

        actions = {
            'central': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        }

        for area_id in ['A', 'B', 'C']:
            try:
                gen_indices = engine.get_area_generators(_get_area_id(area_id))
                n_gens = len(gen_indices)
                gen_actions = np.full(n_gens, 0.5)

                for i, gen_idx in enumerate(gen_indices):
                    try:
                        p_min, p_max = engine.get_generator_limits(gen_idx)
                        # AGC is handled internally by the DynamicsCoordinator
                        # Just apply droop as the primary response overlay
                        new_frac = np.clip(0.5 + droop_adj * 0.3, 0.0, 1.0)
                        gen_actions[i] = new_frac
                    except Exception:
                        pass

                der_actions = np.array([0.0, 0.0, 0.5, 0.0])
                actions[f'mg_{area_id}'] = np.concatenate([gen_actions, der_actions])
            except Exception:
                actions[f'mg_{area_id}'] = np.full(10, 0.5)

        return actions
