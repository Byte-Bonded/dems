"""
Physics Engine — Simulation backbone for the DEMS multi-agent system

This module provides the physical simulation layer that all RL agents interact
with through the MultiAgentStepCoordinator. It owns the SuperGrid, power flow,
dynamics, DER management, and stochastic profile generation.

The old single-agent RL interface (ObservationBuilder, RewardCalculator,
ActionMapper, GridOrchestrator) has been replaced by the hierarchical
multi-agent system in src/agent/.

IEEE Standard Compliance
========================
* **IEEE Std 39-bus** (New England Test System) × 3 areas = 117-bus SuperGrid
* **IEEE Std 421.5** — Type 1 Excitation System (IEEET1)
* **IEEE/NERC TGOV1** — Governor-turbine primary frequency response
* **IEEE Std 421.5 PSS2A** — Power System Stabilizer
* **IEGC (Indian Grid Code)** — 50 Hz, 49.5–50.5 Hz, 0.95–1.05 pu voltage
* **IEC 61400-27** — Wind turbine power-curve model
"""

from __future__ import annotations

import numpy as np
import pandapower as pp
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

from .supergrid import SuperGrid, SuperGridConfig, AreaID
from .power_flow import PowerFlowRunner, PowerFlowResult, PowerFlowAlgorithm
from .der import DERManager, DERType
from .dynamics import (
    DynamicsCoordinator, IEEE39_GENERATOR_DATA,
    DynamicLoadModel, ProtectionRelay,
)

logger = logging.getLogger(__name__)


# ======================== CONFIGURATION ======================== #

@dataclass
class ScenarioConfig:
    """
    Scenario-level parameters driving the stochastic profiles and episode limits.

    These are NOT hard-coded constants — they define the bounds of the stochastic
    environment that the dynamic simulation operates in.
    """
    # Episode
    episode_length_steps: int = 288          # 24 h at 5-min resolution
    dt_dynamics_s: float = 0.02              # Electromechanical integration step (20 ms)
    dynamics_substeps: int = 250             # FIX BUG-01: 250×0.02=5.0s dynamics per control step
    dt_control_s: float = 300.0              # Control step = 5 min

    # Stochastic profiles (bounds only — actual values vary each episode)
    solar_irradiance_peak_w_m2: float = 1000.0
    wind_speed_mean_m_s: float = 8.0
    wind_speed_std_m_s: float = 3.0
    load_variation_pct: float = 15.0         # ± load variation from base

    # RL reward weights
    w_frequency: float = 0.30
    w_voltage: float = 0.25
    w_economics: float = 0.20
    w_losses: float = 0.10
    w_action_smoothness: float = 0.10
    w_protection: float = 0.05

    # Frequency / voltage operating targets (IEEE / IEGC)
    f_nominal_hz: float = 50.0
    f_min_hz: float = 49.5
    f_max_hz: float = 50.5
    v_min_pu: float = 0.95
    v_max_pu: float = 1.05


# ======================== LOAD / RENEWABLE PROFILES ======================== #

class StochasticProfileGenerator:
    """
    Generates realistic, **non-hardcoded** time-varying profiles for load,
    solar irradiance, and wind speed.  Each ``reset()`` call produces a new
    random realisation so no two episodes are identical.

    Solar model
    -----------
    Gaussian bell centred on solar noon with stochastic cloud cover
    perturbation (correlated noise via Ornstein-Uhlenbeck process).

    Wind model
    ----------
    Mean-reverting Ornstein-Uhlenbeck process with configurable mean and
    volatility.  Cut-in / cut-out handled downstream by DERManager.

    Load model
    ----------
    Typical double-hump (morning/evening) diurnal shape with per-step
    Gaussian noise.
    """

    def __init__(self, cfg: ScenarioConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.T = cfg.episode_length_steps
        self._solar: Optional[np.ndarray] = None
        self._wind: Optional[np.ndarray] = None
        self._load_scale: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Generate fresh stochastic profiles for a new episode."""
        self._solar = self._generate_solar()
        self._wind = self._generate_wind()
        self._load_scale = self._generate_load()

    # ---- public accessors ----

    def solar_irradiance(self, step: int) -> float:
        """W/m² irradiance at the given control step."""
        return float(np.clip(self._solar[step % self.T], 0.0, 1400.0))

    def wind_speed(self, step: int) -> float:
        """m/s wind speed at the given control step."""
        return float(np.clip(self._wind[step % self.T], 0.0, 35.0))

    def load_scale_factor(self, step: int) -> float:
        """Multiplicative load scale factor (≈ 0.7 – 1.3)."""
        return float(np.clip(self._load_scale[step % self.T], 0.5, 1.5))

    def hour_of_day(self, step: int) -> float:
        """Fractional hour of day for current step."""
        return (step * self.cfg.dt_control_s / 3600.0) % 24.0

    # ---- internal generators ----

    def _generate_solar(self) -> np.ndarray:
        """Gaussian bell + correlated cloud noise (Ornstein-Uhlenbeck)."""
        hours = np.arange(self.T) * self.cfg.dt_control_s / 3600.0
        # Solar bell: peaks around 12:00 with σ≈3 h
        bell = np.exp(-0.5 * ((hours % 24 - 12.0) / 3.0) ** 2)
        # Night mask: zero before 6 and after 18
        night = (hours % 24 < 6) | (hours % 24 > 18)
        bell[night] = 0.0

        # Cloud cover perturbation (OU process, mean-reverting)
        cloud = np.zeros(self.T)
        theta, sigma = 0.3, 0.15
        for i in range(1, self.T):
            cloud[i] = cloud[i-1] + theta * (0 - cloud[i-1]) * 0.05 + sigma * self.rng.standard_normal()
        cloud = np.clip(cloud, -0.4, 0.4)

        irradiance = self.cfg.solar_irradiance_peak_w_m2 * (bell + cloud * bell)
        return np.clip(irradiance, 0, 1400.0)

    def _generate_wind(self) -> np.ndarray:
        """Ornstein-Uhlenbeck mean-reverting wind speed."""
        ws = np.zeros(self.T)
        ws[0] = self.cfg.wind_speed_mean_m_s + self.rng.standard_normal() * 2
        theta = 0.15
        for i in range(1, self.T):
            ws[i] = (ws[i-1]
                      + theta * (self.cfg.wind_speed_mean_m_s - ws[i-1])
                      + self.cfg.wind_speed_std_m_s * self.rng.standard_normal())
        return np.clip(ws, 0, 35.0)

    def _generate_load(self) -> np.ndarray:
        """Double-hump diurnal load curve + noise."""
        hours = np.arange(self.T) * self.cfg.dt_control_s / 3600.0
        h = hours % 24
        # Two peaks: morning (9 h) and evening (19 h)
        morning = np.exp(-0.5 * ((h - 9.0) / 2.0) ** 2)
        evening = np.exp(-0.5 * ((h - 19.0) / 2.0) ** 2)
        base = 0.7 + 0.3 * (morning + evening) / 1.0   # ≈ 0.7 — 1.3
        noise = self.rng.normal(0, self.cfg.load_variation_pct / 100 / 3, self.T)
        return np.clip(base + noise, 0.5, 1.5)


# ======================== PHYSICS ENGINE ======================== #

class PhysicsEngine:
    """
    The simulation backbone for the hierarchical multi-agent system.

    Owns every physical layer:
    * ``SuperGrid``         — topology, power flow, generators
    * ``DERManager``        — solar, wind, EV, DR
    * ``DynamicsCoordinator`` — swing eqn, AVR, governor, PSS, AGC
    * ``StochasticProfileGenerator`` — load, solar, wind
    * ``PowerFlowRunner``   — Newton-Raphson AC power flow

    Multi-agent environments call PhysicsEngine methods to:
    - Update stochastic profiles (renewables, load)
    - Apply control actions (gen setpoints, DER dispatch)
    - Solve power flow
    - Step dynamics
    - Read area/global states

    The PhysicsEngine does NOT compute observations or rewards —
    that is handled by the agent layer.
    """

    def __init__(
        self,
        scenario: Optional[ScenarioConfig] = None,
        grid_config: Optional[SuperGridConfig] = None,
        seed: int = 42,
    ):
        self.cfg = scenario or ScenarioConfig()
        self.grid_cfg = grid_config or SuperGridConfig()
        self.rng = np.random.default_rng(seed)

        # Build grid
        logger.info("PhysicsEngine: building 117-bus SuperGrid …")
        self.sg = SuperGrid(config=self.grid_cfg)

        # Initialize subsystems
        self.der_manager: DERManager = self.sg.initialize_der()
        self.dynamics: DynamicsCoordinator = self.sg.initialize_dynamics()

        # Power flow engine
        self.pf_runner = PowerFlowRunner()

        # Profiles
        self.profiles = StochasticProfileGenerator(self.cfg, self.rng)

        # Episode bookkeeping
        self._step_count: int = 0
        self._episode_count: int = 0
        self._last_pf: Optional[PowerFlowResult] = None
        self._done: bool = False
        # FIX C03: Store original load values to avoid float drift
        self._base_load_p: Optional[Dict[int, float]] = None
        self._base_load_q: Optional[Dict[int, float]] = None

        # Run initial power flow so grid state is populated
        self._solve_power_flow()

        logger.info(
            f"PhysicsEngine ready: "
            f"{self.cfg.episode_length_steps} steps/episode"
        )

    # ─── public properties ─────────────────────────────────────────

    @property
    def system_frequency_hz(self) -> float:
        return self.sg.system_frequency_hz

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def is_done(self) -> bool:
        return self._done

    @property
    def last_pf_result(self) -> Optional[PowerFlowResult]:
        return self._last_pf

    # ─── reset ─────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Reset the simulation to a new episode.

        Returns:
            info dict with initial grid state.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.profiles = StochasticProfileGenerator(self.cfg, self.rng)

        # Reset grid
        self.sg.reset_to_base_case()
        self.der_manager = self.sg.initialize_der()
        self.dynamics = self.sg.initialize_dynamics()

        # Fresh profiles
        self.profiles.reset()

        # Apply initial conditions from profiles (step 0)
        self._apply_profile_conditions(step=0)

        # FIX LIGHT-LOAD: Prepare reactive resources for initial load level
        initial_scale = self.profiles.load_scale_factor(0)
        self.sg.prepare_for_load_level(initial_scale)

        # Solve initial power flow
        self._solve_power_flow()

        # Bookkeeping
        self._step_count = 0
        self._episode_count += 1
        self._done = False

        # FIX C03: Snapshot base loads after reset
        self._base_load_p = {int(i): float(self.sg.net.load.at[i, 'p_mw']) for i in self.sg.net.load.index}
        self._base_load_q = {int(i): float(self.sg.net.load.at[i, 'q_mvar']) for i in self.sg.net.load.index}

        return {
            "step": 0,
            "pf_converged": self._last_pf.converged if self._last_pf else False,
            "frequency_hz": self.system_frequency_hz,
        }

    # ─── step (called by MultiAgentStepCoordinator) ────────────────

    def step(self) -> Dict[str, Any]:
        """
        Execute one control step of the physics simulation.

        This method assumes that all agent actions have already been
        applied to the grid (gen setpoints, DER dispatch, load adjustments)
        by the MultiAgentStepCoordinator before calling this.

        Pipeline per step:
        1. Apply stochastic load profile
        2. Solve AC power flow
        3. Step electromechanical dynamics
        4. Count protection trips
        5. Return step info

        Returns:
            info dict with step diagnostics.
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() first.")

        self._step_count += 1
        step = self._step_count

        # 1. Apply load profile from BASE values (FIX C03/N04: no float drift)
        scale = self.profiles.load_scale_factor(step)
        der_load_indices = self._get_der_load_indices()
        if self._base_load_p is not None:
            for load_idx in self.sg.net.load.index:
                if int(load_idx) in der_load_indices:
                    continue
                i = int(load_idx)
                base_p = self._base_load_p.get(i, self.sg.net.load.at[load_idx, 'p_mw'])
                base_q = self._base_load_q.get(i, self.sg.net.load.at[load_idx, 'q_mvar'])
                self.sg.net.load.at[load_idx, 'p_mw'] = base_p * scale
                self.sg.net.load.at[load_idx, 'q_mvar'] = base_q * scale
        else:
            for area_id in AreaID:
                self.sg.scale_loads(area_id, scale)

        # 2. Adjust reactive resources for current load level
        self.sg.prepare_for_load_level(scale)

        # 3. Solve AC power flow
        pf_result = self._solve_power_flow()

        # 4. Step dynamics (multiple sub-steps for accuracy)
        dyn_result = None
        if pf_result.converged:
            for _ in range(self.cfg.dynamics_substeps):
                dyn_result = self.sg.step_dynamics(dt=self.cfg.dt_dynamics_s)

        # 5. Count protection trips
        trips = 0
        if dyn_result and "protection" in dyn_result:
            trips = sum(1 for s in dyn_result["protection"].values() if s.get("tripped"))

        # 6. Restore loads to BASE values
        if self._base_load_p is not None:
            for load_idx in self.sg.net.load.index:
                if int(load_idx) in der_load_indices:
                    continue
                i = int(load_idx)
                self.sg.net.load.at[load_idx, 'p_mw'] = self._base_load_p.get(i, 0)
                self.sg.net.load.at[load_idx, 'q_mvar'] = self._base_load_q.get(i, 0)
        else:
            for area_id in AreaID:
                self.sg.scale_loads(area_id, 1.0 / max(scale, 1e-6))

        # 7. Done?
        self._done = step >= self.cfg.episode_length_steps

        info: Dict[str, Any] = {
            "step": step,
            "pf_converged": pf_result.converged,
            "frequency_hz": self.system_frequency_hz,
            "total_gen_mw": pf_result.total_generation_mw,
            "total_load_mw": pf_result.total_load_mw,
            "losses_mw": pf_result.total_losses_mw,
            "voltage_violations": pf_result.num_voltage_violations,
            "line_overloads": pf_result.num_line_overloads,
            "protection_trips": trips,
            "load_scale": scale,
        }

        return info

    # ─── action application (called before step) ──────────────────

    def apply_generator_setpoint(self, gen_idx: int, p_mw: float) -> None:
        """Set a generator's active power setpoint."""
        self.sg.set_generator_setpoint(gen_idx, p_mw)

    def apply_renewable_update(self, step: int) -> None:
        """Update renewable generation from stochastic profiles."""
        self._apply_profile_conditions(step)

    def apply_solar_curtailment(self, name: str, curtail_fraction: float) -> None:
        """Apply solar curtailment (0=no curtail, 1=full curtail)."""
        dm = self.der_manager
        if dm is None:
            return
        idx = dm.der_indices.get(name)
        if idx is not None and idx in dm.net.sgen.index:
            uncurtailed = dm.net.sgen.at[idx, 'p_mw']
            dm.net.sgen.at[idx, 'p_mw'] = uncurtailed * (1.0 - np.clip(curtail_fraction, 0, 1))

    def apply_wind_curtailment(self, name: str, curtail_fraction: float) -> None:
        """Apply wind curtailment (0=no curtail, 1=full curtail)."""
        dm = self.der_manager
        if dm is None:
            return
        idx = dm.der_indices.get(name)
        if idx is not None and idx in dm.net.sgen.index:
            uncurtailed = dm.net.sgen.at[idx, 'p_mw']
            dm.net.sgen.at[idx, 'p_mw'] = uncurtailed * (1.0 - np.clip(curtail_fraction, 0, 1))

    # ─── query methods ─────────────────────────────────────────────

    def get_global_state(self) -> Dict:
        """Full grid state dict (areas, tie-lines, global metrics)."""
        return self.sg.get_global_state()

    def get_area_state(self, area: AreaID) -> Dict:
        """Per-area state dict."""
        return self.sg.get_area_state(area)

    def get_der_status(self) -> Dict:
        """Aggregated DER status by type."""
        if self.der_manager:
            return self.der_manager.get_status()
        return {}

    def get_generator_states(self) -> Dict:
        """Per-generator dynamic state."""
        if self.dynamics:
            return self.dynamics.get_generator_states()
        return {}

    def get_tie_line_flows(self) -> List[Dict]:
        """Tie-line power flows and loading."""
        return self.sg.get_tie_line_flows()

    def get_controllable_generators(self) -> Dict:
        """Generator metadata grouped by area."""
        return self.sg.get_controllable_generators()

    def get_area_generators(self, area: AreaID) -> List[int]:
        """Return generator indices for a specific area."""
        offset = {"A": 0, "B": 39, "C": 78}[area.value]
        gen_indices = []
        for idx in self.sg.net.gen.index:
            bus = int(self.sg.net.gen.at[idx, "bus"])
            if offset <= bus < offset + 39:
                gen_indices.append(int(idx))
        return gen_indices

    def get_area_der_specs(self, area: AreaID) -> List:
        """Return DER specs for a specific area."""
        if self.der_manager is None:
            return []
        return [s for s in self.der_manager.der_specs if s.area_id == area.value]

    def get_generator_limits(self, gen_idx: int) -> Tuple[float, float]:
        """Return (p_min_mw, p_max_mw) for a generator."""
        net = self.sg.net
        return (
            float(net.gen.at[gen_idx, "min_p_mw"]),
            float(net.gen.at[gen_idx, "max_p_mw"]),
        )

    # ─── internal helpers ──────────────────────────────────────────

    def _solve_power_flow(self) -> PowerFlowResult:
        """Run NR power flow and return result."""
        result = self.pf_runner.run(self.sg.net)
        self._last_pf = result
        if not result.converged:
            logger.warning(f"Step {self._step_count}: PF did not converge")
        return result

    def _get_der_load_indices(self) -> set:
        """Return pandapower load indices that are DER-managed."""
        indices = set()
        dm = self.der_manager
        if dm is None:
            return indices
        for spec in dm.der_specs:
            if spec.der_type in (DERType.EV_CHARGING, DERType.DEMAND_RESPONSE):
                idx = dm.der_indices.get(spec.name)
                if idx is not None:
                    indices.add(int(idx))
        return indices

    def _apply_profile_conditions(self, step: int) -> None:
        """Push stochastic profile values into DER manager."""
        dm = self.der_manager
        if dm is None:
            return

        irr = self.profiles.solar_irradiance(step)
        ws = self.profiles.wind_speed(step)
        hour = int(self.profiles.hour_of_day(step))

        for spec in dm.der_specs:
            if spec.der_type == DERType.SOLAR_PV:
                dm.set_solar_output(spec.name, irr)
            elif spec.der_type == DERType.WIND:
                dm.set_wind_output(spec.name, ws)

        dm.update_ev_charging_by_hour(hour)


# Backward compatibility alias
GridOrchestrator = PhysicsEngine
