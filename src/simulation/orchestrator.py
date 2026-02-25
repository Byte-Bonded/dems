"""
Grid Simulation Orchestrator — Single Entry Point for RL Agent Interaction

This module provides the sole interface between the RL agent and the IEEE 39-bus
tri-area dynamic power system simulation.  Every interaction (observation, action,
reward, reset) passes through the ``GridOrchestrator`` class so that the agent
never needs to call supergrid, power_flow, der, or dynamics modules directly.

IEEE Standard Compliance
========================
* **IEEE Std 39-bus** (New England Test System) × 3 areas = 117-bus SuperGrid
* **IEEE Std 421.5** — Type 1 Excitation System (IEEET1) on every generator
* **IEEE/NERC TGOV1** — Governor-turbine primary frequency response
* **IEEE Std 421.5 PSS2A** — Power System Stabilizer on large units (≥600 MVA)
* **IEGC (Indian Grid Code)** — 50 Hz, 49.5–50.5 Hz operating band, 0.95–1.05 pu voltage
* **IEC 61400-27** — Wind turbine power-curve model

Dynamic Simulation Pipeline (per step)
=======================================
1. Apply RL agent actions (generator setpoints, DER dispatch, load control)
2. Solve AC power flow (Newton-Raphson, pandapower)
3. Step electromechanical dynamics (swing eqn, AVR, governor, PSS)
4. Update AGC secondary frequency control
5. Update DER states (battery SOC, irradiance, wind)
6. Update frequency-/voltage-dependent loads
7. Check protection relays
8. Compute IEEE-compliant observation & reward
9. Check termination criteria
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


class ActionType(Enum):
    """Enumeration of RL action components."""
    GEN_SETPOINT = "gen_setpoint"
    BATTERY_POWER = "battery_power"
    SOLAR_CURTAIL = "solar_curtail"
    WIND_CURTAIL = "wind_curtail"
    EV_UTILIZATION = "ev_utilization"
    DEMAND_RESPONSE = "demand_response"


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


# ======================== OBSERVATION BUILDER ======================== #

class ObservationBuilder:
    """
    Constructs a flat, normalised float32 vector from the raw grid state.

    Observation vector layout (per area × 3 + global):
    ──────────────────────────────────────────────────
    Per area (9 values × 3 areas = 27):
        0  total_generation_mw / 10000       (normalised)
        1  total_load_mw / 10000
        2  net_interchange_mw / 1000
        3  avg_voltage_pu
        4  min_voltage_pu
        5  max_voltage_pu
        6  num_voltage_violations / 39
        7  total_solar_mw / 100
        8  total_wind_mw / 100

    Global (15 values):
        27 system_frequency_hz / 50
        28 frequency_deviation_hz (clipped ±1)
        29 total_generation_mw / 30000
        30 total_load_mw / 30000
        31 total_losses_mw / 1000
        32 has_voltage_violations (0/1)
        33 has_thermal_violations (0/1)
        34 avg_battery_soc
        35 total_der_generation_mw / 1000
        36 hour_of_day / 24
        37 solar_irradiance / 1000
        38 wind_speed / 25
        39 load_scale_factor
        40 step / episode_length
        41 reward_last_step

    Total = 42

    All values clipped to [−2, 2] to avoid exploding gradients.
    """

    OBS_DIM = 42

    def __init__(self, cfg: ScenarioConfig, sg: Optional['SuperGrid'] = None):
        self.cfg = cfg
        self.sg = sg

    def build(
        self,
        grid_state: Dict,
        der_status: Dict,
        frequency_hz: float,
        profiles: StochasticProfileGenerator,
        step: int,
        last_reward: float,
    ) -> np.ndarray:
        obs = np.zeros(self.OBS_DIM, dtype=np.float32)

        # Per-area features
        for i, area_key in enumerate(["A", "B", "C"]):
            a = grid_state["areas"].get(area_key, {})
            base = i * 9
            obs[base + 0] = a.get("total_generation_mw", 0) / 10000.0
            obs[base + 1] = a.get("total_load_mw", 0) / 10000.0
            obs[base + 2] = a.get("net_interchange_mw", 0) / 1000.0
            obs[base + 3] = a.get("avg_voltage_pu", 1.0)
            obs[base + 4] = a.get("min_voltage_pu", 1.0)
            obs[base + 5] = a.get("max_voltage_pu", 1.0)
            # Count voltage violations in area
            v_min = a.get("min_voltage_pu", 1.0)
            v_max = a.get("max_voltage_pu", 1.0)
            violations = int(v_min < self.cfg.v_min_pu) + int(v_max > self.cfg.v_max_pu)
            obs[base + 6] = violations / 2.0
            # FIX C04: Per-area DER output (instead of total/3)
            area_solar = 0.0
            area_wind = 0.0
            if self.sg is not None and self.sg.der_manager is not None:
                for spec in self.sg.der_manager.der_specs:
                    if spec.area_id == area_key:
                        idx_der = self.sg.der_manager.der_indices.get(spec.name)
                        if idx_der is not None:
                            if spec.der_type == DERType.SOLAR_PV and idx_der in self.sg.net.sgen.index:
                                area_solar += self.sg.net.sgen.at[idx_der, 'p_mw']
                            elif spec.der_type == DERType.WIND and idx_der in self.sg.net.sgen.index:
                                area_wind += self.sg.net.sgen.at[idx_der, 'p_mw']
            obs[base + 7] = area_solar / 100.0
            obs[base + 8] = area_wind / 100.0

        # Global features
        gm = grid_state.get("global_metrics", {})
        obs[27] = frequency_hz / 50.0
        obs[28] = np.clip(frequency_hz - 50.0, -1, 1)
        obs[29] = gm.get("total_generation_mw", 0) / 30000.0
        obs[30] = gm.get("total_load_mw", 0) / 30000.0
        obs[31] = gm.get("total_losses_mw", 0) / 1000.0
        obs[32] = float(gm.get("has_voltage_violations", False))
        obs[33] = float(gm.get("has_thermal_violations", False))
        obs[34] = der_status.get("battery", {}).get("current_soc_pct", 50) / 100.0
        obs[35] = (
            der_status.get("solar", {}).get("current_output_mw", 0) +
            der_status.get("wind", {}).get("current_output_mw", 0)
        ) / 1000.0
        obs[36] = profiles.hour_of_day(step) / 24.0
        obs[37] = profiles.solar_irradiance(step) / 1000.0
        obs[38] = profiles.wind_speed(step) / 25.0
        obs[39] = profiles.load_scale_factor(step)
        obs[40] = step / max(self.cfg.episode_length_steps, 1)
        obs[41] = np.clip(last_reward, -5, 5) / 5.0

        return np.clip(obs, -2.0, 2.0)


# ======================== REWARD CALCULATOR ======================== #

class RewardCalculator:
    """
    Computes a multi-objective scalar reward that incentivises:
    * Frequency regulation (IEGC 49.5–50.5 Hz)
    * Voltage profile (IEEE 0.95–1.05 pu)
    * Economic generation cost minimisation
    * Loss minimisation
    * Smooth control actions
    * Zero protection trips

    All sub-rewards are in [−1, 1] and weighted by ScenarioConfig weights.
    """

    def __init__(self, cfg: ScenarioConfig):
        self.cfg = cfg
        self._prev_action: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._prev_action = None

    def compute(
        self,
        pf_result: PowerFlowResult,
        frequency_hz: float,
        protection_trips: int,
        action: np.ndarray,
    ) -> Tuple[float, Dict[str, float]]:
        """Return (total_reward, breakdown_dict)."""
        c = self.cfg

        # 1. Frequency reward — penalise quadratically from nominal
        f_dev = abs(frequency_hz - c.f_nominal_hz)
        f_max_dev = (c.f_max_hz - c.f_min_hz) / 2.0   # 0.5 Hz
        r_freq = 1.0 - min((f_dev / f_max_dev) ** 2, 1.0)

        # 2. Voltage reward — fraction of buses within limits
        if pf_result.converged:
            total_buses = 117  # tri-area
            v_ok = total_buses - pf_result.num_voltage_violations
            r_volt = v_ok / total_buses
        else:
            r_volt = -1.0

        # 3. Economic reward — incentivise lower generation cost (losses proxy)
        if pf_result.converged:
            loss_pct = pf_result.total_losses_mw / max(pf_result.total_generation_mw, 1) * 100
            r_econ = 1.0 - min(loss_pct / 3.0, 1.0)   # 0% loss → 1, ≥3% → 0
        else:
            r_econ = -1.0

        # 4. Loss reward — absolute MW losses
        if pf_result.converged:
            r_loss = 1.0 - min(pf_result.total_losses_mw / 500.0, 1.0)
        else:
            r_loss = -1.0

        # 5. Action smoothness — penalise large Δaction between steps
        if self._prev_action is not None and len(action) == len(self._prev_action):
            delta = np.mean(np.abs(action - self._prev_action))
            r_smooth = 1.0 - min(delta / 0.3, 1.0)
        else:
            r_smooth = 0.5   # Neutral on first step
        self._prev_action = action.copy()

        # 6. Protection trip penalty
        r_prot = 1.0 if protection_trips == 0 else -1.0

        # Weighted sum
        total = (
            c.w_frequency * r_freq
            + c.w_voltage * r_volt
            + c.w_economics * r_econ
            + c.w_losses * r_loss
            + c.w_action_smoothness * r_smooth
            + c.w_protection * r_prot
        )

        breakdown = {
            "r_frequency": r_freq,
            "r_voltage": r_volt,
            "r_economics": r_econ,
            "r_losses": r_loss,
            "r_smoothness": r_smooth,
            "r_protection": r_prot,
            "total": total,
        }
        return float(total), breakdown


# ======================== ACTION MAPPER ======================== #

class ActionMapper:
    """
    Maps a flat action array from the RL agent to concrete grid control commands.

    Action vector layout
    ────────────────────
    The RL agent outputs a 1-D float32 array in [0, 1].  This mapper linearly
    rescales each slice to the appropriate physical range and applies it to
    the SuperGrid / DERManager.

    Indices (default 117-bus, 18 DERs):
        0..26  : Generator MW setpoints (27 generators, fraction of Pmax)
        27..29 : Battery MW command   (3 batteries, −1 → full charge, +1 → full discharge)
        30..32 : Solar curtailment    (3 solar farms, 0 → no curtail, 1 → full curtail)
        33..35 : Wind curtailment     (3 wind farms, 0 → no curtail, 1 → full curtail)
        36..40 : EV utilisation       (5 stations, 0 → off, 1 → full)
        41..46 : Demand response      (6 programs, 0 → none, 1 → max curtail)

    Total action dim = 47
    """

    def __init__(self, sg: SuperGrid):
        self.sg = sg
        self._gen_indices: List[int] = []
        self._gen_pmax: List[float] = []
        self._gen_pmin: List[float] = []
        self._battery_names: List[str] = []
        self._battery_max_mw: List[float] = []
        self._solar_names: List[str] = []
        self._wind_names: List[str] = []
        self._ev_names: List[str] = []
        self._dr_names: List[str] = []

    @property
    def action_dim(self) -> int:
        return (
            len(self._gen_indices)
            + len(self._battery_names)
            + len(self._solar_names)
            + len(self._wind_names)
            + len(self._ev_names)
            + len(self._dr_names)
        )

    def configure(self) -> None:
        """Discover controllable elements from the live grid."""
        net = self.sg.net

        # Generators (sorted by index for deterministic ordering)
        for idx in sorted(net.gen.index):
            self._gen_indices.append(int(idx))
            self._gen_pmax.append(float(net.gen.at[idx, "max_p_mw"]))
            self._gen_pmin.append(float(net.gen.at[idx, "min_p_mw"]))

        # DER elements (only if DER manager exists)
        dm = self.sg.der_manager
        if dm is None:
            return

        for spec in dm.der_specs:
            if spec.der_type == DERType.BESS:
                self._battery_names.append(spec.name)
                self._battery_max_mw.append(spec.capacity_mw)
            elif spec.der_type == DERType.SOLAR_PV:
                self._solar_names.append(spec.name)
            elif spec.der_type == DERType.WIND:
                self._wind_names.append(spec.name)
            elif spec.der_type == DERType.EV_CHARGING:
                self._ev_names.append(spec.name)
            elif spec.der_type == DERType.DEMAND_RESPONSE:
                self._dr_names.append(spec.name)

    def apply(self, action: np.ndarray, profiles: StochasticProfileGenerator, step: int) -> Dict[str, Any]:
        """
        Translate the flat RL action vector into physical grid commands.

        Returns a dict of what was applied, useful for logging / debugging.
        """
        applied: Dict[str, Any] = {}
        idx = 0
        net = self.sg.net
        dm = self.sg.der_manager

        # --- Generator setpoints ---
        n_gen = len(self._gen_indices)
        gen_actions = action[idx: idx + n_gen]
        idx += n_gen
        for i, gi in enumerate(self._gen_indices):
            frac = float(np.clip(gen_actions[i] if i < len(gen_actions) else 0.5, 0, 1))
            p_mw = self._gen_pmin[i] + frac * (self._gen_pmax[i] - self._gen_pmin[i])
            self.sg.set_generator_setpoint(gi, p_mw)
        applied["gen_setpoints"] = gen_actions.tolist() if n_gen else []

        if dm is None:
            return applied

        # --- Battery dispatch ---
        n_batt = len(self._battery_names)
        batt_actions = action[idx: idx + n_batt]
        idx += n_batt
        for i, name in enumerate(self._battery_names):
            frac = float(np.clip(batt_actions[i] if i < len(batt_actions) else 0.5, 0, 1))
            p_mw = (frac - 0.5) * 2.0 * self._battery_max_mw[i]  # −max..+max
            dm.set_battery_power(name, p_mw)
        applied["battery_commands"] = batt_actions.tolist() if n_batt else []

        # --- Solar curtailment ---
        n_solar = len(self._solar_names)
        solar_actions = action[idx: idx + n_solar]
        idx += n_solar
        irradiance = profiles.solar_irradiance(step)
        for i, name in enumerate(self._solar_names):
            curtail = float(np.clip(solar_actions[i] if i < len(solar_actions) else 0, 0, 1))
            # FIX BUG-10: Curtail inverter power output, not irradiance input
            dm.set_solar_output(name, irradiance)  # full irradiance first
            idx_sgen = dm.der_indices[name]
            uncurtailed_mw = dm.net.sgen.at[idx_sgen, 'p_mw']
            dm.net.sgen.at[idx_sgen, 'p_mw'] = uncurtailed_mw * (1.0 - curtail)
        applied["solar_curtail"] = solar_actions.tolist() if n_solar else []

        # --- Wind curtailment (FIX C02: curtail power output, not wind speed) ---
        n_wind = len(self._wind_names)
        wind_actions = action[idx: idx + n_wind]
        idx += n_wind
        ws = profiles.wind_speed(step)
        for i, name in enumerate(self._wind_names):
            curtail = float(np.clip(wind_actions[i] if i < len(wind_actions) else 0, 0, 1))
            # First set wind at full speed to compute uncurtailed power
            dm.set_wind_output(name, ws)
            # Then read the output and curtail the power directly
            spec = dm._get_spec(name)
            wind_idx = dm.der_indices[name]
            uncurtailed_mw = dm.net.sgen.at[wind_idx, 'p_mw']
            dm.net.sgen.at[wind_idx, 'p_mw'] = uncurtailed_mw * (1.0 - curtail)
        applied["wind_curtail"] = wind_actions.tolist() if n_wind else []

        # --- EV utilisation ---
        n_ev = len(self._ev_names)
        ev_actions = action[idx: idx + n_ev]
        idx += n_ev
        for i, name in enumerate(self._ev_names):
            util = float(np.clip(ev_actions[i] if i < len(ev_actions) else 0.3, 0, 1))
            dm.set_ev_charging_load(name, util)
        applied["ev_util"] = ev_actions.tolist() if n_ev else []

        # --- Demand response ---
        n_dr = len(self._dr_names)
        dr_actions = action[idx: idx + n_dr]
        idx += n_dr
        for i, name in enumerate(self._dr_names):
            curt = float(np.clip(dr_actions[i] if i < len(dr_actions) else 0, 0, 1))
            dm.set_demand_response_curtailment(name, curt)
        applied["dr_curtail"] = dr_actions.tolist() if n_dr else []

        return applied


# ======================== GRID ORCHESTRATOR ======================== #

class GridOrchestrator:
    """
    **The single entry point for the RL agent to interact with the grid.**

    Lifecycle::

        orch = GridOrchestrator()          # builds 117-bus grid + dynamics + DER
        obs   = orch.reset()               # new episode with fresh stochastic profiles
        while not done:
            obs, reward, done, info = orch.step(action)
        summary = orch.episode_summary()

    The orchestrator owns every layer of the simulation:

    * ``SuperGrid``         — topology, power flow, generators
    * ``DERManager``        — solar, wind, battery, EV, DR
    * ``DynamicsCoordinator`` — swing eqn, AVR, governor, PSS, AGC
    * ``StochasticProfileGenerator`` — load, solar, wind (Ornstein-Uhlenbeck)
    * ``PowerFlowRunner``   — Newton-Raphson AC power flow
    * ``RewardCalculator``  — multi-objective IEEE-compliant reward
    * ``ObservationBuilder`` — normalised 42-dim observation vector
    * ``ActionMapper``      — maps RL actions → physical commands

    The agent never touches any submodule directly.
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
        logger.info("GridOrchestrator: building 117-bus SuperGrid …")
        self.sg = SuperGrid(config=self.grid_cfg)

        # Initialize subsystems
        self.der_manager: DERManager = self.sg.initialize_der()
        self.dynamics: DynamicsCoordinator = self.sg.initialize_dynamics()

        # Power flow engine
        self.pf_runner = PowerFlowRunner()

        # Profiles
        self.profiles = StochasticProfileGenerator(self.cfg, self.rng)

        # Observation / reward / action
        self.obs_builder = ObservationBuilder(self.cfg, self.sg)
        self.reward_calc = RewardCalculator(self.cfg)
        self.action_mapper = ActionMapper(self.sg)
        self.action_mapper.configure()

        # Episode bookkeeping
        self._step_count: int = 0
        self._episode_count: int = 0
        self._cumulative_reward: float = 0.0
        self._last_reward: float = 0.0
        self._last_pf: Optional[PowerFlowResult] = None
        self._history: List[Dict] = []
        self._done: bool = False
        # FIX C03: Store original load values to avoid float drift
        self._base_load_p: Optional[Dict[int, float]] = None
        self._base_load_q: Optional[Dict[int, float]] = None

        # Run initial power flow so grid state is populated
        self._solve_power_flow()

        logger.info(
            f"GridOrchestrator ready: {self.observation_dim}-D obs, "
            f"{self.action_dim}-D action, "
            f"{self.cfg.episode_length_steps} steps/episode"
        )

    # ─── public properties ─────────────────────────────────────────

    @property
    def observation_dim(self) -> int:
        return ObservationBuilder.OBS_DIM

    @property
    def action_dim(self) -> int:
        return self.action_mapper.action_dim

    @property
    def system_frequency_hz(self) -> float:
        return self.sg.system_frequency_hz

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def is_done(self) -> bool:
        return self._done

    # ─── reset ─────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """
        Reset the simulation to a new episode.

        * Rebuilds the grid to IEEE 39-bus base case
        * Generates fresh stochastic profiles
        * Runs initial power flow
        * Returns the first observation

        Parameters:
            seed: optional RNG seed for reproducibility.

        Returns:
            obs: np.ndarray of shape (42,).
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.profiles = StochasticProfileGenerator(self.cfg, self.rng)

        # Reset grid
        self.sg.reset_to_base_case()
        self.der_manager = self.sg.initialize_der()
        self.dynamics = self.sg.initialize_dynamics()

        # Re-configure action mapper on fresh grid
        self.action_mapper = ActionMapper(self.sg)
        self.action_mapper.configure()

        # Fresh profiles
        self.profiles.reset()

        # Reset reward
        self.reward_calc.reset()

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
        self._cumulative_reward = 0.0
        self._last_reward = 0.0
        self._history = []
        self._done = False

        # FIX C03: Snapshot base loads after reset
        self._base_load_p = {int(i): float(self.sg.net.load.at[i, 'p_mw']) for i in self.sg.net.load.index}
        self._base_load_q = {int(i): float(self.sg.net.load.at[i, 'q_mvar']) for i in self.sg.net.load.index}

        return self._build_observation()

    # ─── step ──────────────────────────────────────────────────────

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one control step of the simulation.

        Pipeline per step:
        1.  Apply RL actions to generators, DER, loads
        2.  Apply stochastic load profile
        3.  Solve AC power flow (Newton-Raphson)
        4.  Step electromechanical dynamics (multiple sub-steps)
        5.  Update battery SOC
        6.  Compute reward
        7.  Build observation
        8.  Check termination

        Parameters:
            action: np.ndarray of shape (action_dim,) with values in [0, 1].

        Returns:
            obs:    np.ndarray (42,)
            reward: float
            done:   bool
            info:   dict with diagnostics
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() first.")

        action = np.asarray(action, dtype=np.float32).flatten()
        if len(action) < self.action_dim:
            action = np.pad(action, (0, self.action_dim - len(action)), constant_values=0.5)
        action = np.clip(action, 0.0, 1.0)

        self._step_count += 1
        step = self._step_count

        # 1. Apply actions
        applied = self.action_mapper.apply(action, self.profiles, step)

        # 2. Apply load profile from BASE values (FIX C03/N04: no float drift)
        #    FIX NEW-BUG-01: Skip DER-managed loads (EV, DR) so RL agent
        #    actions from step 1 are not overwritten by profile scaling.
        scale = self.profiles.load_scale_factor(step)
        der_load_indices = self._get_der_load_indices()
        if self._base_load_p is not None:
            for load_idx in self.sg.net.load.index:
                if int(load_idx) in der_load_indices:
                    continue  # Preserve RL-controlled DER load
                i = int(load_idx)
                base_p = self._base_load_p.get(i, self.sg.net.load.at[load_idx, 'p_mw'])
                base_q = self._base_load_q.get(i, self.sg.net.load.at[load_idx, 'q_mvar'])
                self.sg.net.load.at[load_idx, 'p_mw'] = base_p * scale
                self.sg.net.load.at[load_idx, 'q_mvar'] = base_q * scale
        else:
            for area_id in AreaID:
                self.sg.scale_loads(area_id, scale)

        # 3. FIX LIGHT-LOAD: Adjust reactive resources for current load level
        #    Enables/disables switchable shunt capacitors based on load fraction
        self.sg.prepare_for_load_level(scale)

        # 4. Solve AC power flow
        pf_result = self._solve_power_flow()

        # 5. Step dynamics (multiple sub-steps for accuracy)
        dyn_result = None
        if pf_result.converged:
            for _ in range(self.cfg.dynamics_substeps):
                dyn_result = self.sg.step_dynamics(dt=self.cfg.dt_dynamics_s)

        # 6. Update battery SOC
        if self.der_manager:
            hours = self.cfg.dt_control_s / 3600.0
            self.der_manager.update_battery_soc(timestep_hours=hours)

        # 7. Count protection trips
        trips = 0
        if dyn_result and "protection" in dyn_result:
            trips = sum(1 for s in dyn_result["protection"].values() if s.get("tripped"))

        # 8. Reward
        reward, reward_breakdown = self.reward_calc.compute(
            pf_result, self.system_frequency_hz, trips, action
        )
        self._last_reward = reward
        self._cumulative_reward += reward

        # 9. Observation BEFORE restoration (FIX BUG-02: avoid stale loads)
        obs = self._build_observation()

        # 10. Restore loads to BASE values (FIX C03: exact restore)
        #    FIX NEW-BUG-01: Only restore non-DER loads; DER loads stay.
        if self._base_load_p is not None:
            for load_idx in self.sg.net.load.index:
                if int(load_idx) in der_load_indices:
                    continue  # DER loads managed by agent, not restored
                i = int(load_idx)
                self.sg.net.load.at[load_idx, 'p_mw'] = self._base_load_p.get(i, 0)
                self.sg.net.load.at[load_idx, 'q_mvar'] = self._base_load_q.get(i, 0)
        else:
            for area_id in AreaID:
                self.sg.scale_loads(area_id, 1.0 / max(scale, 1e-6))

        # 11. Done?
        self._done = step >= self.cfg.episode_length_steps

        # Info
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
            "reward_breakdown": reward_breakdown,
            "cumulative_reward": self._cumulative_reward,
            "applied_actions": applied,
        }

        self._history.append(info)
        self._last_pf = pf_result

        return obs, reward, self._done, info

    # ─── query methods (agent can call these for debug / logging) ──

    def get_grid_state(self) -> Dict:
        """Full grid state dict (areas, tie-lines, global metrics)."""
        return self.sg.get_global_state()

    def get_der_status(self) -> Dict:
        """Aggregated DER status by type."""
        if self.der_manager:
            return self.der_manager.get_status()
        return {}

    def get_generator_states(self) -> Dict:
        """Per-generator dynamic state (frequency, angle, power)."""
        if self.dynamics:
            return self.dynamics.get_generator_states()
        return {}

    def get_tie_line_flows(self) -> List[Dict]:
        """Tie-line power flows and loading."""
        return self.sg.get_tie_line_flows()

    def get_controllable_generators(self) -> Dict:
        """Generator metadata grouped by area."""
        return self.sg.get_controllable_generators()

    def episode_summary(self) -> Dict:
        """Summary statistics for the completed episode."""
        if not self._history:
            return {}
        return {
            "episode": self._episode_count,
            "steps": len(self._history),
            "cumulative_reward": self._cumulative_reward,
            "avg_reward": self._cumulative_reward / max(len(self._history), 1),
            "pf_convergence_rate": sum(
                1 for h in self._history if h["pf_converged"]
            ) / max(len(self._history), 1),
            "avg_frequency_hz": np.mean(
                [h["frequency_hz"] for h in self._history]
            ),
            "max_freq_deviation_hz": max(
                abs(h["frequency_hz"] - 50.0) for h in self._history
            ),
            "total_voltage_violations": sum(
                h["voltage_violations"] for h in self._history
            ),
            "total_protection_trips": sum(
                h["protection_trips"] for h in self._history
            ),
        }

    # ─── internal helpers ──────────────────────────────────────────

    def _solve_power_flow(self) -> PowerFlowResult:
        """Run NR power flow and return result."""
        result = self.pf_runner.run(self.sg.net)
        self._last_pf = result
        if not result.converged:
            logger.warning(f"Step {self._step_count}: PF did not converge")
        return result

    def _get_der_load_indices(self) -> set:
        """Return the set of pandapower load indices that are DER-managed
        (EV charging stations and demand response programs).
        FIX NEW-BUG-01: These loads are RL-agent-controlled and must
        be excluded from blanket load-profile scaling."""
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

    def _build_observation(self) -> np.ndarray:
        """Construct the normalised observation vector."""
        grid_state = self.sg.get_global_state()
        der_status = self.get_der_status()
        return self.obs_builder.build(
            grid_state=grid_state,
            der_status=der_status,
            frequency_hz=self.system_frequency_hz,
            profiles=self.profiles,
            step=self._step_count,
            last_reward=self._last_reward,
        )
