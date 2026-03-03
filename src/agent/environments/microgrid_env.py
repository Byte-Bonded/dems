"""
Microgrid-level Gymnasium environment for per-area PPO agent.

Each MicrogridEnv wraps a PhysicsEngine and only sees/controls ONE area.

Action space:
  - Generator setpoints for area generators (fraction of Pmax)
  - Solar curtailment fractions
  - Wind curtailment fractions
  - EV utilisation fractions
  - DR curtailment fractions

Observation space:
  - 24-D from MicrogridObsBuilder (area-local state)

Reward:
  - MicrogridReward (voltage + economy + DER + constraints + smoothness)
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Any, Dict, List, Optional, Tuple

from src.simulation.orchestrator import PhysicsEngine, ScenarioConfig
from src.simulation.supergrid import AreaID, SuperGridConfig
from src.simulation.der import DERType

from ..observations import MicrogridObsBuilder, MG_OBS_DIM
from ..rewards import MicrogridReward
from ..constraints import GridConstraintValidator


class MicrogridEnv(gym.Env):
    """
    Gymnasium environment for a single microgrid area.

    One instance per area (A, B, C). The MultiAgentStepCoordinator
    creates 3 of these, one for each area.

    The environment does NOT step the physics — it only applies
    actions and reads state. The coordinator calls PhysicsEngine.step()
    after all agents have acted.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        physics: PhysicsEngine,
        area_id: AreaID,
        scenario: Optional[ScenarioConfig] = None,
        sub_agents_active: bool = False,
    ):
        super().__init__()
        self.physics = physics
        self.area_id = area_id
        self.cfg = scenario or ScenarioConfig()
        # When sub-agents are enabled the MG env only stores actions for
        # reward/obs; the sub-agents are the authoritative grid writers.
        self._sub_agents_active = sub_agents_active

        # Discover controllable elements for this area
        self._gen_indices = physics.get_area_generators(area_id)
        self._der_specs = physics.get_area_der_specs(area_id)

        self._solar_names = [s.name for s in self._der_specs if s.der_type == DERType.SOLAR_PV]
        self._wind_names = [s.name for s in self._der_specs if s.der_type == DERType.WIND]
        self._ev_names = [s.name for s in self._der_specs if s.der_type == DERType.EV_CHARGING]
        self._dr_names = [s.name for s in self._der_specs if s.der_type == DERType.DEMAND_RESPONSE]

        # Action dimension: gens + solar_curtail + wind_curtail + ev_util + dr_curtail
        self._n_gens = len(self._gen_indices)
        self._action_dim = (
            self._n_gens
            + len(self._solar_names)
            + len(self._wind_names)
            + len(self._ev_names)
            + len(self._dr_names)
        )

        # Gymnasium spaces
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(MG_OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self._action_dim,), dtype=np.float32
        )

        # Subsystems
        self.obs_builder = MicrogridObsBuilder(
            area_id=area_id,
            episode_length=self.cfg.episode_length_steps,
        )
        self.reward_fn = MicrogridReward()
        self.constraint_validator = GridConstraintValidator()

        # Internal state
        self._last_obs: Optional[np.ndarray] = None
        self._step_count = 0

    @property
    def action_dim(self) -> int:
        return self._action_dim

    # ─── Gymnasium API ──────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self._step_count = 0
        self.reward_fn.reset()
        self.constraint_validator.reset()

        obs = self._build_obs()
        self._last_obs = obs
        return obs, {"area": self.area_id.value}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Apply area-level actions and return (obs, reward, terminated, truncated, info).

        NOTE: The physics step is NOT called here — the coordinator does that.
        This method only applies actions and computes reward from the resulting state.
        """
        action = np.clip(np.asarray(action, dtype=np.float32).flatten(), 0.0, 1.0)
        if len(action) < self._action_dim:
            action = np.pad(action, (0, self._action_dim - len(action)), constant_values=0.5)

        self._step_count += 1

        # 1. Apply actions to physics engine
        self._apply_actions(action)

        # 2. Build observation from current state
        obs = self._build_obs()

        # 3. Compute constraint report
        pf = self.physics.last_pf_result
        bus_v = np.array([1.0] * 39)  # default
        line_loading = np.array([0.0] * 46)  # default
        if pf and pf.converged:
            net = self.physics.sg.net
            # Get area buses
            offset = {"A": 0, "B": 39, "C": 78}[self.area_id.value]
            area_buses = list(range(offset, offset + 39))
            bus_v = np.nan_to_num(np.array([
                float(net.res_bus.at[b, "vm_pu"]) if b in net.res_bus.index else 1.0
                for b in area_buses
            ]), nan=1.0)
            # Area line loading
            area_lines = [
                i for i in net.line.index
                if int(net.line.at[i, "from_bus"]) in area_buses
                or int(net.line.at[i, "to_bus"]) in area_buses
            ]
            line_loading = np.nan_to_num(np.array([
                float(net.res_line.at[i, "loading_percent"]) if i in net.res_line.index else 0.0
                for i in area_lines
            ]), nan=0.0)

        report = self.constraint_validator.validate(
            frequency_hz=self.physics.system_frequency_hz,
            bus_voltages_pu=bus_v,
            line_loading_pct=line_loading,
        )

        # 4. Compute reward
        area_state = self.physics.get_area_state(self.area_id)
        der_status = self.physics.get_der_status()
        reward, reward_breakdown = self.reward_fn.compute(
            area_state=area_state,
            constraint_report=report,
            action=action,
            der_status=der_status,
        )

        terminated = self.physics.is_done
        truncated = False
        info = {
            "area": self.area_id.value,
            "step": self._step_count,
            "reward_breakdown": reward_breakdown,
            "violations": report.total_violations,
        }

        self._last_obs = obs
        return obs, float(reward), terminated, truncated, info

    # ─── action application ─────────────────────────────────────────

    def _apply_actions(self, action: np.ndarray) -> None:
        """Map flat action to physical commands.

        FIX PF-CONV: When sub-agents are active, skip direct grid writes.
        The sub-agents (inverter, renewable, load) are the authoritative
        writers and will apply the controls themselves.
        """
        idx = 0
        net = self.physics.sg.net
        skip_grid = self._sub_agents_active

        # Generator setpoints
        for gi in self._gen_indices:
            frac = float(action[idx]) if idx < len(action) else 0.5
            idx += 1
            if not skip_grid:
                p_min, p_max = self.physics.get_generator_limits(gi)
                p_mw = p_min + frac * (p_max - p_min)
                self.physics.apply_generator_setpoint(gi, p_mw)

        # Solar curtailment
        for name in self._solar_names:
            curtail = float(action[idx]) if idx < len(action) else 0.0
            idx += 1
            if not skip_grid:
                self.physics.apply_solar_curtailment(name, curtail)

        # Wind curtailment
        for name in self._wind_names:
            curtail = float(action[idx]) if idx < len(action) else 0.0
            idx += 1
            if not skip_grid:
                self.physics.apply_wind_curtailment(name, curtail)

        # EV utilisation
        dm = self.physics.der_manager
        for name in self._ev_names:
            util = float(action[idx]) if idx < len(action) else 0.3
            idx += 1
            if not skip_grid and dm:
                dm.set_ev_charging_load(name, util)

        # DR curtailment
        for name in self._dr_names:
            curtail = float(action[idx]) if idx < len(action) else 0.0
            idx += 1
            if not skip_grid and dm:
                dm.set_demand_response_curtailment(name, curtail)

    # ─── observation building ───────────────────────────────────────

    def _build_obs(self) -> np.ndarray:
        """Construct observation for this area."""
        area_state = self.physics.get_area_state(self.area_id)
        der_status = self.physics.get_der_status()
        gen_states = self.physics.get_generator_states()

        return self.obs_builder.build(
            area_state=area_state,
            der_status=der_status,
            gen_states=gen_states,
            frequency_hz=self.physics.system_frequency_hz,
            profiles=self.physics.profiles,
            step=self.physics.step_count,
            area_generators=self._gen_indices,
            net=self.physics.sg.net,
        )
