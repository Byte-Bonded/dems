"""
Load sub-agent environment.

Controls EV utilisation + DR curtailment within one area.
Action: [ev_util_1, ..., dr_curtail_1, ...] fractions in [0, 1].
Observation: 10-D narrow load obs.
Reward: customer comfort + voltage support + frequency response.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, List, Optional, Tuple

from src.simulation.orchestrator import PhysicsEngine
from src.simulation.supergrid import AreaID
from src.simulation.der import DERType

from ...observations import SubAgentObsBuilder, LOAD_SUB_OBS_DIM
from ...rewards import LoadSubReward


class LoadSubEnv(gym.Env):
    """Sub-agent env for load management (EV + DR) in a single area."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        physics: PhysicsEngine,
        area_id: AreaID,
        episode_length: int = 288,
    ):
        super().__init__()
        self.physics = physics
        self.area_id = area_id

        der_specs = physics.get_area_der_specs(area_id)
        self._ev_specs = [s for s in der_specs if s.der_type == DERType.EV_CHARGING]
        self._dr_specs = [s for s in der_specs if s.der_type == DERType.DEMAND_RESPONSE]
        self._n_units = len(self._ev_specs) + len(self._dr_specs)

        self.observation_space = spaces.Box(
            low=-2.0, high=2.0,
            shape=(LOAD_SUB_OBS_DIM * max(self._n_units, 1),),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(max(self._n_units, 1),),
            dtype=np.float32,
        )

        self.obs_builder = SubAgentObsBuilder(episode_length=episode_length)
        self.reward_fn = LoadSubReward()
        self._step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        obs = self._build_obs()
        return obs, {"area": self.area_id.value, "n_units": self._n_units}

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float32).flatten(), 0.0, 1.0)
        self._step_count += 1

        dm = self.physics.der_manager
        freq = self.physics.system_frequency_hz

        total_r = 0.0
        idx = 0

        # EV utilisation
        for spec in self._ev_specs:
            util = float(action[idx]) if idx < len(action) else 0.3
            idx += 1
            if dm:
                dm.set_ev_charging_load(spec.name, util)

        # DR curtailment
        dr_curtails = []
        for spec in self._dr_specs:
            curtail = float(action[idx]) if idx < len(action) else 0.0
            idx += 1
            dr_curtails.append(curtail)
            if dm:
                dm.set_demand_response_curtailment(spec.name, curtail)

        # Average DR curtailment for reward
        avg_dr = np.mean(dr_curtails) if dr_curtails else 0.0
        avg_ev = float(np.mean(action[:len(self._ev_specs)])) if self._ev_specs else 0.5
        area_state = self.physics.get_area_state(self.area_id)
        v_avg = area_state.get("avg_voltage_pu", 1.0)

        reward, _ = self.reward_fn.compute(
            dr_curtailment_frac=avg_dr,
            ev_utilisation_frac=avg_ev,
            v_local_pu=v_avg,
            frequency_hz=freq,
        )

        obs = self._build_obs()
        terminated = self.physics.is_done
        info = {"area": self.area_id.value, "step": self._step_count}
        return obs, float(reward), terminated, False, info

    def _build_obs(self) -> np.ndarray:
        freq = self.physics.system_frequency_hz
        step = self.physics.step_count
        profiles = self.physics.profiles
        area_state = self.physics.get_area_state(self.area_id)

        obs_parts = []
        total_load = area_state.get("total_load_mw", 0)
        v_avg = area_state.get("avg_voltage_pu", 1.0)

        for _ in self._ev_specs:
            obs_parts.append(self.obs_builder.build_load(
                total_load_mw=total_load,
                ev_load_mw=0.0,
                dr_curtail_frac=0.0,
                v_local_pu=v_avg,
                frequency_hz=freq,
                load_scale=profiles.load_scale_factor(step),
                step=step,
                profiles=profiles,
            ))
        for _ in self._dr_specs:
            obs_parts.append(self.obs_builder.build_load(
                total_load_mw=total_load,
                ev_load_mw=0.0,
                dr_curtail_frac=0.0,
                v_local_pu=v_avg,
                frequency_hz=freq,
                load_scale=profiles.load_scale_factor(step),
                step=step,
                profiles=profiles,
            ))

        if not obs_parts:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        return np.concatenate(obs_parts).astype(np.float32)
