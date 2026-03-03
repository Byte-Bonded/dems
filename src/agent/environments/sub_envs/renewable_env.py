"""
Renewable sub-agent environment.

Controls solar + wind curtailment within one area.
Action: curtailment fractions [0, 1] for each solar/wind unit.
Observation: 10-D narrow renewable obs per unit.
Reward: maximise output while respecting voltage.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, List, Optional, Tuple

from src.simulation.orchestrator import PhysicsEngine
from src.simulation.supergrid import AreaID
from src.simulation.der import DERType

from ...observations import SubAgentObsBuilder, RENEWABLE_SUB_OBS_DIM
from ...rewards import RenewableSubReward


class RenewableSubEnv(gym.Env):
    """Sub-agent env for renewable curtailment in a single area."""

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
        self._solar_specs = [s for s in der_specs if s.der_type == DERType.SOLAR_PV]
        self._wind_specs = [s for s in der_specs if s.der_type == DERType.WIND]
        self._n_units = len(self._solar_specs) + len(self._wind_specs)

        self.observation_space = spaces.Box(
            low=-2.0, high=2.0,
            shape=(RENEWABLE_SUB_OBS_DIM * max(self._n_units, 1),),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(max(self._n_units, 1),),
            dtype=np.float32,
        )

        self.obs_builder = SubAgentObsBuilder(episode_length=episode_length)
        self.reward_fn = RenewableSubReward()
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
        net = self.physics.sg.net

        total_r = 0.0
        idx = 0

        # Solar curtailment
        for spec in self._solar_specs:
            curtail = float(action[idx]) if idx < len(action) else 0.0
            idx += 1
            self.physics.apply_solar_curtailment(spec.name, curtail)
            # Compute reward
            sgen_idx = dm.der_indices.get(spec.name) if dm else None
            p_out = float(net.sgen.at[sgen_idx, 'p_mw']) if sgen_idx is not None and sgen_idx in net.sgen.index else 0.0
            bus = spec.bus_id if hasattr(spec, 'bus_id') else 0
            v_pu = float(net.res_bus.at[bus, 'vm_pu']) if bus in net.res_bus.index else 1.0
            # NaN guard for non-converged PF
            if np.isnan(v_pu): v_pu = 1.0
            if np.isnan(p_out): p_out = 0.0
            r, _ = self.reward_fn.compute(p_out, spec.capacity_mw, curtail, v_pu)
            total_r += r

        # Wind curtailment
        for spec in self._wind_specs:
            curtail = float(action[idx]) if idx < len(action) else 0.0
            idx += 1
            self.physics.apply_wind_curtailment(spec.name, curtail)
            sgen_idx = dm.der_indices.get(spec.name) if dm else None
            p_out = float(net.sgen.at[sgen_idx, 'p_mw']) if sgen_idx is not None and sgen_idx in net.sgen.index else 0.0
            bus = spec.bus_id if hasattr(spec, 'bus_id') else 0
            v_pu = float(net.res_bus.at[bus, 'vm_pu']) if bus in net.res_bus.index else 1.0
            # NaN guard for non-converged PF
            if np.isnan(v_pu): v_pu = 1.0
            if np.isnan(p_out): p_out = 0.0
            r, _ = self.reward_fn.compute(p_out, spec.capacity_mw, curtail, v_pu)
            total_r += r

        obs = self._build_obs()
        reward = total_r / max(self._n_units, 1)
        terminated = self.physics.is_done
        info = {"area": self.area_id.value, "step": self._step_count}
        return obs, float(reward), terminated, False, info

    def _build_obs(self) -> np.ndarray:
        freq = self.physics.system_frequency_hz
        step = self.physics.step_count
        profiles = self.physics.profiles

        obs_parts = []
        for spec in self._solar_specs:
            obs_parts.append(self.obs_builder.build_renewable(
                resource_value=profiles.solar_irradiance(step),
                resource_max=1000.0,
                current_output_mw=0.0,
                rated_mw=spec.capacity_mw,
                curtail_frac=0.0,
                v_local_pu=1.0,
                frequency_hz=freq,
                step=step,
                profiles=profiles,
            ))
        for spec in self._wind_specs:
            obs_parts.append(self.obs_builder.build_renewable(
                resource_value=profiles.wind_speed(step),
                resource_max=25.0,
                current_output_mw=0.0,
                rated_mw=spec.capacity_mw,
                curtail_frac=0.0,
                v_local_pu=1.0,
                frequency_hz=freq,
                step=step,
                profiles=profiles,
            ))

        if not obs_parts:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        return np.concatenate(obs_parts).astype(np.float32)
