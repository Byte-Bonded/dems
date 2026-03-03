"""
Inverter sub-agent environment.

Controls individual generator setpoints within one area.
Action: fractional setpoint [0, 1] for each generator.
Observation: 10-D narrow inverter obs.
Reward: voltage support + power tracking.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, List, Optional, Tuple

from src.simulation.orchestrator import PhysicsEngine
from src.simulation.supergrid import AreaID

from ...observations import SubAgentObsBuilder, INVERTER_SUB_OBS_DIM
from ...rewards import InverterSubReward
from ...constraints import GridConstraintValidator


class InverterSubEnv(gym.Env):
    """Sub-agent env for generator inverter control in a single area."""

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

        self._gen_indices = physics.get_area_generators(area_id)
        self._n_gens = len(self._gen_indices)

        # Each gen gets one action dim + one obs
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0,
            shape=(INVERTER_SUB_OBS_DIM * self._n_gens,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(self._n_gens,),
            dtype=np.float32,
        )

        self.obs_builder = SubAgentObsBuilder(episode_length=episode_length)
        self.reward_fn = InverterSubReward()
        self.constraint_validator = GridConstraintValidator()
        self._step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        self.constraint_validator.reset()
        obs = self._build_obs()
        return obs, {"area": self.area_id.value, "n_gens": self._n_gens}

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float32).flatten(), 0.0, 1.0)
        self._step_count += 1

        # Apply generator setpoints
        net = self.physics.sg.net
        p_targets = []
        for i, gi in enumerate(self._gen_indices):
            frac = float(action[i]) if i < len(action) else 0.5
            p_min, p_max = self.physics.get_generator_limits(gi)
            p_mw = p_min + frac * (p_max - p_min)
            self.physics.apply_generator_setpoint(gi, p_mw)
            p_targets.append(p_mw)

        obs = self._build_obs()

        # Compute per-gen rewards and average
        total_r = 0.0
        area_state = self.physics.get_area_state(self.area_id)
        report = self.constraint_validator.validate(
            frequency_hz=self.physics.system_frequency_hz,
            bus_voltages_pu=np.ones(39),
            line_loading_pct=np.zeros(46),
        )

        for i, gi in enumerate(self._gen_indices):
            bus = int(net.gen.at[gi, "bus"]) if gi in net.gen.index else 0
            v_pu = float(net.res_bus.at[bus, "vm_pu"]) if bus in net.res_bus.index else 1.0
            p_actual = float(net.res_gen.at[gi, "p_mw"]) if gi in net.res_gen.index else 0.0
            p_target = p_targets[i] if i < len(p_targets) else 0.0
            # NaN guard for non-converged PF
            if np.isnan(v_pu):
                v_pu = 1.0
            if np.isnan(p_actual):
                p_actual = p_target
            r, _ = self.reward_fn.compute(v_pu, p_actual, p_target, report)
            total_r += r

        reward = total_r / max(self._n_gens, 1)
        terminated = self.physics.is_done
        info = {"area": self.area_id.value, "step": self._step_count}
        return obs, float(reward), terminated, False, info

    def _build_obs(self) -> np.ndarray:
        net = self.physics.sg.net
        area_state = self.physics.get_area_state(self.area_id)
        freq = self.physics.system_frequency_hz
        step = self.physics.step_count

        obs_parts = []
        for gi in self._gen_indices:
            obs_parts.append(self.obs_builder.build_inverter(
                gen_idx=gi,
                net=net,
                frequency_hz=freq,
                area_state=area_state,
                step=step,
                profiles=self.physics.profiles,
            ))

        if not obs_parts:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        return np.concatenate(obs_parts).astype(np.float32)
