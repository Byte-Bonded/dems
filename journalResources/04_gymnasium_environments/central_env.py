"""
Central coordination Gymnasium environment for the global PPO agent.

The CentralCoordEnv observes inter-area state (tie-lines, global frequency,
area summaries) and outputs inter-area coordination signals:
  - Area power-balance targets (MW to shift between areas)
  - Frequency bias setting

It does NOT directly set generator MW. Instead, its output is fed as
context to the 3 MicrogridEnv agents.

Action space:
  - 6-D: [area_A_bias, area_B_bias, area_C_bias,
          tie_AB_target, tie_BC_target, tie_AC_target]

Observation space:
  - 24-D from CentralObsBuilder
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Any, Dict, List, Optional, Tuple

from src.simulation.orchestrator import PhysicsEngine, ScenarioConfig
from src.simulation.supergrid import AreaID

from ..observations import CentralObsBuilder, CENTRAL_OBS_DIM
from ..rewards import CentralReward
from ..constraints import GridConstraintValidator


CENTRAL_ACTION_DIM = 6


class CentralCoordEnv(gym.Env):
    """
    Gymnasium environment for the central coordination agent.

    Observes global/inter-area state, outputs coordination signals.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        physics: PhysicsEngine,
        scenario: Optional[ScenarioConfig] = None,
    ):
        super().__init__()
        self.physics = physics
        self.cfg = scenario or ScenarioConfig()

        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(CENTRAL_OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(CENTRAL_ACTION_DIM,), dtype=np.float32
        )

        self.obs_builder = CentralObsBuilder(
            episode_length=self.cfg.episode_length_steps,
        )
        self.reward_fn = CentralReward()
        self.constraint_validator = GridConstraintValidator()

        self._step_count = 0
        self._last_coordination: Optional[Dict] = None

    @property
    def action_dim(self) -> int:
        return CENTRAL_ACTION_DIM

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
        self._last_coordination = None

        obs = self._build_obs()
        return obs, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Process central coordination action.

        The action encodes:
          [0..2]: area bias signals (how much each area should adjust gen)
          [3..5]: tie-line flow targets (normalised)

        These signals are NOT applied directly to generators — they are
        stored as coordination context for the MicrogridEnvs to read.
        """
        action = np.clip(np.asarray(action, dtype=np.float32).flatten(), -1.0, 1.0)
        if len(action) < CENTRAL_ACTION_DIM:
            action = np.pad(action, (0, CENTRAL_ACTION_DIM - len(action)))

        self._step_count += 1

        # Parse coordination signals
        self._last_coordination = {
            "area_bias": {
                "A": float(action[0]),
                "B": float(action[1]),
                "C": float(action[2]),
            },
            "tie_targets": {
                "AB": float(action[3]) * 200.0,  # ±200 MW
                "BC": float(action[4]) * 200.0,
                "AC": float(action[5]) * 200.0,
            },
        }

        # Build observation
        obs = self._build_obs()

        # Compute constraint report on global state
        pf = self.physics.last_pf_result
        if pf and pf.converged:
            net = self.physics.sg.net
            bus_v = np.nan_to_num(net.res_bus["vm_pu"].values, nan=1.0) if len(net.res_bus) > 0 else np.ones(117)
            line_loading = (
                np.nan_to_num(net.res_line["loading_percent"].values, nan=0.0)
                if len(net.res_line) > 0
                else np.zeros(130)
            )
        else:
            bus_v = np.ones(117)
            line_loading = np.zeros(130)

        report = self.constraint_validator.validate(
            frequency_hz=self.physics.system_frequency_hz,
            bus_voltages_pu=bus_v,
            line_loading_pct=line_loading,
        )

        # Count protection trips from last step info
        # (the physics engine tracks this via dynamics)
        trips = 0

        # Compute reward
        tie_flows = self.physics.get_tie_line_flows()
        reward, reward_breakdown = self.reward_fn.compute(
            frequency_hz=self.physics.system_frequency_hz,
            tie_line_flows=tie_flows,
            protection_trips=trips,
            constraint_report=report,
            action=action,
            pf_converged=(pf.converged if pf else False),
        )

        terminated = self.physics.is_done
        truncated = False
        info = {
            "step": self._step_count,
            "coordination": self._last_coordination,
            "reward_breakdown": reward_breakdown,
            "violations": report.total_violations,
        }

        return obs, float(reward), terminated, truncated, info

    def get_coordination_signals(self) -> Optional[Dict]:
        """Return the latest coordination signals for MG agents."""
        return self._last_coordination

    # ─── observation ───────────────────────────────────────────────

    def _build_obs(self) -> np.ndarray:
        global_state = self.physics.get_global_state()
        tie_flows = self.physics.get_tie_line_flows()
        return self.obs_builder.build(
            global_state=global_state,
            tie_line_flows=tie_flows,
            frequency_hz=self.physics.system_frequency_hz,
            profiles=self.physics.profiles,
            step=self.physics.step_count,
        )
