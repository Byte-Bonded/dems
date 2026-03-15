"""
Multi-Agent Step Coordinator.

Orchestrates the hierarchical step sequence:
1. Central agent observes global state and outputs coordination signals
2. MG agents observe local state + central signals, output local actions
3. Sub-agents refine specific control tasks
4. PhysicsEngine executes one timestep
5. All agents receive obs/reward from the resulting state

This is the single entry point for training and evaluation loops.
"""

import numpy as np
import logging
from typing import Any, Dict, List, Optional, Tuple

from src.simulation.orchestrator import PhysicsEngine, ScenarioConfig
from src.simulation.supergrid import AreaID, SuperGridConfig

from .environments import (
    MicrogridEnv,
    CentralCoordEnv,
    InverterSubEnv,
    RenewableSubEnv,
    LoadSubEnv,
)

logger = logging.getLogger(__name__)


class MultiAgentStepCoordinator:
    """
    Coordinates the hierarchical multi-agent step cycle.

    Architecture:
        ┌──────────────────────┐
        │   Central Agent      │  (1× CentralCoordEnv)
        │   global obs → coord │
        └──────┬───────────────┘
               │ coordination signals
        ┌──────┴───────────────┐
        │  MG Agent A/B/C      │  (3× MicrogridEnv)
        │  local obs → actions │
        └──────┬───────────────┘
               │ detailed commands
        ┌──────┴───────────────┐
        │  Sub-agents ×9       │  (3× Inverter, Renewable, Load)
        │  narrow obs → refine │
        └──────────────────────┘
               │
        ┌──────┴───────────────┐
        │   PhysicsEngine      │  one timestep
        └──────────────────────┘

    Usage::

        coord = MultiAgentStepCoordinator(seed=42)
        all_obs = coord.reset()
        while not coord.is_done:
            actions = {name: agent.predict(obs) for name, obs in all_obs.items()}
            all_obs, all_rewards, done, info = coord.step(actions)
    """

    AREA_IDS = [AreaID.AREA_A, AreaID.AREA_B, AreaID.AREA_C]

    def __init__(
        self,
        scenario: Optional[ScenarioConfig] = None,
        grid_config: Optional[SuperGridConfig] = None,
        seed: int = 42,
        enable_sub_agents: bool = True,
    ):
        self.cfg = scenario or ScenarioConfig()
        self.grid_cfg = grid_config or SuperGridConfig()
        self.enable_sub_agents = enable_sub_agents

        # Create physics engine (shared across all envs)
        logger.info("MultiAgentStepCoordinator: building PhysicsEngine …")
        self.physics = PhysicsEngine(
            scenario=self.cfg,
            grid_config=self.grid_cfg,
            seed=seed,
        )

        # Create environments
        self.central_env = CentralCoordEnv(
            physics=self.physics,
            scenario=self.cfg,
        )

        self.mg_envs: Dict[str, MicrogridEnv] = {}
        for area_id in self.AREA_IDS:
            key = f"mg_{area_id.value}"
            self.mg_envs[key] = MicrogridEnv(
                physics=self.physics,
                area_id=area_id,
                scenario=self.cfg,
                sub_agents_active=enable_sub_agents,
            )

        # Sub-agent environments (optional)
        self.sub_envs: Dict[str, Any] = {}
        if enable_sub_agents:
            for area_id in self.AREA_IDS:
                a = area_id.value
                self.sub_envs[f"inverter_{a}"] = InverterSubEnv(
                    physics=self.physics,
                    area_id=area_id,
                    episode_length=self.cfg.episode_length_steps,
                )
                self.sub_envs[f"renewable_{a}"] = RenewableSubEnv(
                    physics=self.physics,
                    area_id=area_id,
                    episode_length=self.cfg.episode_length_steps,
                )
                self.sub_envs[f"load_{a}"] = LoadSubEnv(
                    physics=self.physics,
                    area_id=area_id,
                    episode_length=self.cfg.episode_length_steps,
                )

        self._step_count = 0
        self._episode_count = 0

        logger.info(
            f"MultiAgentStepCoordinator ready: "
            f"1 central + {len(self.mg_envs)} MG + {len(self.sub_envs)} sub-agents"
        )

    @property
    def is_done(self) -> bool:
        return self.physics.is_done

    @property
    def all_env_names(self) -> List[str]:
        """Return all environment names in hierarchy order."""
        names = ["central"]
        names.extend(sorted(self.mg_envs.keys()))
        names.extend(sorted(self.sub_envs.keys()))
        return names

    def get_env(self, name: str):
        """Get an environment by name."""
        if name == "central":
            return self.central_env
        if name in self.mg_envs:
            return self.mg_envs[name]
        if name in self.sub_envs:
            return self.sub_envs[name]
        raise KeyError(f"Unknown environment: {name}")

    # ─── reset ──────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Reset all environments and the physics engine.

        Returns:
            Dict mapping env_name → initial observation.
        """
        # Reset physics engine
        self.physics.reset(seed=seed)
        self._step_count = 0
        self._episode_count += 1

        # Reset all envs
        all_obs = {}
        obs, _ = self.central_env.reset()
        all_obs["central"] = obs

        for name, env in self.mg_envs.items():
            obs, _ = env.reset()
            all_obs[name] = obs

        for name, env in self.sub_envs.items():
            obs, _ = env.reset()
            all_obs[name] = obs

        return all_obs

    # ─── step ───────────────────────────────────────────────────────

    def step(
        self,
        actions: Dict[str, np.ndarray],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], bool, Dict]:
        """
        Execute one hierarchical step.

        Args:
            actions: Dict mapping env_name → action array.
                     Must contain at least "central" and "mg_A/B/C".

        Returns:
            all_obs: Dict[env_name, obs_array]
            all_rewards: Dict[env_name, float]
            done: bool
            info: Dict with step diagnostics
        """
        self._step_count += 1

        # 1. Apply renewable profile update FIRST so agents curtail from
        #    the correct profile baseline (not stale values from last step).
        #    FIX PF-CONV: This also snapshots _uncurtailed_sgen so
        #    curtailment functions are absolute, not compounding.
        self.physics.apply_renewable_update(self._step_count)

        # 2. Central agent acts (produces coordination signals)
        central_action = actions.get("central", np.zeros(self.central_env.action_dim))
        c_obs, c_reward, c_term, c_trunc, c_info = self.central_env.step(central_action)

        # 3. MG agents act
        #    When sub-agents are enabled, MG env skips grid writes for
        #    controls that sub-agents will handle (generators, renewables,
        #    EV, DR) — it only stores its actions for reward/obs computation.
        #    This prevents double-write conflicts.
        mg_results = {}
        for name, env in self.mg_envs.items():
            mg_action = actions.get(name, np.zeros(env.action_dim))
            mg_obs, mg_reward, mg_term, mg_trunc, mg_info = env.step(mg_action)
            mg_results[name] = (mg_obs, mg_reward, mg_term, mg_trunc, mg_info)

        # 4. Sub-agents act (authoritative writers when enabled)
        sub_results = {}
        for name, env in self.sub_envs.items():
            sub_action = actions.get(name, np.zeros(env.action_space.shape[0]))
            sub_obs, sub_reward, sub_term, sub_trunc, sub_info = env.step(sub_action)
            sub_results[name] = (sub_obs, sub_reward, sub_term, sub_trunc, sub_info)

        # 5. Step physics engine (one timestep: load profile + PF + dynamics)
        step_info = self.physics.step()

        # 6. Collect observations and rewards
        all_obs = {"central": c_obs}
        all_rewards = {"central": c_reward}

        for name, (obs, reward, _, _, info) in mg_results.items():
            all_obs[name] = obs
            all_rewards[name] = reward

        for name, (obs, reward, _, _, info) in sub_results.items():
            all_obs[name] = obs
            all_rewards[name] = reward

        done = self.physics.is_done

        combined_info = {
            "step": self._step_count,
            "physics": step_info,
            "central": c_info,
            "mg": {k: v[4] for k, v in mg_results.items()},
            "sub": {k: v[4] for k, v in sub_results.items()},
        }

        return all_obs, all_rewards, done, combined_info

    # ─── episode summary ────────────────────────────────────────────

    def episode_summary(self) -> Dict:
        """Summary statistics for the completed episode."""
        return {
            "episode": self._episode_count,
            "steps": self._step_count,
            "frequency_hz": self.physics.system_frequency_hz,
            "pf_converged": (
                self.physics.last_pf_result.converged
                if self.physics.last_pf_result else False
            ),
        }
