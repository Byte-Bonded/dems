"""
CTDE (Centralised Training, Decentralised Execution) training module.

Training hierarchy:
1. All agents share the PhysicsEngine via MultiAgentStepCoordinator
2. During training, a shared critic (optional) sees global state
3. During execution, each agent only sees its own observation

The HierarchicalTrainer manages:
- Round-robin or simultaneous training across agent levels
- Periodic evaluation
- Model checkpointing
- TensorBoard logging
"""

import logging
import time
import json
import gymnasium as gym
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from src.simulation.orchestrator import ScenarioConfig
from src.simulation.supergrid import AreaID, SuperGridConfig

from src.agent.coordinator import MultiAgentStepCoordinator
from src.agent.ppo_agents import (
    CentralPPOAgent,
    MicrogridPPOAgent,
    SubPPOAgent,
    PPOAgentWrapper,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for hierarchical training."""
    total_timesteps: int = 500_000
    eval_freq: int = 10_000
    n_eval_episodes: int = 5
    save_freq: int = 50_000
    log_dir: str = "logs/rl"
    tensorboard_log: str = "logs/tb"
    # Whether to train sub-agents
    train_sub_agents: bool = True
    # Steps per level before cycling (round-robin)
    steps_per_level: int = 4096
    seed: int = 42
    # Device: 'auto', 'cuda', 'mps', or 'cpu'
    device: str = "auto"
    # GPU VRAM hard cap in GB (0 = unlimited)
    gpu_vram_limit_gb: float = 7.0
    # GPU-optimised batch sizes (used when device != cpu)
    gpu_batch_size_central: int = 1024
    gpu_batch_size_mg: int = 512
    gpu_batch_size_sub: int = 512


class StandaloneEnvWrapper(gym.Env):
    """
    Wraps an environment for standalone training with SB3.
    
    When stepped, it queries the other agents for their actions using
    the provided coordinator state. This allows training one level
    while others act as part of the environment.
    """
    def __init__(self, agent_name: str, coordinator: MultiAgentStepCoordinator, agents: Dict[str, Any]):
        super().__init__()
        self.agent_name = agent_name
        self.coordinator = coordinator
        self.agents = agents
        
        inner_env = coordinator.get_env(agent_name)
        self.observation_space = inner_env.observation_space
        self.action_space = inner_env.action_space
        self._last_all_obs: Optional[Dict[str, np.ndarray]] = None

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        self._last_all_obs = self.coordinator.reset(seed=seed)
        return self._last_all_obs[self.agent_name], {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        actions = {}
        # Get actions for all agents
        for name, agent in self.agents.items():
            if name == self.agent_name:
                actions[name] = action
            else:
                obs = self._last_all_obs.get(name)
                if obs is not None:
                    # SB3 agents predict returns (action, states)
                    actions[name], _ = agent.model.predict(obs, deterministic=False)
                else:
                    env = self.coordinator.get_env(name)
                    actions[name] = np.zeros(env.action_space.shape)
        
        # Step coordinator
        new_obs, rewards, done, info = self.coordinator.step(actions)
        self._last_all_obs = new_obs
        
        return new_obs[self.agent_name], float(rewards[self.agent_name]), done, False, info


class HierarchicalTrainer:
    """
    Manages CTDE training for the full hierarchy.

    Training loop:
    1. Collect rollouts from all agents simultaneously
    2. Update policies (central → MG → sub)
    3. Evaluate periodically
    4. Checkpoint models

    Usage::

        trainer = HierarchicalTrainer(config=TrainingConfig())
        trainer.train()
        trainer.save_all("models/")
    """

    def __init__(
        self,
        training_config: Optional[TrainingConfig] = None,
        scenario: Optional[ScenarioConfig] = None,
        grid_config: Optional[SuperGridConfig] = None,
    ):
        self.tc = training_config or TrainingConfig()
        self.scenario = scenario or ScenarioConfig()
        self.grid_config = grid_config or SuperGridConfig()

        # Create coordinator (owns physics + all envs)
        self.coordinator = MultiAgentStepCoordinator(
            scenario=self.scenario,
            grid_config=self.grid_config,
            seed=self.tc.seed,
            enable_sub_agents=self.tc.train_sub_agents,
        )

        # Create PPO agents
        self.agents: Dict[str, PPOAgentWrapper] = {}
        self._create_agents()

        # Training history
        self.history: List[Dict] = []
        self._total_steps = 0

        logger.info(f"HierarchicalTrainer: {len(self.agents)} agents created")

    def _create_agents(self) -> None:
        """Instantiate PPO agents for all environments."""
        coord = self.coordinator
        tb_log = self.tc.tensorboard_log

        from src.agent.ppo_agents import detect_device
        device = detect_device(self.tc.device)
        use_gpu = device != "cpu"
        logger.info(f"Training device: {device} (GPU-optimised batches: {use_gpu})")

        # Enforce VRAM cap
        if device == "cuda" and self.tc.gpu_vram_limit_gb > 0:
            try:
                import torch
                total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                # Calculate fraction and clamp to [0.05, 0.95] to avoid invalid values or zero memory
                fraction = self.tc.gpu_vram_limit_gb / total_mem_gb
                safe_fraction = max(0.05, min(0.95, fraction))
                
                torch.cuda.set_per_process_memory_fraction(safe_fraction, device=0)
                logger.info(
                    f"VRAM cap: {self.tc.gpu_vram_limit_gb:.1f} GB "
                    f"({safe_fraction * 100:.1f}% of {total_mem_gb:.1f} GB total)"
                )
                if safe_fraction != fraction:
                    logger.warning(f"VRAM fraction clamped from {fraction:.2f} to {safe_fraction:.2f}")
            except Exception as e:
                logger.warning(f"Could not set VRAM cap: {e}")

        # Central agent
        central_bs = self.tc.gpu_batch_size_central if use_gpu else 64
        self.agents["central"] = CentralPPOAgent(
            env=coord.central_env,
            tensorboard_log=tb_log,
            seed=self.tc.seed,
            device=device,
            batch_size=central_bs,
        )

        # Microgrid agents
        mg_bs = self.tc.gpu_batch_size_mg if use_gpu else 64
        for name, env in coord.mg_envs.items():
            area_id = name.split("_")[1]
            self.agents[name] = MicrogridPPOAgent(
                env=env,
                area_id=area_id,
                tensorboard_log=tb_log,
                seed=self.tc.seed,
                device=device,
                batch_size=mg_bs,
            )

        # Sub-agents
        sub_bs = self.tc.gpu_batch_size_sub if use_gpu else 32
        for name, env in coord.sub_envs.items():
            parts = name.split("_")
            role = parts[0]
            area_id = parts[1]
            self.agents[name] = SubPPOAgent(
                env=env,
                role=role,
                area_id=area_id,
                tensorboard_log=tb_log,
                seed=self.tc.seed,
                device=device,
                batch_size=sub_bs,
            )

    # ─── training ───────────────────────────────────────────────────

    def train(self) -> Dict:
        """
        Run the full hierarchical training loop.

        Trains all agents in a round-robin fashion using the StandaloneEnvWrapper
        to co-evolve policies.

        Returns:
            Training summary dict.
        """
        logger.info(f"Starting hierarchical training: {self.tc.total_timesteps} total steps")
        start_time = time.time()

        steps_done = 0
        best_eval_reward = -float("inf")

        # Create level-based agent lists
        central_agents = ["central"]
        mg_agents = [name for name in self.agents if name.startswith("mg_")]
        sub_agents = [name for name in self.agents if name.split("_")[0] in ("inverter", "renewable", "load")]

        # Group levels to train
        levels = [central_agents, mg_agents]
        if self.tc.train_sub_agents:
            levels.append(sub_agents)

        while steps_done < self.tc.total_timesteps:
            for level in levels:
                for agent_name in level:
                    agent = self.agents[agent_name]
                    
                    # Wrap the agent's environment so it can step standalone
                    wrapped_env = StandaloneEnvWrapper(agent_name, self.coordinator, self.agents)
                    agent.model.set_env(wrapped_env)
                    
                    # Train for a small burst
                    # Use a fixed number of steps per level update
                    burst_steps = self.tc.steps_per_level
                    logger.info(f"Level Update | Training '{agent_name}' for {burst_steps} steps …")
                    
                    # SB3 learn() will handle its own rollout collection
                    agent.model.learn(
                        total_timesteps=burst_steps,
                        reset_num_timesteps=False,
                    )
                    
                    steps_done += burst_steps
                    
                    # Periodic evaluation
                    if steps_done % self.tc.eval_freq < burst_steps:
                        eval_reward = self._evaluate()
                        if eval_reward > best_eval_reward:
                            best_eval_reward = eval_reward
                            self.save_all(Path(self.tc.log_dir) / "best")
                            logger.info(f"New best eval reward: {eval_reward:.3f}")

                    # Periodic checkpoint
                    if steps_done % self.tc.save_freq < burst_steps:
                        self.save_all(Path(self.tc.log_dir) / f"checkpoint_{steps_done}")

                    if steps_done >= self.tc.total_timesteps:
                        break
                if steps_done >= self.tc.total_timesteps:
                    break

        elapsed = time.time() - start_time
        summary = {
            "total_steps": steps_done,
            "elapsed_seconds": elapsed,
            "best_eval_reward": best_eval_reward,
        }

        # Save final models
        self.save_all(Path(self.tc.log_dir) / "final")

        logger.info(f"Training complete: {summary}")
        return summary

    def _run_episode(self) -> Dict[str, List[float]]:
        """Run one episode and collect rewards per agent."""
        all_obs = self.coordinator.reset()
        episode_rewards: Dict[str, List[float]] = {name: [] for name in self.agents}

        while not self.coordinator.is_done:
            # Get actions from all agents
            actions = {}
            for name, agent in self.agents.items():
                obs = all_obs.get(name)
                if obs is not None:
                    actions[name] = agent.predict(obs, deterministic=False)
                else:
                    env = self.coordinator.get_env(name)
                    actions[name] = np.zeros(env.action_space.shape)

            # Step all agents
            all_obs, all_rewards, done, info = self.coordinator.step(actions)

            # Record rewards
            for name, reward in all_rewards.items():
                if name in episode_rewards:
                    episode_rewards[name].append(reward)

        return episode_rewards

    def _evaluate(self, n_episodes: Optional[int] = None) -> float:
        """Evaluate agents deterministically. Returns mean central reward."""
        n_ep = n_episodes or self.tc.n_eval_episodes
        total_rewards = []

        for _ in range(n_ep):
            all_obs = self.coordinator.reset()
            ep_reward = 0.0

            while not self.coordinator.is_done:
                actions = {}
                for name, agent in self.agents.items():
                    obs = all_obs.get(name)
                    if obs is not None:
                        actions[name] = agent.predict(obs, deterministic=True)
                    else:
                        env = self.coordinator.get_env(name)
                        actions[name] = np.zeros(env.action_space.shape)

                all_obs, all_rewards, done, info = self.coordinator.step(actions)
                ep_reward += all_rewards.get("central", 0)

            total_rewards.append(ep_reward)

        return float(np.mean(total_rewards))

    # ─── save / load ────────────────────────────────────────────────

    def save_all(self, directory: Path) -> None:
        """Save all agent models to a directory."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for name, agent in self.agents.items():
            agent.save(str(directory / name))
        logger.info(f"Saved all {len(self.agents)} agents to {directory}")

    def load_all(self, directory: Path) -> None:
        """Load all agent models from a directory."""
        directory = Path(directory)
        for name, agent in self.agents.items():
            model_path = directory / name
            if model_path.with_suffix(".zip").exists():
                agent.load(str(model_path))
        logger.info(f"Loaded agents from {directory}")
