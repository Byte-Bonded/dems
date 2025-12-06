"""
DEMS Environment for RL Training
Gym-compatible environment for training RL agents
"""

import gym
from gym import spaces
import numpy as np
from typing import Tuple, Dict, Any


class DEMSEnvironment(gym.Env):
    """
    Custom Gym environment for DEMS
    Agents interact with this environment to learn optimal energy management
    """

    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(self, num_nodes: int = 10, max_steps: int = 1000):
        super().__init__()

        self.num_nodes = num_nodes
        self.max_steps = max_steps
        self.current_step = 0

        # Observation space: [storage_level, load, frequency, voltage] * nodes
        obs_size = 4 + (num_nodes * 4)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(obs_size,), dtype=np.float32
        )

        # Action space: distribution decisions for each node
        self.action_space = spaces.Box(
            low=0, high=1, shape=(num_nodes,), dtype=np.float32
        )

        self.state = None
        self.episode_reward = 0

    def _get_obs(self) -> np.ndarray:
        """Get current observation"""
        if self.state is None:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        return self.state.astype(np.float32)

    def _calculate_reward(self, action: np.ndarray) -> float:
        """
        Calculate reward based on:
        - Energy efficiency
        - Grid stability
        - Storage level balance
        """
        efficiency_reward = 1.0
        stability_reward = 0.5
        balance_reward = 0.3

        reward = efficiency_reward + stability_reward + balance_reward
        return float(reward)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step in the environment
        """
        self.current_step += 1

        # Update state based on action
        self.state = np.random.rand(self.observation_space.shape[0]).astype(np.float32)

        # Calculate reward
        reward = self._calculate_reward(action)
        self.episode_reward += reward

        # Check if episode is done
        done = self.current_step >= self.max_steps

        # Info dictionary
        info = {
            "step": self.current_step,
            "episode_reward": self.episode_reward,
            "action_applied": action,
        }

        return self._get_obs(), reward, done, info

    def reset(self) -> np.ndarray:
        """Reset environment"""
        self.current_step = 0
        self.episode_reward = 0
        self.state = np.random.rand(self.observation_space.shape[0]).astype(np.float32)
        return self._get_obs()

    def render(self, mode: str = "human") -> None:
        """Render environment state"""
        print(f"Step: {self.current_step}, Episode Reward: {self.episode_reward:.2f}")

    def close(self) -> None:
        """Close environment"""
        pass
