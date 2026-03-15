"""
PPO agent wrappers for the hierarchical multi-agent system.

Wraps stable-baselines3 PPO agents with configuration appropriate
for each hierarchy level:
- CentralPPOAgent: [256, 256] policy, global obs, coordination actions
- MicrogridPPOAgent: [256, 256] policy, area obs, local actions
- SubPPOAgent: [128, 128] policy, narrow obs, focused actions

Supports:
- Creating fresh agents
- Loading pre-trained agents
- CTDE shared critic (via custom feature extractors)
"""

import logging
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional, Type

import gymnasium as gym

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    torch = None
    _HAS_TORCH = False

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    _HAS_SB3 = True
except ImportError:
    PPO = None
    BaseCallback = None
    _HAS_SB3 = False


def detect_device(requested: str = "auto") -> str:
    """Detect best available device.

    Args:
        requested: 'auto', 'cuda', 'mps', or 'cpu'
    Returns:
        Device string for SB3: 'cuda', 'mps', or 'cpu'
    """
    if not _HAS_TORCH:
        return "cpu"
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return requested

logger = logging.getLogger(__name__)


class PPOAgentWrapper:
    """
    Base wrapper around SB3 PPO for any hierarchy level.

    Handles creation, prediction, saving, and loading.
    """

    def __init__(
        self,
        env: gym.Env,
        name: str,
        net_arch: list,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        tensorboard_log: Optional[str] = None,
        seed: Optional[int] = None,
        device: str = "auto",
    ):
        if not _HAS_SB3:
            raise ImportError("stable-baselines3 is required. Install with: pip install stable-baselines3")

        self.name = name
        self.env = env
        self.device = detect_device(device)

        self.model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            tensorboard_log=tensorboard_log,
            policy_kwargs={"net_arch": net_arch},
            seed=seed,
            verbose=0,
            device=self.device,
        )

        logger.info(
            f"PPOAgentWrapper '{name}': "
            f"obs={env.observation_space.shape}, "
            f"act={env.action_space.shape}, "
            f"arch={net_arch}, "
            f"device={self.device}"
        )

    def predict(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Predict action from observation."""
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return action

    def learn(self, total_timesteps: int, callback=None, **kwargs) -> None:
        """Train the agent."""
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            **kwargs,
        )

    def save(self, path: str) -> None:
        """Save model to disk."""
        self.model.save(path)
        logger.info(f"Saved {self.name} to {path}")

    def load(self, path: str) -> None:
        """Load model from disk."""
        self.model = PPO.load(path, env=self.env, device=self.device)
        logger.info(f"Loaded {self.name} from {path} (device={self.device})")

    @classmethod
    def from_pretrained(cls, path: str, env: gym.Env, name: str, device: str = "auto") -> "PPOAgentWrapper":
        """Create wrapper from a pre-trained model."""
        wrapper = cls.__new__(cls)
        wrapper.name = name
        wrapper.env = env
        wrapper.device = detect_device(device)
        wrapper.model = PPO.load(path, env=env, device=wrapper.device)
        return wrapper


class CentralPPOAgent(PPOAgentWrapper):
    """PPO agent for central coordination. [256, 256] architecture."""

    def __init__(self, env: gym.Env, **kwargs):
        defaults = {
            "name": "central",
            "net_arch": [256, 256],
            "learning_rate": 3e-4,
            "n_steps": 4096,
            "batch_size": 256,
        }
        defaults.update(kwargs)
        super().__init__(env=env, **defaults)


class MicrogridPPOAgent(PPOAgentWrapper):
    """PPO agent for microgrid-level control. [256, 256] architecture."""

    def __init__(self, env: gym.Env, area_id: str = "A", **kwargs):
        defaults = {
            "name": f"mg_{area_id}",
            "net_arch": [256, 256],
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 128,
        }
        defaults.update(kwargs)
        super().__init__(env=env, **defaults)


class SubPPOAgent(PPOAgentWrapper):
    """PPO agent for sub-agent control. [128, 128] architecture."""

    def __init__(self, env: gym.Env, role: str = "inverter", area_id: str = "A", **kwargs):
        defaults = {
            "name": f"{role}_{area_id}",
            "net_arch": [128, 128],
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
        }
        defaults.update(kwargs)
        super().__init__(env=env, **defaults)
