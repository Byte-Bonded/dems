"""
Training callbacks for DEMS RL Agent
Logs episode rewards, losses, and grid metrics during training.
Saves periodic checkpoints and writes training history to JSON.
"""

import os
import json
import time
import numpy as np
from typing import Dict, List, Optional
from stable_baselines3.common.callbacks import BaseCallback


class DEMSTrainingCallback(BaseCallback):
    """
    Custom callback that records per-episode metrics during PPO training.

    Tracks:
    - Episode rewards (mean, min, max)
    - Episode lengths
    - Grid-specific metrics (frequency stability, storage balance, supply-demand error)
    - Policy loss, value loss, entropy (from SB3 logger)
    - Wall-clock training time
    - Periodic model checkpoints

    All metrics are stored in-memory and can be dumped to JSON.
    """

    def __init__(
        self,
        log_dir: str = "logs/rl",
        checkpoint_freq: int = 5000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.checkpoint_freq = checkpoint_freq

        # Tracked metrics
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_mean_rewards: List[float] = []
        self.timesteps_log: List[int] = []
        self.wall_times: List[float] = []

        # Per-step accumulators (reset each episode)
        self._ep_reward = 0.0
        self._ep_len = 0
        self._ep_freq_penalties: List[float] = []
        self._ep_storage_ratios: List[float] = []

        # Aggregate grid metrics per logged window
        self.avg_frequency_penalties: List[float] = []
        self.avg_storage_ratios: List[float] = []
        self.policy_losses: List[float] = []
        self.value_losses: List[float] = []
        self.entropies: List[float] = []

        self._start_time: Optional[float] = None
        self._last_checkpoint_step = 0

    def _on_training_start(self) -> None:
        os.makedirs(self.log_dir, exist_ok=True)
        self._start_time = time.time()

    def _on_step(self) -> bool:
        # Accumulate per-step info
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", [])
        dones = self.locals.get("dones", [])

        for i, info in enumerate(infos):
            reward_val = float(rewards[i]) if i < len(rewards) else 0.0
            self._ep_reward += reward_val
            self._ep_len += 1

            # Grid metrics from info dict
            freq = info.get("global_frequency", 50.0)
            self._ep_freq_penalties.append(abs(freq - 50.0))
            self._ep_storage_ratios.append(info.get("avg_storage_ratio", 0.5))

            done = bool(dones[i]) if i < len(dones) else False
            if done:
                self.episode_rewards.append(self._ep_reward)
                self.episode_lengths.append(self._ep_len)
                self.timesteps_log.append(self.num_timesteps)
                self.wall_times.append(time.time() - self._start_time)

                # Window mean
                window = self.episode_rewards[-100:]
                self.episode_mean_rewards.append(float(np.mean(window)))

                # Grid metrics for this episode
                if self._ep_freq_penalties:
                    self.avg_frequency_penalties.append(float(np.mean(self._ep_freq_penalties)))
                if self._ep_storage_ratios:
                    self.avg_storage_ratios.append(float(np.mean(self._ep_storage_ratios)))

                if self.verbose > 0 and len(self.episode_rewards) % 10 == 0:
                    print(
                        f"  Episode {len(self.episode_rewards):>4d} | "
                        f"Reward: {self._ep_reward:>8.2f} | "
                        f"Mean(100): {self.episode_mean_rewards[-1]:>8.2f} | "
                        f"Len: {self._ep_len}"
                    )

                # Reset accumulators
                self._ep_reward = 0.0
                self._ep_len = 0
                self._ep_freq_penalties = []
                self._ep_storage_ratios = []

        # Periodic checkpoint
        if (
            self.checkpoint_freq > 0
            and self.num_timesteps - self._last_checkpoint_step >= self.checkpoint_freq
        ):
            ckpt_path = os.path.join(self.log_dir, f"checkpoint_{self.num_timesteps}")
            self.model.save(ckpt_path)
            self._last_checkpoint_step = self.num_timesteps
            if self.verbose > 0:
                print(f"  Checkpoint saved: {ckpt_path}")

        return True  # continue training

    def _on_rollout_end(self) -> None:
        """Capture policy/value losses from the SB3 logger after each rollout."""
        try:
            logger = self.model.logger
            if hasattr(logger, "name_to_value"):
                vals = logger.name_to_value
                if "train/policy_gradient_loss" in vals:
                    self.policy_losses.append(float(vals["train/policy_gradient_loss"]))
                if "train/value_loss" in vals:
                    self.value_losses.append(float(vals["train/value_loss"]))
                if "train/entropy_loss" in vals:
                    self.entropies.append(float(vals["train/entropy_loss"]))
        except Exception:
            pass

    def _on_training_end(self) -> None:
        self.save_history()

    def get_metrics(self) -> Dict:
        """Return all tracked metrics as a dictionary."""
        return {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "episode_mean_rewards": self.episode_mean_rewards,
            "timesteps": self.timesteps_log,
            "wall_times": self.wall_times,
            "avg_frequency_penalties": self.avg_frequency_penalties,
            "avg_storage_ratios": self.avg_storage_ratios,
            "policy_losses": self.policy_losses,
            "value_losses": self.value_losses,
            "entropies": self.entropies,
            "total_episodes": len(self.episode_rewards),
            "total_timesteps": self.num_timesteps if hasattr(self, "num_timesteps") else 0,
            "training_time_s": (time.time() - self._start_time) if self._start_time else 0,
        }

    def save_history(self, path: Optional[str] = None) -> str:
        """Save training history to JSON."""
        if path is None:
            path = os.path.join(self.log_dir, "training_history.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.get_metrics(), f, indent=2)
        return path
