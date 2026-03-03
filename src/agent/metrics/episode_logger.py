"""
Episode-level metrics logger.

Collects per-step data across all agents and produces summary
statistics for training monitoring and TensorBoard logging.
"""

import json
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class AgentMetrics:
    """Accumulated metrics for one agent over an episode."""
    rewards: List[float] = field(default_factory=list)
    violations: List[int] = field(default_factory=list)
    actions: List[np.ndarray] = field(default_factory=list)

    @property
    def total_reward(self) -> float:
        return sum(self.rewards)

    @property
    def mean_reward(self) -> float:
        return float(np.mean(self.rewards)) if self.rewards else 0.0

    @property
    def total_violations(self) -> int:
        return sum(self.violations)

    @property
    def action_smoothness(self) -> float:
        """Mean absolute action change between steps."""
        if len(self.actions) < 2:
            return 0.0
        deltas = [
            float(np.mean(np.abs(self.actions[i] - self.actions[i - 1])))
            for i in range(1, len(self.actions))
        ]
        return float(np.mean(deltas))


class EpisodeLogger:
    """
    Logs per-episode metrics for all agents in the hierarchy.

    Usage::

        logger = EpisodeLogger()
        logger.start_episode()
        # ... run episode ...
        for step in steps:
            logger.log_step(agent_name, reward, violations, action)
        summary = logger.end_episode()
    """

    def __init__(self, log_dir: Optional[str] = None):
        self.log_dir = Path(log_dir) if log_dir else None
        self._episode_count = 0
        self._current: Dict[str, AgentMetrics] = {}
        self._history: List[Dict] = []

    def start_episode(self) -> None:
        """Begin a new episode."""
        self._episode_count += 1
        self._current = {}

    def log_step(
        self,
        agent_name: str,
        reward: float,
        violations: int = 0,
        action: Optional[np.ndarray] = None,
    ) -> None:
        """Record one step for one agent."""
        if agent_name not in self._current:
            self._current[agent_name] = AgentMetrics()

        metrics = self._current[agent_name]
        metrics.rewards.append(reward)
        metrics.violations.append(violations)
        if action is not None:
            metrics.actions.append(action.copy())

    def end_episode(self) -> Dict[str, Any]:
        """
        Finalize the episode and return summary.

        Returns:
            Summary dict with per-agent statistics.
        """
        summary: Dict[str, Any] = {
            "episode": self._episode_count,
            "agents": {},
        }

        for name, metrics in self._current.items():
            summary["agents"][name] = {
                "total_reward": metrics.total_reward,
                "mean_reward": metrics.mean_reward,
                "total_violations": metrics.total_violations,
                "action_smoothness": metrics.action_smoothness,
                "steps": len(metrics.rewards),
            }

        self._history.append(summary)

        # Save to disk if configured
        if self.log_dir:
            self._save_summary(summary)

        return summary

    def get_history(self) -> List[Dict]:
        """Return all episode summaries."""
        return self._history

    def _save_summary(self, summary: Dict) -> None:
        """Save episode summary to JSON."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        path = self.log_dir / "episode_metrics.jsonl"
        with open(path, "a") as f:
            f.write(json.dumps(summary, default=str) + "\n")
