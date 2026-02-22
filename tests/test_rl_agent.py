"""
Tests for RL Agent
"""

import pytest
import numpy as np
from src.agent import RLAgent


class TestRLAgent:
    """RL Agent test suite"""

    def test_initialization(self, rl_agent):
        """Test agent initialization"""
        assert rl_agent.algorithm == "PPO"
        assert rl_agent.learning_rate == 0.0003
        assert rl_agent.env is not None

    def test_predict_untrained(self, rl_agent):
        """Test prediction with untrained model"""
        obs_shape = rl_agent.env.observation_space.shape
        state = np.random.rand(*obs_shape).astype(np.float32)
        action, info = rl_agent.predict(state)
        assert action is not None
        assert "info" in info
        assert info["info"] == "untrained"

    def test_get_training_stats(self, rl_agent):
        """Test getting training stats"""
        stats = rl_agent.get_training_stats()
        assert "algorithm" in stats
        assert "training_runs" in stats
        assert "total_timesteps" in stats
        assert stats["algorithm"] == "PPO"

    def test_repr(self, rl_agent):
        """Test string representation"""
        repr_str = repr(rl_agent)
        assert "RLAgent" in repr_str
        assert "PPO" in repr_str
