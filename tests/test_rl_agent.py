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
        assert rl_agent.observation_space_size == 50
        assert rl_agent.action_space_size == 10
        assert rl_agent.learning_rate == 0.0003

    def test_predict_untrained(self, rl_agent):
        """Test prediction with untrained model"""
        state = np.random.rand(50)
        action, info = rl_agent.predict(state)
        assert action is not None
        assert "info" in info
        assert info["info"] == "untrained"

    def test_train(self, rl_agent):
        """Test training"""
        experiences = [1, 2, 3, 4, 5]
        result = rl_agent.train(experiences)
        assert result["status"] == "training"
        assert result["episodes"] == 5

    def test_get_training_stats(self, rl_agent):
        """Test getting training stats"""
        stats = rl_agent.get_training_stats()
        assert "training_episodes" in stats
        assert "avg_reward" in stats

    def test_repr(self, rl_agent):
        """Test string representation"""
        repr_str = repr(rl_agent)
        assert "RLAgent" in repr_str
        assert "50" in repr_str
        assert "10" in repr_str
