"""
Tests for DEMS Environment
"""

import pytest
import numpy as np
from src.agent import DEMSEnvironment


class TestDEMSEnvironment:
    """DEMS Environment test suite"""

    def test_initialization(self, dems_environment):
        """Test environment initialization"""
        assert dems_environment.num_nodes == 10
        assert dems_environment.max_steps == 100
        assert dems_environment.current_step == 0

    def test_reset(self, dems_environment):
        """Test environment reset"""
        obs = dems_environment.reset()
        assert obs is not None
        assert obs.shape == dems_environment.observation_space.shape
        assert dems_environment.current_step == 0
        assert dems_environment.episode_reward == 0

    def test_step(self, dems_environment):
        """Test environment step"""
        dems_environment.reset()
        action = dems_environment.action_space.sample()
        obs, reward, done, info = dems_environment.step(action)

        assert obs is not None
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        assert dems_environment.current_step == 1

    def test_step_termination(self, dems_environment):
        """Test episode termination"""
        dems_environment.reset()
        done = False
        steps = 0

        while not done and steps < 150:
            action = dems_environment.action_space.sample()
            _, _, done, _ = dems_environment.step(action)
            steps += 1

        assert done is True
        assert dems_environment.current_step >= dems_environment.max_steps

    def test_observation_space(self, dems_environment):
        """Test observation space"""
        obs = dems_environment.reset()
        assert obs.shape[0] > 0
        assert obs.dtype == np.float32
