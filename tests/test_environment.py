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
        result = dems_environment.reset()
        # gymnasium reset() returns (obs, info) tuple
        if isinstance(result, tuple):
            obs = result[0]
        else:
            obs = result
        assert obs is not None
        assert obs.shape == dems_environment.observation_space.shape
        assert dems_environment.current_step == 0
        assert dems_environment.episode_reward == 0

    def test_step(self, dems_environment):
        """Test environment step"""
        dems_environment.reset()
        action = dems_environment.action_space.sample()
        result = dems_environment.step(action)
        # gymnasium step() returns (obs, reward, terminated, truncated, info)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs, reward, done, info = result

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
            result = dems_environment.step(action)
            if len(result) == 5:
                _, _, terminated, truncated, _ = result
                done = terminated or truncated
            else:
                _, _, done, _ = result
            steps += 1

        assert done is True

    def test_observation_space(self, dems_environment):
        """Test observation space"""
        result = dems_environment.reset()
        obs = result[0] if isinstance(result, tuple) else result
        assert obs.shape[0] > 0
        assert obs.dtype == np.float32
