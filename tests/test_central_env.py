"""
Tests for CentralCoordEnv gymnasium environment.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock
import gymnasium as gym

from src.agent.environments.central_env import CentralCoordEnv
from src.agent.observations import CENTRAL_OBS_DIM


def _make_mock_physics():
    pe = MagicMock()
    pe.get_area_state.return_value = {
        "total_generation_mw": 300.0, "total_load_mw": 280.0,
        "net_interchange_mw": 20.0, "total_losses_mw": 5.0,
        "max_line_loading_pct": 65.0, "avg_voltage_pu": 1.01,
        "min_voltage_pu": 0.96, "max_voltage_pu": 1.04,
    }
    pe.get_global_state.return_value = {
        "frequency_hz": 50.0, "total_gen_mw": 900.0,
        "total_load_mw": 840.0, "max_line_loading_pct": 70.0, "mean_voltage_pu": 1.0,
        "areas": {
            "A": {"total_generation_mw": 300, "total_load_mw": 280, "net_interchange_mw": 20},
            "B": {"total_generation_mw": 320, "total_load_mw": 290, "net_interchange_mw": 30},
            "C": {"total_generation_mw": 280, "total_load_mw": 270, "net_interchange_mw": 10},
        },
        "global_metrics": {"total_generation_mw": 900, "total_load_mw": 840, "total_losses_mw": 10},
    }
    pe.get_tie_line_flows.return_value = [
        {"flow_mw": 10.0}, {"flow_mw": 5.0}, {"flow_mw": -10.0},
        {"flow_mw": 8.0}, {"flow_mw": -5.0}, {"flow_mw": -8.0},
    ]
    pe.get_der_status.return_value = {}
    pe.get_generator_states.return_value = {}

    # Scalar properties
    pe.system_frequency_hz = 50.0
    pe.is_done = False
    pe.step_count = 0
    pe.last_pf_result = None

    pe.sg = MagicMock()
    pe.sg.net = MagicMock()

    pe.profiles = MagicMock()
    pe.profiles.solar_irradiance.return_value = 800.0
    pe.profiles.wind_speed.return_value = 12.0
    pe.profiles.load_scale_factor.return_value = 1.0
    pe.profiles.hour_of_day.return_value = 12.0

    pe.grid = MagicMock()
    pe.grid.dynamics = MagicMock()
    pe.grid.dynamics.states = {"rocof_hz_s": 0.0}
    return pe


class TestCentralCoordEnv:

    def _make_env(self):
        return CentralCoordEnv(physics=_make_mock_physics())

    def test_creation(self):
        env = self._make_env()
        assert env is not None

    def test_observation_space(self):
        env = self._make_env()
        assert env.observation_space.shape == (CENTRAL_OBS_DIM,)

    def test_action_space(self):
        env = self._make_env()
        assert isinstance(env.action_space, gym.spaces.Box)

    def test_reset(self):
        env = self._make_env()
        obs, info = env.reset()
        assert obs.shape == (CENTRAL_OBS_DIM,)

    def test_step(self):
        env = self._make_env()
        env.reset()
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert obs.shape == (CENTRAL_OBS_DIM,)
        assert isinstance(reward, (int, float))
