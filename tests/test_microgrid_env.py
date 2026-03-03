"""
Tests for MicrogridEnv gymnasium environment.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock
import gymnasium as gym

from src.agent.environments.microgrid_env import MicrogridEnv
from src.agent.observations import MG_OBS_DIM
from src.simulation.supergrid import AreaID
from src.simulation.der import DERType
from src.simulation.orchestrator import ScenarioConfig


def _make_mock_physics():
    """Create a mock PhysicsEngine sufficient for MicrogridEnv."""
    pe = MagicMock()

    pe.get_area_generators.return_value = [0, 1]
    
    # get_generator_limits called with single gen idx → returns (min, max)
    pe.get_generator_limits.side_effect = lambda gi: {
        0: (50.0, 200.0), 1: (40.0, 180.0),
    }.get(gi, (0.0, 100.0))

    solar = MagicMock(der_type=DERType.SOLAR_PV, capacity_kw=5000.0, name="solar_A")
    wind = MagicMock(der_type=DERType.WIND, capacity_kw=8000.0, name="wind_A")
    ev = MagicMock(der_type=DERType.EV_CHARGING, capacity_kw=2000.0, name="ev_A")
    dr = MagicMock(der_type=DERType.DEMAND_RESPONSE, capacity_kw=1000.0, name="dr_A")
    pe.get_area_der_specs.return_value = [solar, wind, ev, dr]

    pe.get_area_state.return_value = {
        "total_generation_mw": 300.0, "total_load_mw": 280.0,
        "net_interchange_mw": 20.0, "total_losses_mw": 5.0,
        "max_line_loading_pct": 65.0, "avg_voltage_pu": 1.01,
        "min_voltage_pu": 0.96, "max_voltage_pu": 1.04,
    }
    pe.get_global_state.return_value = {
        "frequency_hz": 50.0, "total_gen_mw": 900.0,
        "total_load_mw": 840.0, "max_line_loading_pct": 70.0, "mean_voltage_pu": 1.0,
    }
    pe.get_der_status.return_value = {
        "solar": {"current_output_mw": 50.0},
        "wind": {"current_output_mw": 80.0},
        "ev_charging": {"current_load_mw": 10.0},
        "demand_response": {"avg_curtailment_frac": 0.05},
    }
    pe.get_generator_states.return_value = {
        0: {"p_mw": 120.0, "q_mvar": 30.0, "v_pu": 1.01,
            "omega_pu": 1.0, "delta_deg": 15.0, "p_max": 200.0, "p_min": 50.0},
        1: {"p_mw": 100.0, "q_mvar": 20.0, "v_pu": 1.02,
            "omega_pu": 1.0, "delta_deg": 12.0, "p_max": 180.0, "p_min": 40.0},
    }
    pe.get_tie_line_flows.return_value = {}

    # Scalar properties the env reads
    pe.system_frequency_hz = 50.0
    pe.is_done = False
    pe.step_count = 0

    # last_pf_result (set to None to skip pandapower-dependent code path)
    pe.last_pf_result = None

    # sg.net mock (needed by _apply_actions and _build_obs)
    pe.sg = MagicMock()
    pe.sg.net = MagicMock()

    # profiles mock
    pe.profiles = MagicMock()
    pe.profiles.solar_irradiance.return_value = 800.0
    pe.profiles.wind_speed.return_value = 12.0
    pe.profiles.load_scale_factor.return_value = 1.0
    pe.profiles.hour_of_day.return_value = 12.0

    # DER manager
    pe.der_manager = MagicMock()

    # Grid/dynamics for obs builder fallback
    pe.grid = MagicMock()
    pe.grid.dynamics = MagicMock()
    pe.grid.dynamics.get_generator_state.return_value = {"omega_pu": 1.0, "delta_deg": 15.0}
    pe.grid.dynamics.states = {"rocof_hz_s": 0.0}

    return pe


class TestMicrogridEnv:
    """Tests for MicrogridEnv gymnasium environment."""

    def _make_env(self):
        pe = _make_mock_physics()
        return MicrogridEnv(physics=pe, area_id=AreaID.AREA_A)

    def test_creation(self):
        env = self._make_env()
        assert env is not None

    def test_observation_space_shape(self):
        env = self._make_env()
        assert env.observation_space.shape == (MG_OBS_DIM,)

    def test_action_space_bounded(self):
        env = self._make_env()
        assert isinstance(env.action_space, gym.spaces.Box)
        # action_dim = n_gens(2) + solar(1) + wind(1) + ev(1) + dr(1) = 6
        assert env.action_space.shape[0] == 6

    def test_reset_returns_obs_and_info(self):
        env = self._make_env()
        obs, info = env.reset()
        assert obs.shape == (MG_OBS_DIM,)
        assert isinstance(info, dict)

    def test_step_returns_five_tuple(self):
        env = self._make_env()
        env.reset()
        action = env.action_space.sample()
        result = env.step(action)
        assert len(result) == 5

    def test_step_obs_shape(self):
        env = self._make_env()
        env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (MG_OBS_DIM,)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
