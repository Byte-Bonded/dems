"""
Tests for observation builder module.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock

from src.agent.observations import (
    MicrogridObsBuilder,
    CentralObsBuilder,
    SubAgentObsBuilder,
    MG_OBS_DIM,
    CENTRAL_OBS_DIM,
    INVERTER_SUB_OBS_DIM,
    RENEWABLE_SUB_OBS_DIM,
    LOAD_SUB_OBS_DIM,
)
from src.simulation.supergrid import AreaID


def _mock_profiles():
    """Create a mock StochasticProfileGenerator."""
    p = MagicMock()
    p.solar_irradiance.return_value = 800.0
    p.wind_speed.return_value = 12.0
    p.load_scale_factor.return_value = 1.0
    p.hour_of_day.return_value = 12.0
    return p


def _area_state():
    return {
        "total_generation_mw": 300.0,
        "total_load_mw": 280.0,
        "net_interchange_mw": 20.0,
        "total_losses_mw": 5.0,
        "max_line_loading_pct": 65.0,
        "avg_voltage_pu": 1.01,
        "min_voltage_pu": 0.96,
        "max_voltage_pu": 1.04,
    }


def _der_status():
    return {
        "solar": {"current_output_mw": 50.0},
        "wind": {"current_output_mw": 80.0},
        "ev_charging": {"current_load_mw": 10.0},
        "demand_response": {"avg_curtailment_frac": 0.05},
    }


def _gen_states():
    return {
        0: {"omega_pu": 1.0, "delta_deg": 15.0},
        1: {"omega_pu": 1.001, "delta_deg": 12.0},
    }


class TestMicrogridObsBuilder:
    """Tests for MicrogridObsBuilder."""

    def test_output_dimension(self):
        builder = MicrogridObsBuilder(area_id=AreaID.AREA_A, episode_length=288)
        obs = builder.build(
            area_state=_area_state(),
            der_status=_der_status(),
            gen_states=_gen_states(),
            frequency_hz=50.0,
            profiles=_mock_profiles(),
            step=100,
        )
        assert obs.shape == (MG_OBS_DIM,)

    def test_output_bounded(self):
        builder = MicrogridObsBuilder(area_id=AreaID.AREA_A, episode_length=288)
        obs = builder.build(
            area_state=_area_state(),
            der_status=_der_status(),
            gen_states=_gen_states(),
            frequency_hz=50.0,
            profiles=_mock_profiles(),
            step=0,
        )
        assert np.all(np.isfinite(obs))
        assert np.all(obs >= -2.0)
        assert np.all(obs <= 2.0)

    def test_different_timesteps(self):
        builder = MicrogridObsBuilder(area_id=AreaID.AREA_A, episode_length=288)
        profiles = _mock_profiles()
        obs1 = builder.build(
            area_state=_area_state(), der_status=_der_status(),
            gen_states=_gen_states(), frequency_hz=50.0,
            profiles=profiles, step=0,
        )
        # Change profile return for different step
        profiles.hour_of_day.return_value = 18.0
        obs2 = builder.build(
            area_state=_area_state(), der_status=_der_status(),
            gen_states=_gen_states(), frequency_hz=50.0,
            profiles=profiles, step=144,
        )
        assert not np.allclose(obs1, obs2)


class TestCentralObsBuilder:
    """Tests for CentralObsBuilder."""

    def test_output_dimension(self):
        builder = CentralObsBuilder(episode_length=288)
        global_state = {
            "areas": {
                "A": {"total_generation_mw": 300, "total_load_mw": 280, "net_interchange_mw": 20},
                "B": {"total_generation_mw": 320, "total_load_mw": 290, "net_interchange_mw": 30},
                "C": {"total_generation_mw": 280, "total_load_mw": 270, "net_interchange_mw": 10},
            },
            "global_metrics": {"total_generation_mw": 900, "total_load_mw": 840, "total_losses_mw": 10},
        }
        tie_line_flows = [{"flow_mw": 10.0}, {"flow_mw": 5.0}, {"flow_mw": -10.0},
                          {"flow_mw": 8.0}, {"flow_mw": -5.0}, {"flow_mw": -8.0}]
        obs = builder.build(
            global_state=global_state,
            tie_line_flows=tie_line_flows,
            frequency_hz=50.0,
            profiles=_mock_profiles(),
            step=100,
        )
        assert obs.shape == (CENTRAL_OBS_DIM,)

    def test_output_finite(self):
        builder = CentralObsBuilder(episode_length=288)
        obs = builder.build(
            global_state={"areas": {}, "global_metrics": {}},
            tie_line_flows=[],
            frequency_hz=50.0,
            profiles=_mock_profiles(),
            step=100,
        )
        assert np.all(np.isfinite(obs))


class TestSubAgentObsBuilder:
    """Tests for SubAgentObsBuilder."""

    def _mock_net(self):
        """Mock pandapower net with gen/res_gen/res_bus."""
        net = MagicMock()
        net.gen.index = [0, 1]
        net.gen.at = MagicMock(side_effect=lambda idx, col: {
            (0, "bus"): 5, (0, "max_p_mw"): 200, (0, "min_p_mw"): 50,
            (1, "bus"): 10, (1, "max_p_mw"): 180, (1, "min_p_mw"): 40,
        }.get((idx, col), 0.0))
        net.res_gen.index = [0, 1]
        net.res_gen.at = MagicMock(side_effect=lambda idx, col: {
            (0, "p_mw"): 120.0, (0, "q_mvar"): 30.0,
            (1, "p_mw"): 100.0, (1, "q_mvar"): 20.0,
        }.get((idx, col), 0.0))
        net.res_bus.index = list(range(117))
        net.res_bus.at = MagicMock(side_effect=lambda idx, col: 1.01)
        return net

    def test_inverter_obs_dimension(self):
        builder = SubAgentObsBuilder(episode_length=288)
        obs = builder.build_inverter(
            gen_idx=0,
            net=self._mock_net(),
            frequency_hz=50.0,
            area_state=_area_state(),
            step=100,
            profiles=_mock_profiles(),
        )
        assert obs.shape == (INVERTER_SUB_OBS_DIM,)

    def test_renewable_obs_dimension(self):
        builder = SubAgentObsBuilder(episode_length=288)
        obs = builder.build_renewable(
            resource_value=800.0, resource_max=1000.0,
            current_output_mw=40.0, rated_mw=50.0,
            curtail_frac=0.0, v_local_pu=1.01,
            frequency_hz=50.0, step=100, profiles=_mock_profiles(),
        )
        assert obs.shape == (RENEWABLE_SUB_OBS_DIM,)

    def test_load_obs_dimension(self):
        builder = SubAgentObsBuilder(episode_length=288)
        obs = builder.build_load(
            total_load_mw=280.0, ev_load_mw=10.0,
            dr_curtail_frac=0.05, v_local_pu=1.0,
            frequency_hz=50.0, load_scale=1.0,
            step=100, profiles=_mock_profiles(),
        )
        assert obs.shape == (LOAD_SUB_OBS_DIM,)

    def test_all_finite(self):
        builder = SubAgentObsBuilder(episode_length=288)
        profiles = _mock_profiles()
        obs1 = builder.build_inverter(
            gen_idx=0, net=self._mock_net(), frequency_hz=50.0,
            area_state=_area_state(), step=100, profiles=profiles,
        )
        obs2 = builder.build_renewable(
            resource_value=800.0, resource_max=1000.0,
            current_output_mw=40.0, rated_mw=50.0,
            curtail_frac=0.0, v_local_pu=1.01,
            frequency_hz=50.0, step=100, profiles=profiles,
        )
        obs3 = builder.build_load(
            total_load_mw=280.0, ev_load_mw=10.0,
            dr_curtail_frac=0.05, v_local_pu=1.0,
            frequency_hz=50.0, load_scale=1.0,
            step=100, profiles=profiles,
        )
        for obs in [obs1, obs2, obs3]:
            assert np.all(np.isfinite(obs))
