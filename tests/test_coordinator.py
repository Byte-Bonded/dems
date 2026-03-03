"""
Tests for MultiAgentStepCoordinator.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from src.simulation.supergrid import AreaID
from src.simulation.der import DERType


def _mock_pe_instance():
    """Build a fully-mocked PhysicsEngine instance."""
    pe = MagicMock()
    pe.get_area_generators.return_value = [0, 1]
    pe.get_generator_limits.side_effect = lambda gi: {
        0: (50.0, 200.0), 1: (40.0, 180.0),
    }.get(gi, (0.0, 100.0))

    solar = MagicMock(der_type=DERType.SOLAR_PV, capacity_kw=5000.0, capacity_mw=5.0, name="solar")
    wind = MagicMock(der_type=DERType.WIND, capacity_kw=8000.0, capacity_mw=8.0, name="wind")
    ev = MagicMock(der_type=DERType.EV_CHARGING, capacity_kw=2000.0, capacity_mw=2.0, name="ev")
    dr = MagicMock(der_type=DERType.DEMAND_RESPONSE, capacity_kw=1000.0, capacity_mw=1.0, name="dr")
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

    pe.system_frequency_hz = 50.0
    pe.is_done = False
    pe.step_count = 0
    pe.last_pf_result = None

    pe.sg = MagicMock()
    pe.sg.net = MagicMock()
    pe.sg.net.gen = MagicMock()
    pe.sg.net.gen.index = [0, 1]
    pe.sg.net.gen.at = MagicMock(side_effect=lambda i, c: {
        (0, "bus"): 5, (0, "max_p_mw"): 200, (0, "min_p_mw"): 50,
        (1, "bus"): 10, (1, "max_p_mw"): 180, (1, "min_p_mw"): 40,
    }.get((i, c), 0.0))
    pe.sg.net.res_gen = MagicMock()
    pe.sg.net.res_gen.index = [0, 1]
    pe.sg.net.res_gen.at = MagicMock(side_effect=lambda i, c: 120.0)
    pe.sg.net.res_bus = MagicMock()
    pe.sg.net.res_bus.index = list(range(117))
    pe.sg.net.res_bus.at = MagicMock(return_value=1.01)
    pe.sg.net.res_line = MagicMock()

    pe.profiles = MagicMock()
    pe.profiles.solar_irradiance.return_value = 800.0
    pe.profiles.wind_speed.return_value = 12.0
    pe.profiles.load_scale_factor.return_value = 1.0
    pe.profiles.hour_of_day.return_value = 12.0

    pe.der_manager = MagicMock()
    pe.step.return_value = {"frequency_hz": 50.0, "converged": True}
    pe.reset.return_value = None

    return pe


class TestMultiAgentStepCoordinator:
    """Tests for the coordinator that orchestrates the full hierarchy."""

    @pytest.fixture
    def mock_physics_class(self):
        """Patch PhysicsEngine so coordinator can be instantiated without real grid."""
        with patch("src.agent.coordinator.PhysicsEngine") as MockPE:
            MockPE.return_value = _mock_pe_instance()
            yield MockPE

    def test_coordinator_creation(self, mock_physics_class):
        from src.agent.coordinator import MultiAgentStepCoordinator
        coord = MultiAgentStepCoordinator()
        assert coord is not None

    def test_coordinator_has_all_envs(self, mock_physics_class):
        from src.agent.coordinator import MultiAgentStepCoordinator
        coord = MultiAgentStepCoordinator()
        assert coord.central_env is not None
        assert len(coord.mg_envs) == 3
        assert len(coord.sub_envs) == 9

    def test_coordinator_reset(self, mock_physics_class):
        from src.agent.coordinator import MultiAgentStepCoordinator
        coord = MultiAgentStepCoordinator()
        all_obs = coord.reset()
        assert "central" in all_obs
        assert any("mg" in k for k in all_obs)
