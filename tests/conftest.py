"""
Test configuration and fixtures for hierarchical multi-agent PPO system.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock

from src.core import GridManager
from src.simulation.der import DERType


@pytest.fixture
def grid_manager():
    """Create GridManager instance for tests."""
    return GridManager(num_nodes=10)


@pytest.fixture
def mock_physics_engine():
    """Create a fully-mocked PhysicsEngine for unit tests."""
    pe = MagicMock()

    pe.get_area_generators.return_value = [0, 1, 2]
    pe.get_generator_limits.return_value = {
        0: (50.0, 200.0), 1: (40.0, 180.0), 2: (30.0, 150.0),
    }

    solar = MagicMock(der_type=DERType.SOLAR_PV, capacity_kw=5000.0)
    wind = MagicMock(der_type=DERType.WIND, capacity_kw=8000.0)
    ev = MagicMock(der_type=DERType.EV_CHARGING, capacity_kw=2000.0)
    dr = MagicMock(der_type=DERType.DEMAND_RESPONSE, capacity_kw=1000.0)
    pe.get_area_der_specs.return_value = [solar, wind, ev, dr]

    pe.get_area_state.return_value = {
        "total_gen_mw": 300.0, "total_load_mw": 280.0,
        "net_interchange_mw": 20.0, "gen_headroom_mw": 100.0,
        "spinning_reserve_mw": 50.0, "max_line_loading_pct": 65.0,
        "mean_voltage_pu": 1.01, "min_voltage_pu": 0.96, "max_voltage_pu": 1.04,
    }
    pe.get_global_state.return_value = {
        "frequency_hz": 50.0, "total_gen_mw": 900.0,
        "total_load_mw": 840.0, "max_line_loading_pct": 70.0, "mean_voltage_pu": 1.0,
    }
    pe.get_tie_line_flows.return_value = {
        "A_B": 10.0, "A_C": 5.0, "B_A": -10.0,
        "B_C": 8.0, "C_A": -5.0, "C_B": -8.0,
    }
    pe.get_der_status.return_value = {
        "solar_mw": 50.0, "wind_mw": 80.0, "ev_mw": 10.0, "dr_mw": 5.0,
    }
    pe.get_generator_states.return_value = {
        0: {"p_mw": 120.0, "q_mvar": 30.0, "v_pu": 1.01,
            "omega_pu": 1.0, "delta_deg": 15.0, "p_max": 200.0, "p_min": 50.0},
        1: {"p_mw": 100.0, "q_mvar": 20.0, "v_pu": 1.02,
            "omega_pu": 1.0, "delta_deg": 12.0, "p_max": 180.0, "p_min": 40.0},
        2: {"p_mw": 80.0, "q_mvar": 15.0, "v_pu": 1.0,
            "omega_pu": 1.0, "delta_deg": 10.0, "p_max": 150.0, "p_min": 30.0},
    }

    pe.grid = MagicMock()
    pe.grid.dynamics = MagicMock()
    pe.grid.dynamics.get_generator_state.return_value = {"omega_pu": 1.0, "delta_deg": 15.0}
    pe.grid.dynamics.states = {"rocof_hz_s": 0.0}
    pe.grid.net = MagicMock()
    pe.grid.net.res_bus.__getitem__ = lambda self, k: np.ones(117)
    pe.grid.net.res_line.__getitem__ = lambda self, k: np.zeros(130)
    pe.step.return_value = {"frequency_hz": 50.0, "converged": True}
    pe.reset.return_value = None

    return pe
