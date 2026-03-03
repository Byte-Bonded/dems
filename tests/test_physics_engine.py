"""
Tests for PhysicsEngine (replacement for old GridOrchestrator).
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


class TestPhysicsEngine:
    """Tests for the PhysicsEngine class."""

    @pytest.fixture
    def mock_grid(self):
        """Create a mock SuperGrid for PhysicsEngine."""
        grid = MagicMock()
        grid.net = MagicMock()
        grid.net.gen = MagicMock()
        grid.net.gen.index = [0, 1, 2]
        grid.net.load = MagicMock()
        grid.net.load.__len__ = lambda self: 50
        grid.net.res_bus = MagicMock()
        grid.net.res_bus.__getitem__ = lambda self, k: np.ones(117)
        grid.net.res_line = MagicMock()
        grid.net.res_line.__getitem__ = lambda self, k: np.full(130, 50.0) if k == "loading_percent" else np.zeros(130)
        
        grid.run_power_flow.return_value = True
        grid.dynamics = MagicMock()
        grid.dynamics.states = {"rocof_hz_s": 0.0}
        grid.dynamics.get_generator_state.return_value = {
            "omega_pu": 1.0, "delta_deg": 15.0, "p_mw": 120.0
        }
        grid.dynamics.step.return_value = {"frequency_hz": 50.0}
        
        grid.der_manager = MagicMock()
        grid.der_manager.get_total_generation.return_value = 130.0
        grid.der_manager.get_status.return_value = {
            "solar_mw": 50.0, "wind_mw": 80.0, "ev_mw": 10.0, "dr_mw": 5.0
        }
        
        grid.get_area_buses.return_value = list(range(39))
        grid.get_area_generators.return_value = [0, 1]
        
        return grid

    def test_physics_engine_import(self):
        """PhysicsEngine should be importable."""
        from src.simulation.orchestrator import PhysicsEngine
        assert PhysicsEngine is not None

    def test_backward_compat_alias(self):
        """GridOrchestrator should be an alias for PhysicsEngine."""
        from src.simulation.orchestrator import GridOrchestrator, PhysicsEngine
        assert GridOrchestrator is PhysicsEngine

    def test_stochastic_profile_generator_exists(self):
        """StochasticProfileGenerator should still be available."""
        from src.simulation.orchestrator import StochasticProfileGenerator
        assert StochasticProfileGenerator is not None

    def test_scenario_config_exists(self):
        """ScenarioConfig should still be available."""
        from src.simulation.orchestrator import ScenarioConfig
        assert ScenarioConfig is not None
