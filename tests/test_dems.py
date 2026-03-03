"""
DEMS - Consolidated Test Suite
All essential tests in a single file
"""

import pytest
import numpy as np

from src.grid import DEMSGrid
from src.core import GridMonitor, GridManager


# ======================== FIXTURES ======================== #

@pytest.fixture
def grid():
    """
    Create a new DEMSGrid test fixture.
    
    Returns:
        DEMSGrid: A fresh DEMSGrid instance for use in tests.
    """
    return DEMSGrid()


@pytest.fixture
def grid_monitor():
    """Create GridMonitor instance for stability monitoring"""
    return GridMonitor()


@pytest.fixture
def grid_manager():
    """Create GridManager instance for grid management"""
    return GridManager(num_nodes=10)


# ======================== GRID TESTS ======================== #

class TestGrid:
    """SuperGrid and power flow tests"""
    
    def test_grid_initialization(self, grid):
        """
        Verify the DEMSGrid creates a SuperGrid and that the network contains 117 buses.
        
        Asserts that the grid's supergrid is initialized and that len(grid.supergrid.net.bus) == 117.
        """
        assert grid.supergrid is not None
        assert len(grid.supergrid.net.bus) == 117
        
    def test_power_flow_convergence(self, grid):
        """Test power flow converges with valid voltage profile"""
        result = grid.run_power_flow()
        
        assert result.converged
        assert result.total_generation_mw > 0
        assert result.total_load_mw > 0
        assert 0.95 <= result.min_voltage_pu <= 1.05
        assert 0.95 <= result.max_voltage_pu <= 1.10
        
    def test_three_area_topology(self, grid):
        """Test tri-area grid structure"""
        areas = grid.supergrid.net.bus["zone"].unique()
        assert len(areas) == 3
        assert len(grid.supergrid.net.gen) >= 27
        
    def test_der_integration(self, grid):
        """Test DER manager initialization"""
        der = grid.supergrid.initialize_der()
        status = der.get_status()
        
        assert status["solar"]["unit_count"] > 0
        assert status["wind"]["unit_count"] > 0


# ======================== DYNAMICS TESTS ======================== #

class TestDynamics:
    """Dynamic simulation tests"""
    
    def test_dynamics_initialization(self, grid):
        """
        Verify dynamics are initialized and the system frequency is set to 50.0 Hz.
        
        Parameters:
            grid: DEMSGrid test fixture used to run power flow and initialize dynamics.
        """
        grid.run_power_flow()
        grid.supergrid.initialize_dynamics()
        
        assert grid.supergrid.dynamics is not None
        assert grid.supergrid.get_system_frequency() == 50.0
        
    def test_frequency_response(self, grid):
        """Test frequency changes during simulation"""
        grid.run_power_flow()
        grid.supergrid.initialize_dynamics()
        
        # Step dynamics
        for _ in range(10):
            result = grid.supergrid.step_dynamics(dt=0.01)
            
        freq = result["system_frequency_hz"]
        assert 49.0 < freq < 51.0  # Within ±1 Hz


# ======================== CORE TESTS ======================== #

class TestCore:
    """Core module tests"""

    def test_grid_manager(self, grid_manager):
        """
        Verify that GridManager reports its configured node count and includes 'num_nodes' in its exported grid state.
        
        Asserts that `num_nodes` equals 10 and that the dictionary returned by `get_grid_state()` contains the key "num_nodes".
        """
        assert grid_manager.num_nodes == 10
        state = grid_manager.get_grid_state()
        assert "num_nodes" in state


# ======================== INTEGRATION TESTS ======================== #

class TestIntegration:
    """End-to-end integration tests"""
    
    def test_full_workflow(self):
        """Test complete Grid → DER → Power Flow workflow"""
        grid = DEMSGrid()
        
        # Initialize DER
        der = grid.supergrid.initialize_der()
        
        # Run power flow
        result = grid.run_power_flow()
        assert result.converged
        
        # Verify DER in power balance
        status = der.get_status()
        assert status["solar"]["current_output_mw"] >= 0
        assert status["wind"]["current_output_mw"] >= 0
        
    def test_multiple_power_flows(self):
        """
        Verifies that consecutive executions of the grid power flow converge on each of three runs.
        
        Asserts failure if any run does not converge.
        """
        grid = DEMSGrid()
        
        for i in range(3):
            result = grid.run_power_flow()
            assert result.converged, f"Run {i+1} failed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])