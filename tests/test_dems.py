"""
DEMS - Consolidated Test Suite
All essential tests in a single file
"""

import pytest
import numpy as np

from src.grid import DEMSGrid
from src.core import EnergyManager, GridManager
from src.agent import DEMSEnvironment, RLAgent


# ======================== FIXTURES ======================== #

@pytest.fixture
def grid():
    """Create DEMSGrid instance"""
    return DEMSGrid()


@pytest.fixture
def energy_manager():
    """Create EnergyManager instance"""
    return EnergyManager(grid_size=10, storage_capacity=1000.0)


@pytest.fixture
def grid_manager():
    """Create GridManager instance"""
    return GridManager(num_nodes=10)


@pytest.fixture
def environment():
    """Create DEMSEnvironment instance"""
    return DEMSEnvironment(num_nodes=10, max_steps=100)


# ======================== GRID TESTS ======================== #

class TestGrid:
    """SuperGrid and power flow tests"""
    
    def test_grid_initialization(self, grid):
        """Test 117-bus SuperGrid initialization"""
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
        assert status["battery"]["unit_count"] > 0


# ======================== DYNAMICS TESTS ======================== #

class TestDynamics:
    """Dynamic simulation tests"""
    
    def test_dynamics_initialization(self, grid):
        """Test dynamics module initializes"""
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


# ======================== ENVIRONMENT TESTS ======================== #

class TestEnvironment:
    """RL Environment tests"""
    
    def test_initialization(self, environment):
        """Test environment initialization"""
        assert environment.num_nodes == 10
        assert environment.max_steps == 100
        
    def test_reset(self, environment):
        """Test reset returns valid observation"""
        obs = environment.reset()
        assert obs.shape == environment.observation_space.shape
        assert environment.current_step == 0
        
    def test_step(self, environment):
        """Test step returns valid outputs"""
        environment.reset()
        action = environment.action_space.sample()
        obs, reward, done, info = environment.step(action)
        
        assert obs is not None
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        
    def test_episode_termination(self, environment):
        """Test episode terminates at max_steps"""
        environment.reset()
        for _ in range(100):
            _, _, done, _ = environment.step(environment.action_space.sample())
        assert done


# ======================== RL AGENT TESTS ======================== #

class TestRLAgent:
    """RL Agent tests"""
    
    def test_agent_initialization(self, environment):
        """Test agent initializes with environment"""
        agent = RLAgent(env=environment)
        assert agent.observation_space is not None
        assert agent.action_space is not None


# ======================== CORE TESTS ======================== #

class TestCore:
    """Core module tests"""
    
    def test_energy_manager(self, energy_manager):
        """Test EnergyManager basic functionality"""
        state = energy_manager.get_current_state()
        assert "storage_level" in state
        assert state["storage_level"] == 500.0
        
    def test_grid_manager(self, grid_manager):
        """Test GridManager basic functionality"""
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
        total_der = status["solar"]["current_output_mw"] + status["wind"]["current_output_mw"]
        assert total_der > 0
        
    def test_multiple_power_flows(self):
        """Test repeated power flow runs"""
        grid = DEMSGrid()
        
        for i in range(3):
            result = grid.run_power_flow()
            assert result.converged, f"Run {i+1} failed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
