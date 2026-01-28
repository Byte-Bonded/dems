"""
Test configuration and fixtures
"""

import pytest
from src.core import EnergyManager, GridManager
from src.agent import RLAgent, DEMSEnvironment


@pytest.fixture
def energy_manager():
    """Create EnergyManager instance for tests"""
    return EnergyManager(grid_size=10, storage_capacity=1000.0)


@pytest.fixture
def grid_manager():
    """Create GridManager instance for tests"""
    return GridManager(num_nodes=10)


@pytest.fixture
def rl_agent():
    """Create RLAgent instance for tests"""
    return RLAgent(observation_space_size=50, action_space_size=10)


@pytest.fixture
def dems_environment():
    """Create DEMSEnvironment instance for tests"""
    return DEMSEnvironment(num_nodes=10, max_steps=100)
