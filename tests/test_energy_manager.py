"""
Tests for Energy Manager
"""

import pytest
from src.core import EnergyManager


class TestEnergyManager:
    """Energy Manager test suite"""

    def test_initialization(self, energy_manager):
        """Test energy manager initialization"""
        assert energy_manager.grid_size == 10
        assert energy_manager.storage_capacity == 1000.0
        assert energy_manager.current_storage == 500.0

    def test_get_current_state(self, energy_manager):
        """Test getting current state"""
        state = energy_manager.get_current_state()
        assert "timestamp" in state
        assert "storage_level" in state
        assert "storage_capacity" in state
        assert state["storage_level"] == 500.0

    def test_optimize_distribution(self, energy_manager):
        """Test optimization distribution"""
        result = energy_manager.optimize_distribution()
        assert "status" in result
        assert result["status"] == "pending"

    def test_repr(self, energy_manager):
        """Test string representation"""
        repr_str = repr(energy_manager)
        assert "EnergyManager" in repr_str
        assert "10" in repr_str
