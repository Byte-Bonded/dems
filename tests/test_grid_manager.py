"""
Tests for Grid Manager
"""

import pytest
from src.core import GridManager, GridNode


class TestGridManager:
    """Grid Manager test suite"""

    def test_initialization(self, grid_manager):
        """Test grid manager initialization"""
        assert grid_manager.num_nodes == 10
        assert len(grid_manager.nodes) == 10

    def test_node_initialization(self, grid_manager):
        """Test that nodes are properly initialized"""
        for node_id, node in grid_manager.nodes.items():
            assert node.node_id == node_id
            assert node.voltage == 230.0
            assert node.frequency == 50.0
            assert node.capacity == 100.0

    def test_get_grid_state(self, grid_manager):
        """Test getting grid state"""
        state = grid_manager.get_grid_state()
        assert "num_nodes" in state
        assert "total_load" in state
        assert "total_capacity" in state
        assert "utilization" in state
        assert state["num_nodes"] == 10

    def test_update_node_load(self, grid_manager):
        """Test updating node load"""
        success = grid_manager.update_node_load(0, 50.0)
        assert success is True
        assert grid_manager.nodes[0].load == 50.0

    def test_update_invalid_node(self, grid_manager):
        """Test updating invalid node"""
        success = grid_manager.update_node_load(999, 50.0)
        assert success is False

    def test_check_stability(self, grid_manager):
        """Test stability check"""
        is_stable = grid_manager.check_stability()
        assert isinstance(is_stable, bool)
        assert is_stable is True  # Default frequency is 50.0
