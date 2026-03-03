"""
Tests for Grid Manager / Grid Monitor
"""

import pytest
from src.core import GridManager, GridMonitor, StabilityStatus


class TestGridMonitor:
    """GridMonitor test suite"""

    def test_initialization(self):
        """Test GridMonitor initialization"""
        monitor = GridMonitor()
        assert monitor.metrics_history == []
        assert len(monitor._alerts) == 0

    def test_health_score(self):
        """Test health score is a valid number"""
        monitor = GridMonitor()
        score = monitor.get_health_score()
        assert isinstance(score, float)
        assert 0.0 <= score <= 100.0

    def test_statistics_empty(self):
        """Test get_statistics returns empty dict when no data"""
        monitor = GridMonitor()
        stats = monitor.get_statistics()
        assert isinstance(stats, dict)

    def test_recent_alerts(self):
        """Test get_recent_alerts"""
        monitor = GridMonitor()
        alerts = monitor.get_recent_alerts(limit=5)
        assert isinstance(alerts, list)


class TestGridManagerCompat:
    """Backward-compatible GridManager test suite"""

    def test_initialization(self):
        """Test legacy GridManager initialization"""
        gm = GridManager(num_nodes=10)
        assert gm.num_nodes == 10

    def test_get_grid_state(self):
        """Test get_grid_state includes num_nodes"""
        gm = GridManager(num_nodes=10)
        state = gm.get_grid_state()
        assert isinstance(state, dict)
        assert "num_nodes" in state
        assert state["num_nodes"] == 10
