"""
Tests for EpisodeLogger metrics module.
"""

import json
import os
import tempfile
import numpy as np
import pytest

from src.agent.metrics import EpisodeLogger


class TestEpisodeLogger:
    """Tests for EpisodeLogger."""

    def test_creation(self):
        logger = EpisodeLogger()
        assert logger is not None

    def test_start_episode(self):
        logger = EpisodeLogger()
        logger.start_episode()
        assert logger._episode_count == 1

    def test_log_step(self):
        logger = EpisodeLogger()
        logger.start_episode()
        logger.log_step(
            agent_name="central",
            reward=0.85,
            violations=0,
            action=np.zeros(6),
        )
        assert "central" in logger._current
        assert len(logger._current["central"].rewards) == 1

    def test_end_episode(self):
        logger = EpisodeLogger()
        logger.start_episode()
        logger.log_step(agent_name="central", reward=0.8, violations=0)
        logger.log_step(agent_name="central", reward=0.9, violations=0)
        summary = logger.end_episode()
        assert "episode" in summary
        assert "agents" in summary
        assert "central" in summary["agents"]
        assert summary["agents"]["central"]["total_reward"] == pytest.approx(1.7)

    def test_disk_logging(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = EpisodeLogger(log_dir=tmpdir)
            logger.start_episode()
            logger.log_step(agent_name="mg_0", reward=0.75, violations=1)
            summary = logger.end_episode()
            # Check that at least one file was created in the log dir
            files = os.listdir(tmpdir)
            assert len(files) >= 1
