"""
Tests for reward function module.
"""

import numpy as np
import pytest

from src.agent.rewards import (
    MicrogridReward,
    CentralReward,
    InverterSubReward,
    RenewableSubReward,
    LoadSubReward,
)
from src.agent.rewards.reward_functions import RewardWeights
from src.agent.constraints.grid_constraints import ConstraintReport, ConstraintViolation, ViolationType


def _clean_report():
    return ConstraintReport()


def _dirty_report(n=3, severity=2.0):
    return ConstraintReport(violations=[
        ConstraintViolation(ViolationType.FREQUENCY_LOW, severity / n, "sys", 49.3, 49.5)
        for _ in range(n)
    ])


class TestRewardWeights:
    """Test default weight configuration."""

    def test_mg_weights_sum_to_one(self):
        w = RewardWeights()
        total = w.mg_voltage + w.mg_economy + w.mg_der_util + w.mg_constraint + w.mg_smoothness
        assert total == pytest.approx(1.0)

    def test_custom_weights(self):
        w = RewardWeights(mg_voltage=0.5, mg_economy=0.5, mg_der_util=0.0,
                          mg_constraint=0.0, mg_smoothness=0.0)
        assert w.mg_voltage == 0.5


class TestMicrogridReward:
    """Tests for MicrogridReward."""

    def setup_method(self):
        self.reward_fn = MicrogridReward()

    def _good_area_state(self):
        return {
            "min_voltage_pu": 1.0, "max_voltage_pu": 1.0, "avg_voltage_pu": 1.0,
            "total_generation_mw": 300.0, "total_load_mw": 280.0,
        }

    def _good_der_status(self):
        return {
            "solar": {"current_output_mw": 50.0},
            "wind": {"current_output_mw": 80.0},
        }

    def test_perfect_state_high_reward(self):
        """Perfect voltage, no violations -> high reward."""
        r, breakdown = self.reward_fn.compute(
            area_state=self._good_area_state(),
            constraint_report=_clean_report(),
            action=np.array([0.5, 0.5, 0.5]),
            der_status=self._good_der_status(),
        )
        assert r > 0.5, f"Expected high reward for perfect state, got {r}"

    def test_violation_penalty(self):
        """Constraint violations should lower the reward."""
        self.reward_fn.reset()
        r_clean, _ = self.reward_fn.compute(
            area_state=self._good_area_state(),
            constraint_report=_clean_report(),
            action=np.array([0.5]),
            der_status=self._good_der_status(),
        )
        self.reward_fn.reset()
        r_dirty, _ = self.reward_fn.compute(
            area_state=self._good_area_state(),
            constraint_report=_dirty_report(5, 3.0),
            action=np.array([0.5]),
            der_status=self._good_der_status(),
        )
        assert r_dirty < r_clean

    def test_return_type(self):
        r, breakdown = self.reward_fn.compute(
            area_state=self._good_area_state(),
            constraint_report=_clean_report(),
            action=np.array([0.5]),
            der_status=self._good_der_status(),
        )
        assert isinstance(r, float)
        assert isinstance(breakdown, dict)


class TestCentralReward:
    """Tests for CentralReward."""

    def setup_method(self):
        self.reward_fn = CentralReward()

    def test_nominal_frequency_high_reward(self):
        r, _ = self.reward_fn.compute(
            frequency_hz=50.0,
            tie_line_flows=[],
            protection_trips=0,
            constraint_report=_clean_report(),
            action=np.zeros(6),
        )
        assert r > 0.5

    def test_frequency_deviation_lowers_reward(self):
        self.reward_fn.reset()
        r_good, _ = self.reward_fn.compute(
            frequency_hz=50.0,
            tie_line_flows=[],
            protection_trips=0,
            constraint_report=_clean_report(),
            action=np.zeros(6),
        )
        self.reward_fn.reset()
        r_bad, _ = self.reward_fn.compute(
            frequency_hz=49.0,
            tie_line_flows=[],
            protection_trips=0,
            constraint_report=_clean_report(),
            action=np.zeros(6),
        )
        assert r_bad < r_good


class TestInverterSubReward:
    """Tests for InverterSubReward."""

    def test_good_voltage_high_reward(self):
        r_fn = InverterSubReward()
        reward, _ = r_fn.compute(
            v_local_pu=1.0,
            p_actual_mw=100.0,
            p_target_mw=100.0,
            constraint_report=_clean_report(),
        )
        assert reward > 0.5

    def test_voltage_deviation_penalty(self):
        r_fn = InverterSubReward()
        r1, _ = r_fn.compute(v_local_pu=1.0, p_actual_mw=100.0, p_target_mw=100.0,
                              constraint_report=_clean_report())
        r2, _ = r_fn.compute(v_local_pu=0.90, p_actual_mw=100.0, p_target_mw=100.0,
                              constraint_report=_clean_report())
        assert r2 < r1


class TestRenewableSubReward:
    """Tests for RenewableSubReward."""

    def test_full_output_high_reward(self):
        r_fn = RenewableSubReward()
        reward, _ = r_fn.compute(
            output_mw=100.0, rated_mw=100.0, curtailment_frac=0.0, v_local_pu=1.0,
        )
        assert reward > 0.5

    def test_curtailment_lowers_reward(self):
        r_fn = RenewableSubReward()
        r1, _ = r_fn.compute(output_mw=100.0, rated_mw=100.0, curtailment_frac=0.0, v_local_pu=1.0)
        r2, _ = r_fn.compute(output_mw=50.0, rated_mw=100.0, curtailment_frac=0.5, v_local_pu=1.0)
        assert r2 < r1


class TestLoadSubReward:
    """Tests for LoadSubReward."""

    def test_full_service_high_reward(self):
        r_fn = LoadSubReward()
        reward, _ = r_fn.compute(
            dr_curtailment_frac=0.0, ev_utilisation_frac=1.0,
            v_local_pu=1.0, frequency_hz=50.0,
        )
        assert reward > 0.5

    def test_load_shed_penalty(self):
        r_fn = LoadSubReward()
        r1, _ = r_fn.compute(dr_curtailment_frac=0.0, ev_utilisation_frac=1.0,
                              v_local_pu=1.0, frequency_hz=50.0)
        r2, _ = r_fn.compute(dr_curtailment_frac=0.8, ev_utilisation_frac=0.2,
                              v_local_pu=1.0, frequency_hz=50.0)
        assert r2 < r1
