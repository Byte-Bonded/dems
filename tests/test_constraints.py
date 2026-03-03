"""
Tests for grid constraint validation module.
"""

import numpy as np
import pytest

from src.agent.constraints import (
    GridConstraintValidator,
    ConstraintViolation,
    ViolationType,
    IEGC_FREQ_MIN_HZ,
    IEGC_FREQ_MAX_HZ,
    IEEE_V_MIN_PU,
    IEEE_V_MAX_PU,
    LINE_LOADING_WARNING_PCT,
    LINE_LOADING_CRITICAL_PCT,
)
from src.agent.constraints.grid_constraints import ConstraintReport


class TestGridConstraintValidator:
    """Tests for GridConstraintValidator."""

    def setup_method(self):
        self.validator = GridConstraintValidator()

    def test_no_violations_clean_state(self):
        """Clean state: no frequency, voltage, or thermal violations."""
        report = self.validator.validate(
            frequency_hz=50.0,
            bus_voltages_pu=np.ones(117),
            line_loading_pct=np.zeros(130),
        )
        assert report.total_violations == 0
        assert report.total_severity == 0.0
        assert not report.has_frequency_violation
        assert not report.has_voltage_violation
        assert not report.has_thermal_violation

    def test_under_frequency_violation(self):
        """Detect under-frequency violation below 49.5 Hz."""
        report = self.validator.validate(
            frequency_hz=49.3,
            bus_voltages_pu=np.ones(117),
            line_loading_pct=np.zeros(130),
        )
        assert report.has_frequency_violation
        assert report.count_by_type(ViolationType.FREQUENCY_LOW) == 1
        v = report.violations[0]
        assert v.value == 49.3
        assert v.limit == IEGC_FREQ_MIN_HZ
        assert v.severity > 0

    def test_over_frequency_violation(self):
        """Detect over-frequency violation above 50.5 Hz."""
        report = self.validator.validate(
            frequency_hz=50.8,
            bus_voltages_pu=np.ones(117),
            line_loading_pct=np.zeros(130),
        )
        assert report.has_frequency_violation
        assert report.count_by_type(ViolationType.FREQUENCY_HIGH) == 1

    def test_voltage_low_violation(self):
        """Detect under-voltage violation below 0.95 pu."""
        bus_v = np.ones(117)
        bus_v[10] = 0.92
        bus_v[50] = 0.94
        report = self.validator.validate(
            frequency_hz=50.0,
            bus_voltages_pu=bus_v,
            line_loading_pct=np.zeros(130),
        )
        assert report.has_voltage_violation
        assert report.count_by_type(ViolationType.VOLTAGE_LOW) == 2

    def test_voltage_high_violation(self):
        """Detect over-voltage violation above 1.05 pu."""
        bus_v = np.ones(117)
        bus_v[5] = 1.08
        report = self.validator.validate(
            frequency_hz=50.0,
            bus_voltages_pu=bus_v,
            line_loading_pct=np.zeros(130),
        )
        assert report.count_by_type(ViolationType.VOLTAGE_HIGH) == 1

    def test_line_overload_violation(self):
        """Detect line thermal overload above 100%."""
        line_loading = np.zeros(130)
        line_loading[0] = 105.0
        line_loading[10] = 120.0
        report = self.validator.validate(
            frequency_hz=50.0,
            bus_voltages_pu=np.ones(117),
            line_loading_pct=line_loading,
        )
        assert report.has_thermal_violation
        assert report.count_by_type(ViolationType.LINE_OVERLOAD) == 2

    def test_generator_limit_violation(self):
        """Detect generator MW limit violation."""
        gen_p = {0: 600.0, 1: 100.0}
        gen_limits = {0: (50.0, 500.0), 1: (20.0, 300.0)}
        report = self.validator.validate(
            frequency_hz=50.0,
            bus_voltages_pu=np.ones(117),
            line_loading_pct=np.zeros(130),
            gen_p_mw=gen_p,
            gen_limits=gen_limits,
        )
        assert report.count_by_type(ViolationType.GEN_LIMIT) == 1

    def test_generator_ramp_violation(self):
        """Detect generator ramp-rate violation."""
        gen_limits = {0: (50.0, 500.0)}
        # First step: set baseline
        self.validator.validate(
            frequency_hz=50.0,
            bus_voltages_pu=np.ones(117),
            line_loading_pct=np.zeros(130),
            gen_p_mw={0: 200.0},
            gen_limits=gen_limits,
        )
        # Second step: large ramp
        report = self.validator.validate(
            frequency_hz=50.0,
            bus_voltages_pu=np.ones(117),
            line_loading_pct=np.zeros(130),
            gen_p_mw={0: 400.0},  # Δ200 MW, max = 0.1 × 500 = 50 MW
            gen_limits=gen_limits,
        )
        assert report.count_by_type(ViolationType.GEN_RAMP) == 1

    def test_frequency_normalised(self):
        """Test frequency deviation normalisation."""
        assert GridConstraintValidator.frequency_deviation_normalised(50.0) == pytest.approx(0.0)
        assert GridConstraintValidator.frequency_deviation_normalised(50.5) == pytest.approx(1.0)
        assert GridConstraintValidator.frequency_deviation_normalised(49.5) == pytest.approx(-1.0)

    def test_reset_clears_ramp_state(self):
        """Reset should clear previous gen setpoints."""
        gen_limits = {0: (50.0, 500.0)}
        self.validator.validate(
            frequency_hz=50.0,
            bus_voltages_pu=np.ones(117),
            line_loading_pct=np.zeros(130),
            gen_p_mw={0: 200.0},
            gen_limits=gen_limits,
        )
        self.validator.reset()
        # After reset, no ramp violation even with large jump
        report = self.validator.validate(
            frequency_hz=50.0,
            bus_voltages_pu=np.ones(117),
            line_loading_pct=np.zeros(130),
            gen_p_mw={0: 400.0},
            gen_limits=gen_limits,
        )
        assert report.count_by_type(ViolationType.GEN_RAMP) == 0


class TestConstraintReport:
    """Tests for ConstraintReport."""

    def test_empty_report(self):
        report = ConstraintReport()
        assert report.total_violations == 0
        assert report.total_severity == 0.0
        assert not report.has_frequency_violation
        assert not report.has_voltage_violation
        assert not report.has_thermal_violation

    def test_mixed_violations(self):
        report = ConstraintReport(violations=[
            ConstraintViolation(ViolationType.FREQUENCY_LOW, 0.5, "system", 49.3, 49.5),
            ConstraintViolation(ViolationType.VOLTAGE_LOW, 0.3, "bus_10", 0.93, 0.95),
            ConstraintViolation(ViolationType.LINE_OVERLOAD, 0.8, "line_5", 110.0, 100.0),
        ])
        assert report.total_violations == 3
        assert report.total_severity == pytest.approx(1.6)
        assert report.has_frequency_violation
        assert report.has_voltage_violation
        assert report.has_thermal_violation
