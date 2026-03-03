"""
Grid constraint validation module.

Validates physical and regulatory constraints for the Indian grid
(IEGC 50 Hz standards) and IEEE bus voltage standards.

Used by reward functions and observation builders to detect violations.
"""

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

# ─── IEGC / IEEE Constants ──────────────────────────────────────────
IEGC_FREQ_MIN_HZ = 49.5
IEGC_FREQ_MAX_HZ = 50.5
IEGC_FREQ_NOMINAL_HZ = 50.0

IEEE_V_MIN_PU = 0.95
IEEE_V_MAX_PU = 1.05

LINE_LOADING_WARNING_PCT = 80.0
LINE_LOADING_CRITICAL_PCT = 100.0

# Generator ramp-rate default (fraction of Pmax per step)
DEFAULT_GEN_RAMP_FRAC = 0.10

# UFLS thresholds (Hz)
UFLS_STAGE_1_HZ = 49.0
UFLS_STAGE_2_HZ = 48.5
UFLS_STAGE_3_HZ = 48.0


class ViolationType(Enum):
    """Categories of constraint violation."""
    FREQUENCY_LOW = "frequency_low"
    FREQUENCY_HIGH = "frequency_high"
    VOLTAGE_LOW = "voltage_low"
    VOLTAGE_HIGH = "voltage_high"
    LINE_OVERLOAD = "line_overload"
    GEN_RAMP = "gen_ramp"
    GEN_LIMIT = "gen_limit"
    DER_LIMIT = "der_limit"


@dataclass
class ConstraintViolation:
    """A single constraint violation event."""
    violation_type: ViolationType
    severity: float  # 0..1 normalised severity
    element_id: str  # bus/line/gen identifier
    value: float     # the actual value
    limit: float     # the violated limit
    details: str = ""


@dataclass
class ConstraintReport:
    """Aggregated violation report for one time step."""
    violations: List[ConstraintViolation] = field(default_factory=list)

    @property
    def total_violations(self) -> int:
        return len(self.violations)

    @property
    def total_severity(self) -> float:
        return sum(v.severity for v in self.violations)

    @property
    def has_frequency_violation(self) -> bool:
        return any(
            v.violation_type in (ViolationType.FREQUENCY_LOW, ViolationType.FREQUENCY_HIGH)
            for v in self.violations
        )

    @property
    def has_voltage_violation(self) -> bool:
        return any(
            v.violation_type in (ViolationType.VOLTAGE_LOW, ViolationType.VOLTAGE_HIGH)
            for v in self.violations
        )

    @property
    def has_thermal_violation(self) -> bool:
        return any(
            v.violation_type == ViolationType.LINE_OVERLOAD
            for v in self.violations
        )

    def count_by_type(self, vtype: ViolationType) -> int:
        return sum(1 for v in self.violations if v.violation_type == vtype)


class GridConstraintValidator:
    """
    Validates all grid constraints and produces a ConstraintReport.

    Constraint categories:
    1. Frequency: IEGC 49.5–50.5 Hz
    2. Voltage: IEEE 0.95–1.05 pu per bus
    3. Thermal: Line loading ≤ 100% (warning at 80%)
    4. Generator: Pmin ≤ P ≤ Pmax, ramp rate
    5. DER: operating envelopes
    """

    def __init__(
        self,
        f_min: float = IEGC_FREQ_MIN_HZ,
        f_max: float = IEGC_FREQ_MAX_HZ,
        v_min: float = IEEE_V_MIN_PU,
        v_max: float = IEEE_V_MAX_PU,
        line_warn_pct: float = LINE_LOADING_WARNING_PCT,
        line_crit_pct: float = LINE_LOADING_CRITICAL_PCT,
        gen_ramp_frac: float = DEFAULT_GEN_RAMP_FRAC,
    ):
        self.f_min = f_min
        self.f_max = f_max
        self.v_min = v_min
        self.v_max = v_max
        self.line_warn_pct = line_warn_pct
        self.line_crit_pct = line_crit_pct
        self.gen_ramp_frac = gen_ramp_frac

        # Track previous gen setpoints for ramp-rate validation
        self._prev_gen_p: Optional[Dict[int, float]] = None

    def reset(self) -> None:
        """Reset tracked state (call at episode start)."""
        self._prev_gen_p = None

    def validate(
        self,
        frequency_hz: float,
        bus_voltages_pu: np.ndarray,
        line_loading_pct: np.ndarray,
        gen_p_mw: Optional[Dict[int, float]] = None,
        gen_limits: Optional[Dict[int, Tuple[float, float]]] = None,
    ) -> ConstraintReport:
        """
        Run all constraint checks for the current step.

        Args:
            frequency_hz: System frequency (Hz).
            bus_voltages_pu: Bus voltage magnitudes (1-D array).
            line_loading_pct: Line loading percentages (1-D array).
            gen_p_mw: {gen_idx: P_mw} current generator outputs.
            gen_limits: {gen_idx: (Pmin, Pmax)} generator operating limits.

        Returns:
            ConstraintReport with all detected violations.
        """
        report = ConstraintReport()

        # 1. Frequency constraints
        self._check_frequency(frequency_hz, report)

        # 2. Voltage constraints
        self._check_voltages(bus_voltages_pu, report)

        # 3. Thermal constraints
        self._check_thermal(line_loading_pct, report)

        # 4. Generator constraints
        if gen_p_mw and gen_limits:
            self._check_generators(gen_p_mw, gen_limits, report)

        # Update previous gen setpoints
        if gen_p_mw is not None:
            self._prev_gen_p = dict(gen_p_mw)

        return report

    # ─── individual checks ─────────────────────────────────────────

    def _check_frequency(self, f_hz: float, report: ConstraintReport) -> None:
        """Check IEGC frequency band."""
        if f_hz < self.f_min:
            dev = self.f_min - f_hz
            max_dev = self.f_min - UFLS_STAGE_3_HZ  # 1.5 Hz range → severity 1
            severity = min(dev / max(max_dev, 0.01), 1.0)
            report.violations.append(ConstraintViolation(
                violation_type=ViolationType.FREQUENCY_LOW,
                severity=severity,
                element_id="system",
                value=f_hz,
                limit=self.f_min,
                details=f"Under-frequency: {f_hz:.3f} Hz < {self.f_min} Hz",
            ))
        elif f_hz > self.f_max:
            dev = f_hz - self.f_max
            severity = min(dev / 1.5, 1.0)
            report.violations.append(ConstraintViolation(
                violation_type=ViolationType.FREQUENCY_HIGH,
                severity=severity,
                element_id="system",
                value=f_hz,
                limit=self.f_max,
                details=f"Over-frequency: {f_hz:.3f} Hz > {self.f_max} Hz",
            ))

    def _check_voltages(self, v_pu: np.ndarray, report: ConstraintReport) -> None:
        """Check IEEE voltage limits per bus."""
        for bus_idx in range(len(v_pu)):
            v = float(v_pu[bus_idx])
            if v < self.v_min:
                dev = self.v_min - v
                severity = min(dev / 0.1, 1.0)  # 0.1 pu → severity 1
                report.violations.append(ConstraintViolation(
                    violation_type=ViolationType.VOLTAGE_LOW,
                    severity=severity,
                    element_id=f"bus_{bus_idx}",
                    value=v,
                    limit=self.v_min,
                ))
            elif v > self.v_max:
                dev = v - self.v_max
                severity = min(dev / 0.1, 1.0)
                report.violations.append(ConstraintViolation(
                    violation_type=ViolationType.VOLTAGE_HIGH,
                    severity=severity,
                    element_id=f"bus_{bus_idx}",
                    value=v,
                    limit=self.v_max,
                ))

    def _check_thermal(self, loading_pct: np.ndarray, report: ConstraintReport) -> None:
        """Check line thermal loading limits."""
        for line_idx in range(len(loading_pct)):
            load = float(loading_pct[line_idx])
            if load > self.line_crit_pct:
                severity = min((load - self.line_crit_pct) / 50.0, 1.0)
                report.violations.append(ConstraintViolation(
                    violation_type=ViolationType.LINE_OVERLOAD,
                    severity=severity,
                    element_id=f"line_{line_idx}",
                    value=load,
                    limit=self.line_crit_pct,
                    details=f"Line {line_idx} overloaded: {load:.1f}%",
                ))

    def _check_generators(
        self,
        gen_p_mw: Dict[int, float],
        gen_limits: Dict[int, Tuple[float, float]],
        report: ConstraintReport,
    ) -> None:
        """Check generator MW limits and ramp rates."""
        for gen_idx, p_mw in gen_p_mw.items():
            if gen_idx in gen_limits:
                p_min, p_max = gen_limits[gen_idx]
                if p_mw < p_min - 0.1:
                    severity = min((p_min - p_mw) / max(p_max, 1), 1.0)
                    report.violations.append(ConstraintViolation(
                        violation_type=ViolationType.GEN_LIMIT,
                        severity=severity,
                        element_id=f"gen_{gen_idx}",
                        value=p_mw,
                        limit=p_min,
                        details=f"Gen {gen_idx} below Pmin: {p_mw:.1f} < {p_min:.1f} MW",
                    ))
                elif p_mw > p_max + 0.1:
                    severity = min((p_mw - p_max) / max(p_max, 1), 1.0)
                    report.violations.append(ConstraintViolation(
                        violation_type=ViolationType.GEN_LIMIT,
                        severity=severity,
                        element_id=f"gen_{gen_idx}",
                        value=p_mw,
                        limit=p_max,
                        details=f"Gen {gen_idx} above Pmax: {p_mw:.1f} > {p_max:.1f} MW",
                    ))

            # Ramp-rate check
            if self._prev_gen_p is not None and gen_idx in self._prev_gen_p:
                prev_p = self._prev_gen_p[gen_idx]
                p_max_for_ramp = gen_limits.get(gen_idx, (0, 500))[1]
                max_delta = self.gen_ramp_frac * p_max_for_ramp
                delta = abs(p_mw - prev_p)
                if delta > max_delta + 0.1:
                    severity = min((delta - max_delta) / max(max_delta, 1), 1.0)
                    report.violations.append(ConstraintViolation(
                        violation_type=ViolationType.GEN_RAMP,
                        severity=severity,
                        element_id=f"gen_{gen_idx}",
                        value=delta,
                        limit=max_delta,
                        details=f"Gen {gen_idx} ramp violation: Δ{delta:.1f} > {max_delta:.1f} MW",
                    ))

    # ─── convenience query methods ─────────────────────────────────

    @staticmethod
    def frequency_deviation_normalised(f_hz: float) -> float:
        """Return frequency deviation normalised to [-1, 1] within IEGC band."""
        f_dev = f_hz - IEGC_FREQ_NOMINAL_HZ
        half_band = (IEGC_FREQ_MAX_HZ - IEGC_FREQ_MIN_HZ) / 2.0
        return np.clip(f_dev / half_band, -2.0, 2.0)

    @staticmethod
    def voltage_deviation_normalised(v_pu: float) -> float:
        """Return voltage deviation normalised to [-1, 1] within IEEE band."""
        v_mid = (IEEE_V_MAX_PU + IEEE_V_MIN_PU) / 2.0
        half_band = (IEEE_V_MAX_PU - IEEE_V_MIN_PU) / 2.0
        return np.clip((v_pu - v_mid) / half_band, -2.0, 2.0)
