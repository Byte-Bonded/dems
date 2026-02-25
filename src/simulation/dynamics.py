"""
Dynamic Power System Models for DEMS SuperGrid
Based on IEEE standards for 50 Hz operation

This module provides:
- Synchronous generator swing equation dynamics
- IEEE Type 1 excitation system (AVR) - IEEE Std 421.5-2016 sec5.1
- IEEE TGOV1 governor-turbine model - NERC PPMV compliant
- Power System Stabilizer (PSS1A) - IEEE Std 421.5-2016 sec8.1
- Automatic Generation Control (AGC) - IEGC compliant
- Frequency and voltage dependent load models (ZIP)
- Protection systems (Indian Grid Code)

All models designed for 50 Hz Indian Grid Code operation.
Integration: Implicit Trapezoidal (unconditionally stable).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# ======================== HELPER ======================== #

def _trap_first_order(x_old: float, u_new: float, T: float, dt: float) -> float:
    """Implicit trapezoidal integration for dx/dt = (u - x) / T.
    Unconditionally stable for any dt > 0 and T > 0.
    Returns x_new."""
    if T <= 0:
        return u_new
    denom = 2.0 * T + dt
    return (x_old * (2.0 * T - dt) + 2.0 * u_new * dt) / denom


# ======================== PARAMETER CLASSES ======================== #

@dataclass
class GeneratorDynamicParams:
    """Generator dynamic parameters for 50 Hz operation"""
    H: float = 3.5
    D: float = 2.0
    MVA_base: float = 100.0
    frequency: float = 50.0
    Xd_prime: float = 0.3
    Xq_prime: float = 0.55
    Td0_prime: float = 8.0
    Tq0_prime: float = 0.4


@dataclass
class ExciterParams:
    """IEEE Type 1 excitation system parameters (IEEET1)
    Reference: IEEE Std 421.5-2016, sec5.1, Figure 5-1"""
    KA: float = 200.0
    TA: float = 0.02
    TR: float = 0.02
    KE: float = 1.0
    TE: float = 0.5
    KF: float = 0.03            # Stabilizing feedback gain (IEEE-02 fix)
    TF: float = 1.0             # Stabilizing feedback time constant (IEEE-02 fix)
    VRMAX: float = 5.0
    VRMIN: float = -5.0
    EMIN: float = 0.0
    EMAX: float = 5.0
    Vref: float = 1.0
    SE_A: float = 0.0039        # Saturation coefficient A (IEEE-01 fix)
    SE_B: float = 1.555         # Saturation coefficient B (IEEE-01 fix)
    # FIX BUG-05: Over-Excitation Limiter (OEL) parameters per IEEE 421.5-2016 §6
    Efd_max_thermal: float = 3.0  # Max sustained Efd (thermal limit)
    OEL_delay_s: float = 10.0     # Time before OEL activates (10-60s typical)
    OEL_gain: float = 10.0        # OEL integrator gain


@dataclass
class GovernorParams:
    """IEEE TGOV1 governor-turbine parameters"""
    R: float = 0.05
    TG: float = 0.2
    TT: float = 0.5
    Pmax: float = 1.5   # Raised from 1.2 — slack gens can operate at 1.35 pu
    Pmin: float = 0.0
    Pref: float = 1.0
    valve_rate_up: float = 0.1    # pu/s (IEEE-06 fix)
    valve_rate_down: float = -0.1 # pu/s (IEEE-06 fix)
    Dt: float = 0.05              # Turbine damping (FIX BUG-11: non-zero default)


@dataclass
class AGCParams:
    """Automatic Generation Control parameters"""
    beta: float = 1000.0
    K_agc: float = 0.5
    T_agc: float = 4.0
    deadband_hz: float = 0.03   # IEGC standard (IEEE-22 fix, was 0.02)


@dataclass
class ProtectionParams:
    """Protection system parameters (Indian Grid Code IEGC)"""
    voltage_low_warn: float = 0.95
    voltage_low_trip: float = 0.90
    voltage_high_warn: float = 1.05
    voltage_high_trip: float = 1.10
    frequency_low_warn: float = 49.7
    frequency_low_trip: float = 49.5
    frequency_high_warn: float = 50.3
    frequency_high_trip: float = 50.5
    voltage_trip_delay: float = 2.0
    frequency_trip_delay: float = 0.2  # IEGC UFLS ~100-200ms (IEEE-23 fix)
    # FIX BUG-08: Multi-stage UFLS per IEGC
    ufls_stages: List[Tuple[float, float]] = field(default_factory=lambda: [
        (49.5, 0.10),   # Stage 1: 49.5 Hz, shed 10%
        (49.2, 0.15),   # Stage 2: 49.2 Hz, shed 15%
        (49.0, 0.20),   # Stage 3: 49.0 Hz, shed 20%
    ])
    # FIX BUG-23: Over-frequency generation trip (OFGT)
    frequency_high_gen_trip: float = 50.5  # Hz for sustained OFGT
    ofgt_delay: float = 0.5               # seconds before OFGT activates


# ======================== GENERATOR DYNAMIC MODELS ======================== #

class SynchronousGeneratorDynamic:
    """IEEE standard synchronous generator model with swing equation
    Designed for 50 Hz operation (Indian Grid Code)"""

    def __init__(self, gen_id: str, bus: int, params: Optional[GeneratorDynamicParams] = None):
        self.gen_id = gen_id
        self.bus = bus
        self.params = params or GeneratorDynamicParams()
        self.omega_base = 2 * np.pi * self.params.frequency
        self.delta = 0.0
        self.omega = 1.0
        self.Eq_prime = 1.0
        self.Ed_prime = 0.0
        self.P_mech = 0.0
        self.P_elec = 0.0
        self.Q_elec = 0.0
        self.Vt = 1.0
        self.exciter: Optional[ExcitationSystem] = None
        self.governor: Optional[GovernorTurbine] = None
        self.pss: Optional[PowerSystemStabilizer] = None

    def initialize(self, P_mw: float, Q_mvar: float, Vt: float, delta_deg: float):
        """Initialize from PF with proper phasor calculation (FIX I01).
        Uses |Vt*exp(j*delta) + jXd_prime*I| instead of scalar formula."""
        self.P_elec = P_mw
        self.Q_elec = Q_mvar
        self.P_mech = P_mw
        self.Vt = Vt
        self.delta = np.radians(delta_deg)
        self.omega = 1.0
        P_pu = P_mw / self.params.MVA_base
        Q_pu = Q_mvar / self.params.MVA_base
        Vt_complex = Vt * np.exp(1j * self.delta)
        if Vt > 1e-6:
            I_complex = (P_pu - 1j * Q_pu) / np.conj(Vt_complex)
        else:
            I_complex = 0.0 + 0.0j
        E_prime = Vt_complex + 1j * self.params.Xd_prime * I_complex
        self.Eq_prime = abs(E_prime)
        logger.debug(f"Gen {self.gen_id} initialized: P={P_mw:.1f} MW, Eq'={self.Eq_prime:.4f}")

    def update(self, dt: float, Vt: float, P_elec_mw: float,
               Efd: Optional[float] = None) -> Tuple[float, float]:
        """Advance swing equation + field flux one step.
        
        FIX NEW-BUG-05: When Efd is provided (from exciter), update Eq'
        via the field-circuit equation dEq'/dt = (Efd - Eq') / Td0'.
        This couples the AVR/PSS output back into the generator model."""
        self.Vt = Vt
        self.P_elec = P_elec_mw
        P_mech_pu = self.P_mech / self.params.MVA_base
        P_elec_pu = P_elec_mw / self.params.MVA_base
        delta_P = P_mech_pu - P_elec_pu
        d_omega = (1.0 / (2 * self.params.H)) * (
            delta_P - self.params.D * (self.omega - 1.0)
        ) * dt
        # FIX BUG-04: Symplectic Euler — update omega FIRST, then use NEW omega for delta
        self.omega += d_omega
        # Safety clamp: prevent numerical runaway (45-55 Hz hard limits)
        self.omega = np.clip(self.omega, 0.90, 1.10)
        d_delta = self.omega_base * (self.omega - 1.0) * dt
        self.delta += d_delta
        # FIX NEW-BUG-05: Field flux dynamics dEq'/dt = (Efd - Eq') / Td0'
        if Efd is not None and self.params.Td0_prime > 0:
            self.Eq_prime = _trap_first_order(
                self.Eq_prime, Efd, self.params.Td0_prime, dt)
        frequency_hz = self.omega * self.params.frequency
        return frequency_hz, np.degrees(self.delta)

    def set_mechanical_power(self, P_mech_mw: float):
        self.P_mech = P_mech_mw

    def get_frequency_hz(self) -> float:
        return self.omega * self.params.frequency

    def get_speed_deviation_pu(self) -> float:
        return self.omega - 1.0


class ExcitationSystem:
    """IEEE Type 1 Excitation System (IEEET1)
    Reference: IEEE Std 421.5-2016, sec5.1, Figure 5-1

    Fixes: IEEE-01 (SE saturation), IEEE-02 (KF feedback),
           IEEE-03 (exciter TF 1/(KE+sTE)), IEEE-04/05 (PF init),
           N01/N02 (implicit trapezoidal)"""

    def __init__(self, params: Optional[ExciterParams] = None):
        self.params = params or ExciterParams()
        self.Vr = 1.0
        self.Efd = 1.0
        self.Vt_filtered = 1.0
        self.Vref = self.params.Vref
        self.Vf = 0.0
        self._Efd_prev = 1.0
        # FIX BUG-05: OEL state
        self._oel_timer = 0.0
        self._oel_active = False
        self._oel_vref_reduction = 0.0

    def initialize_from_pf(self, Efd0: float, Vt0: float):
        """Initialize from PF steady state (FIX IEEE-04/05).
        FIX NEW-BUG-02: Vf and Vref must be set to SS values so the
        regulator does not produce a transient on the first step."""
        self.Efd = Efd0
        self._Efd_prev = Efd0
        self.Vt_filtered = Vt0
        SE = self._saturation(Efd0)
        self.Vr = (self.params.KE + SE) * Efd0
        # FIX NEW-BUG-02a: Vf at SS equals the filtered Efd derivative signal
        self.Vf = (self.params.KF / self.params.TF) * Efd0 if self.params.TF > 0 else 0.0
        # FIX NEW-BUG-02b: Vref must produce zero error at SS
        #   At SS: Ve = Vref - Vt + 0 - 0 = Vr / KA   (Vfb_ss = 0)
        self.Vref = Vt0 + self.Vr / self.params.KA if self.params.KA > 0 else Vt0

    def _saturation(self, Efd: float) -> float:
        """SE(Efd) = A * exp(B * |Efd|) (FIX IEEE-01)"""
        return self.params.SE_A * np.exp(self.params.SE_B * abs(Efd))

    def update(self, dt: float, Vt: float, Vref: Optional[float] = None,
               Vpss: float = 0.0) -> float:
        """Update IEEET1 with implicit trapezoidal (FIX N01/N02)."""
        if Vref is not None:
            self.Vref = Vref
        p = self.params
        # 1. Voltage transducer
        self.Vt_filtered = _trap_first_order(self.Vt_filtered, Vt, p.TR, dt)
        # 2. KF feedback (FIX IEEE-02): KF*s/(1+sTF)
        Vf_input = (p.KF / p.TF) * self.Efd
        self.Vf = _trap_first_order(self.Vf, Vf_input, p.TF, dt)
        Vfb = Vf_input - self.Vf
        # 3. Voltage error
        Ve = self.Vref - self.Vt_filtered + Vpss - Vfb
        # Apply OEL Vref reduction if limiter is active (FIX BUG-05)
        Ve -= self._oel_vref_reduction
        # 4. Regulator: KA/(1+sTA)
        Vr_target = p.KA * Ve
        self.Vr = _trap_first_order(self.Vr, Vr_target, p.TA, dt)
        self.Vr = np.clip(self.Vr, p.VRMIN, p.VRMAX)
        # 5. Exciter: 1/(KE+sTE) with saturation (FIX IEEE-01, IEEE-03)
        SE = self._saturation(self.Efd)
        KE_eff = max(p.KE + SE, 1e-6)
        Efd_target = self.Vr / KE_eff
        self.Efd = _trap_first_order(self.Efd, Efd_target, p.TE / KE_eff, dt)
        self.Efd = np.clip(self.Efd, p.EMIN, p.EMAX)
        # FIX BUG-05: Over-Excitation Limiter (OEL)
        if abs(self.Efd) > p.Efd_max_thermal:
            self._oel_timer += dt
            if self._oel_timer >= p.OEL_delay_s:
                self._oel_active = True
                # Progressively reduce Vref to pull Efd back within limits
                excess = abs(self.Efd) - p.Efd_max_thermal
                self._oel_vref_reduction += p.OEL_gain * excess * dt
                self._oel_vref_reduction = min(self._oel_vref_reduction, 0.5)
        else:
            self._oel_timer = max(0.0, self._oel_timer - dt)
            if self._oel_timer <= 0:
                self._oel_active = False
                self._oel_vref_reduction = max(0.0, self._oel_vref_reduction - 0.01 * dt)
        return self.Efd


class GovernorTurbine:
    """IEEE TGOV1 Governor-Turbine Model
    Fixes: IEEE-06 (rate limits), IEEE-07 (Dt), IEEE-08 (PF init)"""

    def __init__(self, params: Optional[GovernorParams] = None):
        self.params = params or GovernorParams()
        self.Pg = 1.0
        self.Pm = 1.0
        self.Pref = self.params.Pref
        self.base_Pref = self.params.Pref  # ED setpoint (modified by AGC only)

    def initialize_from_pf(self, Pm0_pu: float):
        """Initialize from PF steady state (FIX IEEE-08)."""
        self.Pg = Pm0_pu
        self.Pm = Pm0_pu
        self.Pref = Pm0_pu
        self.base_Pref = Pm0_pu  # Store ED setpoint for AGC/LFC layering

    def update(self, dt: float, omega: float, Pref: Optional[float] = None) -> float:
        """Advance TGOV1 with trapezoidal + rate limiting."""
        if Pref is not None:
            self.Pref = Pref
        p = self.params
        delta_omega = omega - 1.0
        Pg_ref = self.Pref - delta_omega / p.R
        Pg_new = _trap_first_order(self.Pg, Pg_ref, p.TG, dt)
        # Valve rate limiting (FIX IEEE-06)
        dPg = Pg_new - self.Pg
        max_up = p.valve_rate_up * dt
        max_down = p.valve_rate_down * dt
        dPg = np.clip(dPg, max_down, max_up)
        self.Pg += dPg
        self.Pg = np.clip(self.Pg, p.Pmin, p.Pmax)
        # Turbine dynamics
        self.Pm = _trap_first_order(self.Pm, self.Pg, p.TT, dt)
        # Turbine damping (FIX IEEE-07)
        Pm_out = self.Pm - p.Dt * delta_omega
        return Pm_out


class PowerSystemStabilizer:
    """Power System Stabilizer (PSS1A - single speed input)
    Reference: IEEE Std 421.5-2016, sec8.1
    Fixes: IEEE-10 (PSS1A label), IEEE-12 (DC gain=1), IEEE-13 (2 stages),
           IEEE-14 (gain reduced), N03 (signal path)"""

    def __init__(self):
        self.T_washout = 1.41
        self.T_lead1 = 0.154
        self.T_lag1 = 0.033
        self.T_lead2 = 0.154   # 2nd stage (FIX IEEE-13)
        self.T_lag2 = 0.033    # 2nd stage (FIX IEEE-13)
        self.K_pss = 5.0       # Reduced from 9.5 (FIX IEEE-14)
        self.V_max = 0.2
        self.V_min = -0.2
        self.washout_state = 0.0
        self.lead_lag1_state = 0.0
        self.lead_lag2_state = 0.0
        self.output = 0.0

    def update(self, dt: float, speed_deviation: float) -> float:
        """PSS with washout + 2 lead-lag stages, DC gain = 1.0 each."""
        # Washout: sT/(1+sT)
        self.washout_state = _trap_first_order(
            self.washout_state, speed_deviation, self.T_washout, dt)
        washout_out = speed_deviation - self.washout_state
        pss_signal = self.K_pss * washout_out
        # Lead-lag 1: (1+sT_lead)/(1+sT_lag), DC gain = 1.0 (FIX IEEE-12)
        self.lead_lag1_state = _trap_first_order(
            self.lead_lag1_state, pss_signal, self.T_lag1, dt)
        if self.T_lag1 > 1e-10:
            ll1_out = self.lead_lag1_state + self.T_lead1 * (pss_signal - self.lead_lag1_state) / self.T_lag1
        else:
            ll1_out = pss_signal
        # Lead-lag 2 (FIX IEEE-13)
        self.lead_lag2_state = _trap_first_order(
            self.lead_lag2_state, ll1_out, self.T_lag2, dt)
        if self.T_lag2 > 1e-10:
            ll2_out = self.lead_lag2_state + self.T_lead2 * (ll1_out - self.lead_lag2_state) / self.T_lag2
        else:
            ll2_out = ll1_out
        self.output = np.clip(ll2_out, self.V_min, self.V_max)
        return self.output


# ======================== SYSTEM CONTROL ======================== #

class AutomaticGenerationControl:
    """AGC: ACE = delta_Ptie + beta * delta_f
    FIX C05: Participation factors normalized.
    FIX IEEE-22: Deadband 0.03 Hz (IEGC)."""

    def __init__(self, area_id: str, params: Optional[AGCParams] = None):
        self.area_id = area_id
        self.params = params or AGCParams()
        self.ace = 0.0
        self.agc_output = 0.0
        self.integral_ace = 0.0
        self.participating_gens: List[str] = []
        self.participation_factors: Dict[str, float] = {}
        self._factors_normalized = False

    def add_participating_generator(self, gen_id: str, participation_factor: float):
        self.participating_gens.append(gen_id)
        self.participation_factors[gen_id] = participation_factor
        self._factors_normalized = False

    def _normalize_factors(self):
        """FIX C05: Normalize participation factors to sum to 1.0."""
        total = sum(self.participation_factors.values())
        if total > 0:
            for gid in self.participation_factors:
                self.participation_factors[gid] /= total
        self._factors_normalized = True

    def update(self, dt: float, frequency_hz: float, tie_line_error_mw: float = 0.0) -> Dict[str, float]:
        if not self._factors_normalized:
            self._normalize_factors()
        delta_f = frequency_hz - 50.0
        if abs(delta_f) < self.params.deadband_hz:
            delta_f = 0.0
        self.ace = tie_line_error_mw + self.params.beta * delta_f
        p_term = -self.params.K_agc * self.ace
        self.integral_ace += self.ace * dt
        self.integral_ace = np.clip(self.integral_ace, -1000, 1000)
        i_term = -self.integral_ace / self.params.T_agc
        self.agc_output = p_term + i_term
        adjustments = {}
        for gen_id in self.participating_gens:
            pf = self.participation_factors.get(gen_id, 0.0)
            adjustments[gen_id] = self.agc_output * pf
        return adjustments

    def get_ace(self) -> float:
        return self.ace


# ======================== LOAD MODELS ======================== #

class DynamicLoadModel:
    """ZIP load with frequency dependence."""

    def __init__(self, bus: int, P0_mw: float, Q0_mvar: float):
        self.bus = bus
        self.P0 = P0_mw
        self.Q0 = Q0_mvar
        self.Zp = 0.4; self.Ip = 0.3; self.Pp = 0.3
        self.Zq = 0.5; self.Iq = 0.3; self.Pq = 0.2
        self.Kpf = 1.5; self.Kqf = -1.0
        self.P_mw = P0_mw
        self.Q_mvar = Q0_mvar

    def update(self, voltage_pu: float, frequency_hz: float) -> Tuple[float, float]:
        delta_f = (frequency_hz - 50.0) / 50.0
        V = voltage_pu; V2 = V * V
        p_vf = self.Zp * V2 + self.Ip * V + self.Pp
        q_vf = self.Zq * V2 + self.Iq * V + self.Pq
        self.P_mw = self.P0 * p_vf * (1.0 + self.Kpf * delta_f)
        self.Q_mvar = self.Q0 * q_vf * (1.0 + self.Kqf * delta_f)
        return self.P_mw, self.Q_mvar


# ======================== PROTECTION SYSTEMS ======================== #

class ProtectionRelay:
    """Protection relay - Indian Grid Code (IEGC) compliant."""

    def __init__(self, element_id: str, element_type: str,
                 params: Optional[ProtectionParams] = None):
        self.element_id = element_id
        self.element_type = element_type
        self.params = params or ProtectionParams()
        self.tripped = False
        self.trip_reason: Optional[str] = None
        self.alarm_active = False
        self.alarm_reasons: List[str] = []
        self.undervoltage_timer = 0.0
        self.overvoltage_timer = 0.0
        self.underfrequency_timer = 0.0
        self.overfrequency_timer = 0.0
        # FIX BUG-08: Multi-stage UFLS state
        self.ufls_stages_tripped: List[bool] = [False] * len(self.params.ufls_stages)
        self.ufls_stage_timers: List[float] = [0.0] * len(self.params.ufls_stages)
        self.total_load_shed_fraction: float = 0.0
        # FIX BUG-23: OFGT state
        self.ofgt_timer: float = 0.0
        self.ofgt_tripped: bool = False

    def check(self, dt: float, voltage_pu: float, frequency_hz: float) -> Dict:
        self.alarm_reasons = []
        if voltage_pu < self.params.voltage_low_warn:
            self.alarm_reasons.append(f"Low voltage warning: {voltage_pu:.3f} pu")
        if voltage_pu > self.params.voltage_high_warn:
            self.alarm_reasons.append(f"High voltage warning: {voltage_pu:.3f} pu")
        if frequency_hz < self.params.frequency_low_warn:
            self.alarm_reasons.append(f"Low frequency warning: {frequency_hz:.2f} Hz")
        if frequency_hz > self.params.frequency_high_warn:
            self.alarm_reasons.append(f"High frequency warning: {frequency_hz:.2f} Hz")
        self.alarm_active = len(self.alarm_reasons) > 0
        if voltage_pu < self.params.voltage_low_trip:
            self.undervoltage_timer += dt
            if self.undervoltage_timer >= self.params.voltage_trip_delay:
                self.tripped = True
                self.trip_reason = f"Undervoltage trip: {voltage_pu:.3f} pu"
        else:
            self.undervoltage_timer = 0.0
        if voltage_pu > self.params.voltage_high_trip:
            self.overvoltage_timer += dt
            if self.overvoltage_timer >= self.params.voltage_trip_delay:
                self.tripped = True
                self.trip_reason = f"Overvoltage trip: {voltage_pu:.3f} pu"
        else:
            self.overvoltage_timer = 0.0
        if frequency_hz < self.params.frequency_low_trip:
            self.underfrequency_timer += dt
            if self.underfrequency_timer >= self.params.frequency_trip_delay:
                self.tripped = True
                self.trip_reason = f"Underfrequency trip: {frequency_hz:.2f} Hz"
        else:
            self.underfrequency_timer = 0.0
        if frequency_hz > self.params.frequency_high_trip:
            self.overfrequency_timer += dt
            if self.overfrequency_timer >= self.params.frequency_trip_delay:
                self.tripped = True
                self.trip_reason = f"Overfrequency trip: {frequency_hz:.2f} Hz"
        else:
            self.overfrequency_timer = 0.0
        # FIX BUG-08: Multi-stage UFLS per IEGC
        for i, (freq_thresh, shed_frac) in enumerate(self.params.ufls_stages):
            if not self.ufls_stages_tripped[i]:
                if frequency_hz < freq_thresh:
                    self.ufls_stage_timers[i] += dt
                    if self.ufls_stage_timers[i] >= self.params.frequency_trip_delay:
                        self.ufls_stages_tripped[i] = True
                        self.total_load_shed_fraction += shed_frac
                        logger.warning(
                            f"UFLS Stage {i+1}: {self.element_id} shedding "
                            f"{shed_frac*100:.0f}% load at {frequency_hz:.2f} Hz")
                else:
                    self.ufls_stage_timers[i] = max(0.0, self.ufls_stage_timers[i] - dt)
        # FIX BUG-23: Over-frequency generation trip (OFGT)
        if frequency_hz > self.params.frequency_high_gen_trip:
            self.ofgt_timer += dt
            if self.ofgt_timer >= self.params.ofgt_delay:
                self.ofgt_tripped = True
                logger.warning(
                    f"OFGT: {self.element_id} generation trip at {frequency_hz:.2f} Hz")
        else:
            self.ofgt_timer = max(0.0, self.ofgt_timer - dt)
        return {
            "element_id": self.element_id, "tripped": self.tripped,
            "trip_reason": self.trip_reason, "alarm_active": self.alarm_active,
            "alarm_reasons": self.alarm_reasons, "voltage_pu": voltage_pu,
            "frequency_hz": frequency_hz,
            "ufls_load_shed_fraction": self.total_load_shed_fraction,
            "ofgt_tripped": self.ofgt_tripped,
        }

    def reset(self):
        self.tripped = False
        self.trip_reason = None
        self.alarm_active = False
        self.alarm_reasons = []
        self.undervoltage_timer = 0.0
        self.overvoltage_timer = 0.0
        self.underfrequency_timer = 0.0
        self.overfrequency_timer = 0.0
        self.ufls_stages_tripped = [False] * len(self.params.ufls_stages)
        self.ufls_stage_timers = [0.0] * len(self.params.ufls_stages)
        self.total_load_shed_fraction = 0.0
        self.ofgt_timer = 0.0
        self.ofgt_tripped = False


# ======================== LOAD-FREQUENCY CONTROLLER ======================== #

class LoadFrequencyController:
    """Fast supplementary Load-Frequency Controller (LFC).

    Provides real-time generation adjustment to maintain system frequency at
    nominal 50 Hz.  Sits between the fast governor droop (primary, 0-30 s)
    and the slower AGC (secondary, 30 s - 15 min), closing the gap for
    disturbances that exceed primary response capability.

    Uses a PI controller with anti-windup on frequency deviation to compute
    a system-wide MW correction distributed across generators by MVA share.

    Parameters
    ----------
    Kp : float
        Proportional gain in MW / Hz.  Default 200 suits a ~6 GW system.
    Ki : float
        Integral gain in MW / (Hz * s).  Drives steady-state error to zero.
    deadband_hz : float
        Frequency deviation below which the controller does not act (IEGC).
    output_limit_mw : float
        Maximum absolute correction in MW (anti-windup).
    nominal_frequency_hz : float
        System nominal frequency (50 Hz for Indian Grid Code).
    """

    def __init__(
        self,
        Kp: float = 200.0,
        Ki: float = 20.0,
        deadband_hz: float = 0.015,
        output_limit_mw: float = 1000.0,
        nominal_frequency_hz: float = 50.0,
    ):
        self.Kp = Kp
        self.Ki = Ki
        self.deadband_hz = deadband_hz
        self.output_limit_mw = output_limit_mw
        self.nominal_frequency_hz = nominal_frequency_hz
        self.integral: float = 0.0
        self.output_mw: float = 0.0

    def update(self, dt: float, frequency_hz: float) -> float:
        """Compute total generation adjustment in MW.

        Positive -> increase generation (frequency too low).
        Negative -> decrease generation (frequency too high).
        """
        delta_f = frequency_hz - self.nominal_frequency_hz

        if abs(delta_f) < self.deadband_hz:
            # Inside deadband: decay integral slowly to avoid wind-up
            self.integral *= max(0.0, 1.0 - 0.1 * dt)
        else:
            self.integral += delta_f * dt
            # Anti-windup clamp
            max_int = self.output_limit_mw / max(self.Ki, 1e-6)
            self.integral = np.clip(self.integral, -max_int, max_int)

        self.output_mw = -(self.Kp * delta_f + self.Ki * self.integral)
        self.output_mw = np.clip(
            self.output_mw, -self.output_limit_mw, self.output_limit_mw
        )
        return self.output_mw

    def reset(self) -> None:
        """Reset controller state (call on episode reset)."""
        self.integral = 0.0
        self.output_mw = 0.0


# ======================== SYSTEM COORDINATOR ======================== #

class DynamicsCoordinator:
    """Coordinates all dynamic models. FIX C01: AGC adjustments applied."""

    def __init__(self):
        self.generators: Dict[str, SynchronousGeneratorDynamic] = {}
        self.exciters: Dict[str, ExcitationSystem] = {}
        self.governors: Dict[str, GovernorTurbine] = {}
        self.pss_units: Dict[str, PowerSystemStabilizer] = {}
        self.agc_controllers: Dict[str, AutomaticGenerationControl] = {}
        self.loads: Dict[int, DynamicLoadModel] = {}
        self.protection: Dict[str, ProtectionRelay] = {}
        self.system_frequency_hz = 50.0
        self.time_seconds = 0.0
        # AGC runs at realistic 4-second intervals (not every 20ms substep)
        self._agc_interval_s: float = 4.0
        self._agc_timer: float = 0.0
        # Load-Frequency Controller for fast supplementary frequency regulation
        self.lfc: Optional[LoadFrequencyController] = None

    def add_generator(self, gen_id: str, bus: int,
                      params: Optional[GeneratorDynamicParams] = None,
                      with_avr: bool = True, with_governor: bool = True,
                      with_pss: bool = False):
        gen = SynchronousGeneratorDynamic(gen_id, bus, params)
        self.generators[gen_id] = gen
        if with_avr:
            self.exciters[gen_id] = ExcitationSystem()
            gen.exciter = self.exciters[gen_id]
        if with_governor:
            self.governors[gen_id] = GovernorTurbine()
            gen.governor = self.governors[gen_id]
        if with_pss:
            self.pss_units[gen_id] = PowerSystemStabilizer()
            gen.pss = self.pss_units[gen_id]
        self.protection[gen_id] = ProtectionRelay(gen_id, "generator")
        logger.info(f"Added dynamic generator {gen_id} at bus {bus}")

    def add_load(self, bus: int, P0_mw: float, Q0_mvar: float):
        self.loads[bus] = DynamicLoadModel(bus, P0_mw, Q0_mvar)

    def add_agc(self, area_id: str, participating_gens: List[Tuple[str, float]]):
        agc = AutomaticGenerationControl(area_id)
        for gen_id, pf in participating_gens:
            agc.add_participating_generator(gen_id, pf)
        self.agc_controllers[area_id] = agc

    def step(self, dt: float, bus_voltages: Dict[int, float],
             gen_powers: Dict[str, float]) -> Dict:
        """Advance dynamics one step. FIX C01: AGC applied to governors."""
        self.time_seconds += dt
        # 1. Governors
        for gen_id, gov in self.governors.items():
            gen = self.generators[gen_id]
            Pm_pu = gov.update(dt, gen.omega)
            gen.set_mechanical_power(Pm_pu * gen.params.MVA_base)
        # 2. PSS
        pss_signals = {}
        for gen_id, pss in self.pss_units.items():
            gen = self.generators[gen_id]
            pss_signals[gen_id] = pss.update(dt, gen.get_speed_deviation_pu())
        # 3. Exciters
        for gen_id, exc in self.exciters.items():
            gen = self.generators[gen_id]
            Vt = bus_voltages.get(gen.bus, 1.0)
            Vpss = pss_signals.get(gen_id, 0.0)
            exc.update(dt, Vt, Vpss=Vpss)
        # 4. Generators — FIX NEW-BUG-05: Pass Efd from exciter for flux coupling
        frequencies = []
        for gen_id, gen in self.generators.items():
            Vt = bus_voltages.get(gen.bus, 1.0)
            P_elec = gen_powers.get(gen_id, gen.P_elec)
            efd = self.exciters[gen_id].Efd if gen_id in self.exciters else None
            freq, angle = gen.update(dt, Vt, P_elec, Efd=efd)
            frequencies.append(freq)
        # 5. System frequency (CoI)
        if frequencies:
            inertias = [self.generators[g].params.H * self.generators[g].params.MVA_base
                       for g in self.generators]
            total_inertia = sum(inertias)
            if total_inertia > 0:
                self.system_frequency_hz = sum(
                    f * h for f, h in zip(frequencies, inertias)) / total_inertia
            else:
                self.system_frequency_hz = np.mean(frequencies)
        # 6. AGC at realistic 4-second intervals (not every 20ms substep)
        #    Old code ran AGC every substep with gov.Pref += ... * dt,
        #    causing massive over-correction and frequency instability.
        self._agc_timer += dt
        agc_adjustments = {}
        if self._agc_timer >= self._agc_interval_s:
            elapsed = self._agc_timer
            self._agc_timer = 0.0
            for area_id, agc in self.agc_controllers.items():
                adjustments = agc.update(elapsed, self.system_frequency_hz)
                agc_adjustments[area_id] = adjustments
                for gen_id, delta_p_mw in adjustments.items():
                    if gen_id in self.governors:
                        gov = self.governors[gen_id]
                        gen = self.generators[gen_id]
                        delta_p_pu = delta_p_mw / gen.params.MVA_base
                        # Apply as direct correction to base setpoint
                        gov.base_Pref = np.clip(
                            gov.base_Pref + delta_p_pu,
                            gov.params.Pmin, gov.params.Pmax)
        # 6b. LFC — fast supplementary frequency control (every substep)
        if self.lfc is not None:
            lfc_mw = self.lfc.update(dt, self.system_frequency_hz)
            total_mva = sum(g.params.MVA_base
                           for g in self.generators.values())
            if total_mva > 0:
                for gen_id, gov in self.governors.items():
                    gen = self.generators[gen_id]
                    share = gen.params.MVA_base / total_mva
                    lfc_pu = (lfc_mw * share) / gen.params.MVA_base
                    gov.Pref = np.clip(gov.base_Pref + lfc_pu,
                                       gov.params.Pmin, gov.params.Pmax)
        else:
            for gen_id, gov in self.governors.items():
                gov.Pref = gov.base_Pref
        # 7. Loads
        load_updates = {}
        for bus, load in self.loads.items():
            V = bus_voltages.get(bus, 1.0)
            P, Q = load.update(V, self.system_frequency_hz)
            load_updates[bus] = {"P_mw": P, "Q_mvar": Q}
        # 8. Protection
        protection_status = {}
        for elem_id, relay in self.protection.items():
            gen = self.generators.get(elem_id)
            if gen:
                V = bus_voltages.get(gen.bus, 1.0)
                status = relay.check(dt, V, gen.get_frequency_hz())
                protection_status[elem_id] = status
        return {
            "time": self.time_seconds,
            "system_frequency_hz": self.system_frequency_hz,
            "generator_frequencies": {g: self.generators[g].get_frequency_hz()
                                     for g in self.generators},
            "generator_angles": {g: np.degrees(self.generators[g].delta)
                                for g in self.generators},
            "agc_adjustments": agc_adjustments,
            "load_updates": load_updates,
            "protection": protection_status
        }

    def get_system_frequency(self) -> float:
        return self.system_frequency_hz

    def get_generator_states(self) -> Dict:
        return {
            gen_id: {
                "frequency_hz": gen.get_frequency_hz(),
                "angle_deg": np.degrees(gen.delta),
                "omega_pu": gen.omega,
                "P_mech_mw": gen.P_mech,
                "P_elec_mw": gen.P_elec
            }
            for gen_id, gen in self.generators.items()
        }


# ======================== IEEE 39-BUS GENERATOR DATA ======================== #
# Keys = pandapower 0-indexed bus numbers (FIX IEEE-15: was 30-39, now 29-38)
# pandapower case39(): buses 0-38, gens at [29,31-38], ext_grid at [30]
IEEE39_GENERATOR_DATA = {
    29: {"H": 500.0, "MVA": 250, "type": "Hydro", "D": 2.0},  # D raised from 0.1 for stability
    30: {"H": 30.3, "MVA": 520, "type": "Slack", "D": 2.0},
    31: {"H": 35.8, "MVA": 650, "type": "Steam", "D": 2.0},
    32: {"H": 28.6, "MVA": 632, "type": "Steam", "D": 2.0},
    33: {"H": 26.0, "MVA": 508, "type": "Steam", "D": 2.0},
    34: {"H": 34.8, "MVA": 650, "type": "Steam", "D": 2.0},
    35: {"H": 26.4, "MVA": 560, "type": "Steam", "D": 2.0},
    36: {"H": 24.3, "MVA": 540, "type": "Steam", "D": 2.0},
    37: {"H": 34.5, "MVA": 830, "type": "Steam", "D": 2.0},
    38: {"H": 42.0, "MVA": 1000, "type": "Steam", "D": 2.0},
}


def create_ieee39_dynamics(area_offset: int = 0, area_id: str = "A") -> DynamicsCoordinator:
    """Create DynamicsCoordinator with IEEE 39-bus generators (0-indexed buses)."""
    coordinator = DynamicsCoordinator()
    participating_gens = []
    for bus, data in IEEE39_GENERATOR_DATA.items():
        global_bus = area_offset + bus
        gen_id = f"Gen_{area_id}_{bus}"
        params = GeneratorDynamicParams(
            H=data["H"], D=data["D"], MVA_base=data["MVA"], frequency=50.0)
        with_pss = data["MVA"] >= 600
        coordinator.add_generator(gen_id, global_bus, params,
                                 with_avr=True, with_governor=True, with_pss=with_pss)
        if data["type"] != "Slack":
            pf = data["MVA"] / 5000
            participating_gens.append((gen_id, pf))
    coordinator.add_agc(area_id, participating_gens)
    logger.info(f"Created dynamics for Area {area_id}: {len(coordinator.generators)} generators")
    return coordinator


# ======================== BACKWARD COMPATIBILITY ======================== #
# FIX BUG-21: Preserve old import names after PSS1A rename
PSS = PowerSystemStabilizer       # backward compat alias
PSS1A = PowerSystemStabilizer     # IEEE 421.5 alias
