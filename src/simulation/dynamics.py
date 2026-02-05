"""
Dynamic Power System Models for DEMS SuperGrid
Based on IEEE standards for 50 Hz operation

This module provides:
- Synchronous generator swing equation dynamics
- IEEE Type 1 excitation system (AVR)
- IEEE TGOV1 governor-turbine model
- Power System Stabilizer (PSS)
- Automatic Generation Control (AGC)
- Frequency and voltage dependent load models
- Protection systems

All models designed for 50 Hz Indian Grid Code operation.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


# ======================== PARAMETER CLASSES ======================== #

@dataclass
class GeneratorDynamicParams:
    """Generator dynamic parameters for 50 Hz operation"""
    H: float = 3.5              # Inertia constant (seconds)
    D: float = 2.0              # Damping coefficient (pu MW/Hz)
    MVA_base: float = 100.0     # Machine MVA base
    frequency: float = 50.0     # Operating frequency (Hz)
    
    # Transient reactances
    Xd_prime: float = 0.3       # d-axis transient reactance (pu)
    Xq_prime: float = 0.55      # q-axis transient reactance (pu)
    Td0_prime: float = 8.0      # d-axis open-circuit time constant (s)
    Tq0_prime: float = 0.4      # q-axis open-circuit time constant (s)


@dataclass
class ExciterParams:
    """IEEE Type 1 excitation system parameters (IEEET1)"""
    KA: float = 200.0           # Regulator gain
    TA: float = 0.02            # Regulator time constant (s)
    TR: float = 0.02            # Voltage transducer time constant (s)
    KE: float = 1.0             # Exciter feedback gain
    TE: float = 0.5             # Exciter time constant (s)
    VRMAX: float = 5.0          # Regulator output max (pu)
    VRMIN: float = -5.0         # Regulator output min (pu)
    EMIN: float = 0.0           # Field lower limit (pu)
    EMAX: float = 5.0           # Field upper limit (pu)
    Vref: float = 1.0           # Reference voltage (pu)


@dataclass
class GovernorParams:
    """IEEE TGOV1 governor-turbine parameters"""
    R: float = 0.05             # Droop (pu speed change for 1 pu power)
    TG: float = 0.2             # Governor time constant (s)
    TT: float = 0.5             # Turbine time constant (s)
    Pmax: float = 1.2           # Mechanical power max (pu)
    Pmin: float = 0.0           # Mechanical power min (pu)
    Pref: float = 1.0           # Reference power (pu)


@dataclass
class AGCParams:
    """Automatic Generation Control parameters"""
    beta: float = 1000.0        # Frequency bias (MW/Hz) - typical for large area
    K_agc: float = 0.5          # AGC gain
    T_agc: float = 4.0          # AGC time constant (s)
    deadband_hz: float = 0.02   # Frequency deadband (Hz)


@dataclass
class ProtectionParams:
    """Protection system parameters (Indian Grid Code IEGC)"""
    # Voltage limits (pu)
    voltage_low_warn: float = 0.95
    voltage_low_trip: float = 0.90
    voltage_high_warn: float = 1.05
    voltage_high_trip: float = 1.10
    
    # Frequency limits (Hz) - Indian Grid Code
    frequency_low_warn: float = 49.7
    frequency_low_trip: float = 49.5
    frequency_high_warn: float = 50.3
    frequency_high_trip: float = 50.5
    
    # Time delays (seconds)
    voltage_trip_delay: float = 2.0
    frequency_trip_delay: float = 0.3


# ======================== GENERATOR DYNAMIC MODELS ======================== #

class SynchronousGeneratorDynamic:
    """
    IEEE standard synchronous generator model with swing equation
    Designed for 50 Hz operation (Indian Grid Code)
    """
    
    def __init__(self, gen_id: str, bus: int, params: Optional[GeneratorDynamicParams] = None):
        """
        Create a SynchronousGeneratorDynamic instance and initialize its physical parameters, state variables, operating point, and placeholders for auxiliary control models.
        
        Parameters:
            gen_id (str): Unique identifier for the generator.
            bus (int): Bus number where the generator is connected.
            params (GeneratorDynamicParams, optional): Generator parameter set (inertia, damping, base MVA, frequency, transient reactances and time constants). If omitted, defaults are used.
        
        Notes:
            - omega_base is computed from the configured nominal frequency.
            - Initial states: delta (rad), omega (pu), Eq_prime (pu), Ed_prime (pu).
            - Operating point variables (P_mech, P_elec, Q_elec, Vt) are initialized to zero/nominal.
            - exciter, governor, and pss are left unassigned (None) and may be attached after construction.
        """
        self.gen_id = gen_id
        self.bus = bus
        self.params = params or GeneratorDynamicParams()
        
        # Base values for 50 Hz
        self.omega_base = 2 * np.pi * self.params.frequency  # 314.16 rad/s
        
        # State variables
        self.delta = 0.0          # Rotor angle (rad)
        self.omega = 1.0          # Speed (pu, 1.0 = 50 Hz)
        self.Eq_prime = 1.0       # q-axis transient EMF (pu)
        self.Ed_prime = 0.0       # d-axis transient EMF (pu)
        
        # Operating point
        self.P_mech = 0.0         # Mechanical power (MW)
        self.P_elec = 0.0         # Electrical power (MW)
        self.Q_elec = 0.0         # Reactive power (MVAR)
        self.Vt = 1.0             # Terminal voltage (pu)
        
        # Auxiliary models
        self.exciter: Optional[ExcitationSystem] = None
        self.governor: Optional[GovernorTurbine] = None
        self.pss: Optional[PowerSystemStabilizer] = None
        
    def initialize(self, P_mw: float, Q_mvar: float, Vt: float, delta_deg: float):
        """
        Set the generator's operating point and initialize derived dynamic states.
        
        Parameters:
            P_mw (float): Active electrical power at the terminal in MW.
            Q_mvar (float): Reactive electrical power at the terminal in MVAr.
            Vt (float): Terminal voltage magnitude (per‑unit).
            delta_deg (float): Rotor angle in degrees.
        
        Details:
            Updates internal state variables (P_elec, Q_elec, P_mech, Vt, delta, omega)
            and computes the internal transient EMF `Eq_prime` consistent with the
            provided terminal conditions.
        """
        self.P_elec = P_mw
        self.Q_elec = Q_mvar
        self.P_mech = P_mw  # Initial steady state
        self.Vt = Vt
        self.delta = np.radians(delta_deg)
        self.omega = 1.0  # Synchronous speed
        
        # Calculate internal EMF
        P_pu = P_mw / self.params.MVA_base
        Q_pu = Q_mvar / self.params.MVA_base
        I_mag = np.sqrt(P_pu**2 + Q_pu**2) / Vt
        
        self.Eq_prime = np.sqrt(Vt**2 + (self.params.Xd_prime * I_mag)**2)
        
        logger.debug(f"Gen {self.gen_id} initialized: P={P_mw:.1f} MW, delta={delta_deg:.1f}°")
        
    def update(self, dt: float, Vt: float, P_elec_mw: float) -> Tuple[float, float]:
        """
        Advance the generator's swing dynamics by one time step and update its internal rotor state.
        
        Parameters:
            dt (float): Time step in seconds.
            Vt (float): Terminal voltage in per-unit used to update the generator's stored terminal voltage.
            P_elec_mw (float): Electrical output power in MW used to update the generator's stored electrical power.
        
        Returns:
            tuple: (frequency_hz, rotor_angle_deg) where `frequency_hz` is the generator electrical frequency and `rotor_angle_deg` is the rotor angle in degrees.
        
        Side effects:
            Updates the instance's internal states (`omega`, `delta`), and stored attributes (`Vt`, `P_elec`).
        """
        self.Vt = Vt
        self.P_elec = P_elec_mw
        
        # Convert to per-unit
        P_mech_pu = self.P_mech / self.params.MVA_base
        P_elec_pu = P_elec_mw / self.params.MVA_base
        
        # Power imbalance
        delta_P = P_mech_pu - P_elec_pu
        
        # Swing equation: 2H * d²δ/dt² = Pm - Pe - D*(ω-1)
        # In state-space form:
        # dω/dt = (1/2H) * (Pm - Pe - D*(ω-1))
        # dδ/dt = ωbase * (ω - 1)
        
        d_omega = (1.0 / (2 * self.params.H)) * (
            delta_P - self.params.D * (self.omega - 1.0)
        ) * dt
        
        d_delta = self.omega_base * (self.omega - 1.0) * dt
        
        # Update states
        self.omega += d_omega
        self.delta += d_delta
        
        # Calculate actual frequency
        frequency_hz = self.omega * self.params.frequency
        
        return frequency_hz, np.degrees(self.delta)
    
    def set_mechanical_power(self, P_mech_mw: float):
        """
        Set the generator's mechanical input power.
        
        Parameters:
            P_mech_mw (float): Mechanical input power in megawatts for the dynamic model.
        """
        self.P_mech = P_mech_mw
        
    def get_frequency_hz(self) -> float:
        """
        Retrieve the generator's instantaneous electrical frequency.
        
        Returns:
            frequency_hz (float): Instantaneous frequency in hertz.
        """
        return self.omega * self.params.frequency
    
    def get_speed_deviation_pu(self) -> float:
        """
        Calculate the rotor speed deviation relative to synchronous speed.
        
        Returns:
            speed_deviation_pu (float): Rotor speed deviation in per unit (omega - 1.0).
        """
        return self.omega - 1.0


class ExcitationSystem:
    """
    IEEE Type 1 Excitation System (IEEET1)
    Automatic Voltage Regulator (AVR)
    """
    
    def __init__(self, params: Optional[ExciterParams] = None):
        """
        Initialize the IEEE Type 1 excitation system and its internal state.
        
        Creates or uses the provided ExciterParams and sets initial regulator, field, and transducer states:
        - Vr (regulator output) = 1.0 pu
        - Efd (field voltage) = 1.0 pu
        - Vt_filtered (filtered terminal voltage) = 1.0 pu
        Also initializes the local Vref from the exciter parameters.
        
        Parameters:
            params (Optional[ExciterParams]): Exciter configuration to use; a default ExciterParams instance is created when omitted.
        """
        self.params = params or ExciterParams()
        
        # State variables
        self.Vr = 1.0             # Regulator output (pu)
        self.Efd = 1.0            # Field voltage (pu)
        self.Vt_filtered = 1.0    # Filtered terminal voltage
        self.Vref = self.params.Vref
        
    def update(self, dt: float, Vt: float, Vref: Optional[float] = None, 
               Vpss: float = 0.0) -> float:
        """
               Update the IEEE Type 1 excitation system state and compute the field voltage.
               
               Updates the voltage transducer, regulator, and exciter states using first-order dynamics, enforces regulator and field limits, and returns the resulting field voltage.
               
               Parameters:
                   dt (float): Time step in seconds.
                   Vt (float): Measured terminal voltage in per unit.
                   Vref (Optional[float]): If provided, overrides the exciter reference voltage for this update.
                   Vpss (float): Power system stabilizer contribution added to the voltage error (pu).
               
               Returns:
                   float: Field voltage Efd in per unit after limits are applied.
               """
        if Vref is not None:
            self.Vref = Vref
            
        # Voltage transducer (first-order filter)
        dVt_filt = (Vt - self.Vt_filtered) / self.params.TR * dt
        self.Vt_filtered += dVt_filt
        
        # Voltage error (including PSS)
        Ve = self.Vref - self.Vt_filtered + Vpss
        
        # Regulator dynamics
        dVr = (self.params.KA * Ve - self.Vr) / self.params.TA * dt
        self.Vr += dVr
        
        # Apply regulator limits
        self.Vr = np.clip(self.Vr, self.params.VRMIN, self.params.VRMAX)
        
        # Exciter dynamics
        dEfd = (self.params.KE * self.Vr - self.Efd) / self.params.TE * dt
        self.Efd += dEfd
        
        # Apply field voltage limits
        self.Efd = np.clip(self.Efd, self.params.EMIN, self.params.EMAX)
        
        return self.Efd


class GovernorTurbine:
    """
    IEEE TGOV1 Governor-Turbine Model
    Primary frequency control with droop
    """
    
    def __init__(self, params: Optional[GovernorParams] = None):
        """
        Create a TGOV1 governor-turbine instance and initialize its state.
        
        Parameters:
            params (Optional[GovernorParams]): Governor parameters; a default GovernorParams instance is used when omitted.
        
        Details:
            Initializes governor output `Pg` and mechanical power `Pm` to 1.0 pu, and sets `Pref` from `params.Pref`.
        """
        self.params = params or GovernorParams()
        
        # State variables
        self.Pg = 1.0             # Governor output (pu)
        self.Pm = 1.0             # Mechanical power (pu)
        self.Pref = self.params.Pref
        
    def update(self, dt: float, omega: float, Pref: Optional[float] = None) -> float:
        """
        Advance the TGOV1 governor-turbine model by a time step and return the updated turbine mechanical power.
        
        Parameters:
            dt (float): Time step in seconds.
            omega (float): Generator speed in per unit (1.0 = synchronous).
            Pref (Optional[float]): Optional power reference in per unit; if provided, replaces the controller's stored Pref.
        
        Returns:
            float: Updated turbine mechanical power Pm in per unit.
        """
        if Pref is not None:
            self.Pref = Pref
            
        # Speed deviation
        delta_omega = omega - 1.0
        
        # Droop control: Pref adjusted by speed error
        # For 5% droop, 1% speed change = 20% power change
        Pg_ref = self.Pref - delta_omega / self.params.R
        
        # Governor dynamics (first-order)
        dPg = (Pg_ref - self.Pg) / self.params.TG * dt
        self.Pg += dPg
        
        # Apply power limits
        self.Pg = np.clip(self.Pg, self.params.Pmin, self.params.Pmax)
        
        # Turbine dynamics (first-order delay)
        dPm = (self.Pg - self.Pm) / self.params.TT * dt
        self.Pm += dPm
        
        return self.Pm


class PowerSystemStabilizer:
    """
    Power System Stabilizer (PSS)
    Provides damping for electromechanical oscillations
    """
    
    def __init__(self):
        # IEEE PSS2A-type parameters (simplified)
        """
        Initialize a simplified IEEE PSS2A-like power system stabilizer with default tuning and zeroed internal states.
        
        Attributes:
            T_washout (float): Washout time constant in seconds.
            T_lead1 (float): Lead time constant 1 in seconds.
            T_lag1 (float): Lag time constant 1 in seconds.
            K_pss (float): PSS gain.
            V_max (float): Upper output limit (per unit).
            V_min (float): Lower output limit (per unit).
            washout_state (float): Internal washout filter state.
            lead_lag_state (float): Internal lead-lag compensator state.
            output (float): Current PSS output (per unit).
        """
        self.T_washout = 1.41     # Washout time constant (s)
        self.T_lead1 = 0.154      # Lead time constant 1 (s)
        self.T_lag1 = 0.033       # Lag time constant 1 (s)
        self.K_pss = 9.5          # PSS gain
        self.V_max = 0.2          # Output limit (pu)
        self.V_min = -0.2
        
        # State variables
        self.washout_state = 0.0
        self.lead_lag_state = 0.0
        self.output = 0.0
        
    def update(self, dt: float, speed_deviation: float) -> float:
        """
        Compute the Power System Stabilizer (PSS) output for a single time step.
        
        Parameters:
            dt (float): Time step in seconds.
            speed_deviation (float): Generator speed deviation in per unit.
        
        Returns:
            pss_output (float): PSS control signal in per unit, constrained to the configured output limits.
        """
        # Washout filter (high-pass)
        washout_input = speed_deviation
        d_washout = (washout_input - self.washout_state) / self.T_washout * dt
        self.washout_state += d_washout
        washout_out = washout_input - self.washout_state
        
        # Lead-lag compensator
        lead_input = washout_out * self.K_pss
        d_lead_lag = (lead_input * (self.T_lead1 / self.T_lag1) - self.lead_lag_state) / self.T_lag1 * dt
        self.lead_lag_state += d_lead_lag
        
        # Apply output limits
        self.output = np.clip(self.lead_lag_state, self.V_min, self.V_max)
        
        return self.output


# ======================== SYSTEM CONTROL ======================== #

class AutomaticGenerationControl:
    """
    Automatic Generation Control (AGC)
    Secondary frequency control for area regulation
    
    Implements Area Control Error (ACE) based control:
    ACE = ΔPtie + β × Δf
    """
    
    def __init__(self, area_id: str, params: Optional[AGCParams] = None):
        """
        Create an AutomaticGenerationControl controller for a specified control area.
        
        Parameters:
        	area_id (str): Identifier for the control area this AGC instance manages.
        	params (Optional[AGCParams]): AGC configuration parameters; if omitted, defaults are used.
        
        Initial state:
        	- ace: Area Control Error (MW), initialized to 0.0.
        	- agc_output: AGC signal to participating generators (MW), initialized to 0.0.
        	- integral_ace: Time integral of ACE for PI action, initialized to 0.0.
        	- participating_gens: Registered generator IDs for AGC participation.
        	- participation_factors: Mapping of generator ID to its participation fraction.
        """
        self.area_id = area_id
        self.params = params or AGCParams()
        
        # State variables
        self.ace = 0.0            # Area Control Error (MW)
        self.agc_output = 0.0     # AGC signal to generators (MW)
        self.integral_ace = 0.0   # Integral of ACE
        
        # Participating generators
        self.participating_gens: List[str] = []
        self.participation_factors: Dict[str, float] = {}
        
    def add_participating_generator(self, gen_id: str, participation_factor: float):
        """
        Register a generator to participate in the area AGC and assign its participation share.
        
        Parameters:
            gen_id (str): Identifier of the generator to register.
            participation_factor (float): Non-negative weight or share used to proportionally allocate AGC adjustments to this generator.
        """
        self.participating_gens.append(gen_id)
        self.participation_factors[gen_id] = participation_factor
        
    def update(self, dt: float, frequency_hz: float, tie_line_error_mw: float = 0.0) -> Dict[str, float]:
        """
        Compute the area AGC response and allocate per-generator power adjustments.
        
        Applies the configured deadband and area frequency bias to form ACE, uses a PI controller with anti-windup to produce an AGC command, and distributes that command to registered generators according to their participation factors.
        
        Parameters:
            dt (float): Time step in seconds.
            frequency_hz (float): Current area frequency in Hz.
            tie_line_error_mw (float): Tie-line flow error (ΔPtie) in MW; positive means export greater than scheduled.
        
        Returns:
            Dict[str, float]: Mapping from generator ID to AGC power adjustment in MW (positive values increase mechanical setpoint).
        """
        # Frequency deviation from 50 Hz
        delta_f = frequency_hz - 50.0
        
        # Apply deadband
        if abs(delta_f) < self.params.deadband_hz:
            delta_f = 0.0
            
        # Area Control Error
        # ACE = ΔPtie + β × Δf (positive ACE = over-generation)
        self.ace = tie_line_error_mw + self.params.beta * delta_f
        
        # PI controller
        # Proportional term
        p_term = -self.params.K_agc * self.ace
        
        # Integral term (with anti-windup)
        self.integral_ace += self.ace * dt
        self.integral_ace = np.clip(self.integral_ace, -1000, 1000)  # Anti-windup
        i_term = -self.integral_ace / self.params.T_agc
        
        # Total AGC output
        self.agc_output = p_term + i_term
        
        # Distribute to participating generators
        adjustments = {}
        for gen_id in self.participating_gens:
            pf = self.participation_factors.get(gen_id, 0.0)
            adjustments[gen_id] = self.agc_output * pf
            
        return adjustments
    
    def get_ace(self) -> float:
        """
        Retrieve the current area control error for this AGC.
        
        Returns:
            ace_mw (float): Current area control error in megawatts.
        """
        return self.ace


# ======================== LOAD MODELS ======================== #

class DynamicLoadModel:
    """
    Frequency and voltage dependent load model
    ZIP load model with frequency dependence
    
    P = P0 × (Zp × V² + Ip × V + Pp) × (1 + Kpf × Δf)
    Q = Q0 × (Zq × V² + Iq × V + Pq) × (1 + Kqf × Δf)
    """
    
    def __init__(self, bus: int, P0_mw: float, Q0_mvar: float):
        """
        Initialize a ZIP-format dynamic load model with default voltage and frequency dependence coefficients.
        
        Parameters:
            bus (int): Bus number where the load is connected.
            P0_mw (float): Base active power in MW used as the nominal operating point.
            Q0_mvar (float): Base reactive power in MVAR used as the nominal operating point.
        
        Notes:
            - The instance sets default ZIP composition coefficients for active (Zp, Ip, Pp) and reactive (Zq, Iq, Pq) power.
            - Frequency sensitivity coefficients Kpf and Kqf are expressed in percent change per Hz and apply to active and reactive power respectively.
            - P_mw and Q_mvar are initialized to the provided base values and represent the current load state.
        """
        self.bus = bus
        self.P0 = P0_mw           # Base active power (MW)
        self.Q0 = Q0_mvar         # Base reactive power (MVAR)
        
        # ZIP coefficients (typical composite load)
        # Z = constant impedance, I = constant current, P = constant power
        self.Zp = 0.4             # Active power impedance portion
        self.Ip = 0.3             # Active power current portion
        self.Pp = 0.3             # Active power constant portion
        
        self.Zq = 0.5             # Reactive power impedance portion
        self.Iq = 0.3             # Reactive power current portion
        self.Pq = 0.2             # Reactive power constant portion
        
        # Frequency dependence coefficients
        self.Kpf = 1.5            # Active power freq coefficient (%/Hz)
        self.Kqf = -1.0           # Reactive power freq coefficient (%/Hz)
        
        # Current values
        self.P_mw = P0_mw
        self.Q_mvar = Q0_mvar
        
    def update(self, voltage_pu: float, frequency_hz: float) -> Tuple[float, float]:
        """
        Compute and update the load's active and reactive power based on ZIP voltage dependence and linear frequency sensitivity.
        
        Parameters:
            voltage_pu (float): Bus voltage in per unit.
            frequency_hz (float): System frequency in hertz (50 Hz nominal).
        
        Returns:
            tuple: (P_mw, Q_mvar) — updated active power in MW and reactive power in MVar; also stored to self.P_mw and self.Q_mvar. Frequency sensitivity is applied using delta_f = (frequency_hz - 50.0) / 50.0.
        """
        # Frequency deviation (per unit of 50 Hz)
        delta_f = (frequency_hz - 50.0) / 50.0
        
        # Voltage dependence (ZIP model)
        V = voltage_pu
        V2 = V * V
        
        p_voltage_factor = self.Zp * V2 + self.Ip * V + self.Pp
        q_voltage_factor = self.Zq * V2 + self.Iq * V + self.Pq
        
        # Frequency dependence
        p_freq_factor = 1.0 + self.Kpf * delta_f
        q_freq_factor = 1.0 + self.Kqf * delta_f
        
        # Calculate actual load
        self.P_mw = self.P0 * p_voltage_factor * p_freq_factor
        self.Q_mvar = self.Q0 * q_voltage_factor * q_freq_factor
        
        return self.P_mw, self.Q_mvar


# ======================== PROTECTION SYSTEMS ======================== #

class ProtectionRelay:
    """
    Protection relay with voltage and frequency protection
    Indian Grid Code (IEGC) compliant settings
    """
    
    def __init__(self, element_id: str, element_type: str, 
                 params: Optional[ProtectionParams] = None):
        """
        Initialize a protection relay for a network element and set its default state and timers.

        Parameters:
            element_id (str): Unique identifier of the protected element (e.g., bus or device ID).
            element_type (str): Type of the protected element; expected values include "generator", "line", or "load".
            params (Optional[ProtectionParams]): Protection thresholds and delay settings; when omitted, defaults are used.

        Description:
            Creates the relay and initializes its runtime state: trip flag, trip reason, alarm flags/reasons,
            and delayed-trip timers for under/over voltage and under/over frequency.
        """
        self.element_id = element_id
        self.element_type = element_type  # "generator", "line", "load"
        self.params = params or ProtectionParams()
        
        # Relay state
        self.tripped = False
        self.trip_reason: Optional[str] = None
        self.alarm_active = False
        self.alarm_reasons: List[str] = []
        
        # Timers for delayed tripping
        self.undervoltage_timer = 0.0
        self.overvoltage_timer = 0.0
        self.underfrequency_timer = 0.0
        self.overfrequency_timer = 0.0
        
    def check(self, dt: float, voltage_pu: float, frequency_hz: float) -> Dict:
        """
        Evaluate voltage and frequency protection for the element, update internal timers, and set alarm/trip state.
        
        This method checks warn and trip thresholds for voltage and frequency, appends human-readable alarm messages when warn thresholds are crossed, increments time-delayed trip timers, sets the trip state and trip reason when a trip delay is exceeded, and resets timers when conditions clear.
        
        Parameters:
            dt (float): Time step in seconds used to advance trip timers.
            voltage_pu (float): Measured element voltage in per unit.
            frequency_hz (float): Measured system frequency in hertz.
        
        Returns:
            dict: Protection status with keys:
                - "element_id": identifier of the protected element.
                - "tripped": `True` if the element has been tripped, `False` otherwise.
                - "trip_reason": human-readable reason for the trip or empty string if not tripped.
                - "alarm_active": `True` if any warning-level conditions are active.
                - "alarm_reasons": list of human-readable warning messages.
                - "voltage_pu": the input `voltage_pu`.
                - "frequency_hz": the input `frequency_hz`.
        """
        self.alarm_reasons = []
        
        # Check voltage conditions
        if voltage_pu < self.params.voltage_low_warn:
            self.alarm_reasons.append(f"Low voltage warning: {voltage_pu:.3f} pu")
            
        if voltage_pu > self.params.voltage_high_warn:
            self.alarm_reasons.append(f"High voltage warning: {voltage_pu:.3f} pu")
            
        # Check frequency conditions  
        if frequency_hz < self.params.frequency_low_warn:
            self.alarm_reasons.append(f"Low frequency warning: {frequency_hz:.2f} Hz")
            
        if frequency_hz > self.params.frequency_high_warn:
            self.alarm_reasons.append(f"High frequency warning: {frequency_hz:.2f} Hz")
            
        self.alarm_active = len(self.alarm_reasons) > 0
        
        # Check trip conditions with time delays
        # Undervoltage
        if voltage_pu < self.params.voltage_low_trip:
            self.undervoltage_timer += dt
            if self.undervoltage_timer >= self.params.voltage_trip_delay:
                self.tripped = True
                self.trip_reason = f"Undervoltage trip: {voltage_pu:.3f} pu"
        else:
            self.undervoltage_timer = 0.0
            
        # Overvoltage
        if voltage_pu > self.params.voltage_high_trip:
            self.overvoltage_timer += dt
            if self.overvoltage_timer >= self.params.voltage_trip_delay:
                self.tripped = True
                self.trip_reason = f"Overvoltage trip: {voltage_pu:.3f} pu"
        else:
            self.overvoltage_timer = 0.0
            
        # Underfrequency (faster trip)
        if frequency_hz < self.params.frequency_low_trip:
            self.underfrequency_timer += dt
            if self.underfrequency_timer >= self.params.frequency_trip_delay:
                self.tripped = True
                self.trip_reason = f"Underfrequency trip: {frequency_hz:.2f} Hz"
        else:
            self.underfrequency_timer = 0.0
            
        # Overfrequency
        if frequency_hz > self.params.frequency_high_trip:
            self.overfrequency_timer += dt
            if self.overfrequency_timer >= self.params.frequency_trip_delay:
                self.tripped = True
                self.trip_reason = f"Overfrequency trip: {frequency_hz:.2f} Hz"
        else:
            self.overfrequency_timer = 0.0
            
        return {
            "element_id": self.element_id,
            "tripped": self.tripped,
            "trip_reason": self.trip_reason,
            "alarm_active": self.alarm_active,
            "alarm_reasons": self.alarm_reasons,
            "voltage_pu": voltage_pu,
            "frequency_hz": frequency_hz
        }
        
    def reset(self):
        """
        Reset the protection relay to its default (untripped) state.
        
        Clears trip status and reason, deactivates alarms and alarm reasons, and resets all protection timers to zero.
        """
        self.tripped = False
        self.trip_reason = None
        self.alarm_active = False
        self.alarm_reasons = []
        self.undervoltage_timer = 0.0
        self.overvoltage_timer = 0.0
        self.underfrequency_timer = 0.0
        self.overfrequency_timer = 0.0


# ======================== SYSTEM COORDINATOR ======================== #

class DynamicsCoordinator:
    """
    Coordinates all dynamic models in the simulation
    Manages time-stepping and model interactions
    """
    
    def __init__(self):
        """
        Initialize the DynamicsCoordinator internal state used to manage dynamic models and simulation time.
        
        Creates empty registries for generators, exciters, governors, PSS units, AGC controllers, dynamic loads, and protection relays, and sets the simulation clock and nominal system frequency (50.0 Hz).
        """
        self.generators: Dict[str, SynchronousGeneratorDynamic] = {}
        self.exciters: Dict[str, ExcitationSystem] = {}
        self.governors: Dict[str, GovernorTurbine] = {}
        self.pss_units: Dict[str, PowerSystemStabilizer] = {}
        self.agc_controllers: Dict[str, AutomaticGenerationControl] = {}
        self.loads: Dict[int, DynamicLoadModel] = {}
        self.protection: Dict[str, ProtectionRelay] = {}
        
        # System state
        self.system_frequency_hz = 50.0
        self.time_seconds = 0.0
        
    def add_generator(self, gen_id: str, bus: int, 
                      params: Optional[GeneratorDynamicParams] = None,
                      with_avr: bool = True, 
                      with_governor: bool = True,
                      with_pss: bool = False):
        """
                      Register a synchronous generator with the coordinator and optionally attach AVR, governor, and PSS control modules.
                      
                      Parameters:
                          gen_id (str): Unique identifier for the generator.
                          bus (int): Bus number where the generator is connected.
                          params (Optional[GeneratorDynamicParams]): Generator dynamic parameters; defaults are used if None.
                          with_avr (bool): If True, attach an IEEE Type 1 excitation system (AVR) to the generator.
                          with_governor (bool): If True, attach a TGOV1 governor-turbine model to the generator.
                          with_pss (bool): If True, attach a power system stabilizer (PSS) to the generator.
                      
                      Side effects:
                          - Creates and stores the SynchronousGeneratorDynamic instance and any requested control components in the coordinator's internal registries.
                          - Adds a ProtectionRelay for the generator and logs the addition.
                      """
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
            
        # Add protection relay
        self.protection[gen_id] = ProtectionRelay(gen_id, "generator")
        
        logger.info(f"Added dynamic generator {gen_id} at bus {bus}")
        
    def add_load(self, bus: int, P0_mw: float, Q0_mvar: float):
        """
        Register a ZIP-based dynamic load model at the given bus with an initial active and reactive power setpoint.
        
        Parameters:
        	bus (int): Bus identifier where the dynamic load is attached.
        	P0_mw (float): Initial active power demand in megawatts.
        	Q0_mvar (float): Initial reactive power demand in megavars.
        """
        self.loads[bus] = DynamicLoadModel(bus, P0_mw, Q0_mvar)
        
    def add_agc(self, area_id: str, participating_gens: List[Tuple[str, float]]):
        """
        Register an Automatic Generation Control (AGC) instance for a given area and configure its participating generators.
        
        Parameters:
            area_id (str): Identifier for the control area.
            participating_gens (List[Tuple[str, float]]): Iterable of (gen_id, participation_factor) pairs specifying each generator's participation share in AGC. 
        """
        agc = AutomaticGenerationControl(area_id)
        for gen_id, pf in participating_gens:
            agc.add_participating_generator(gen_id, pf)
        self.agc_controllers[area_id] = agc
        
    def step(self, dt: float, bus_voltages: Dict[int, float], 
             gen_powers: Dict[str, float]) -> Dict:
        """
             Advance the entire dynamic simulation by a single time step.
             
             Parameters:
                 dt (float): Time step in seconds.
                 bus_voltages (Dict[int, float]): Mapping from bus number to per-unit terminal voltage.
                 gen_powers (Dict[str, float]): Mapping from generator ID to electrical output power in MW; if a generator ID is absent its last known electrical power is used.
             
             Returns:
                 state (Dict): Snapshot of the updated system state containing:
                     - time: current simulation time in seconds
                     - system_frequency_hz: aggregated system frequency in Hz
                     - generator_frequencies: mapping {gen_id: frequency_hz}
                     - generator_angles: mapping {gen_id: rotor angle in degrees}
                     - agc_adjustments: mapping {area_id: {gen_id: delta_P_mw}} from AGC updates
                     - load_updates: mapping {bus: {"P_mw": P, "Q_mvar": Q}} for dynamic loads
                     - protection: mapping {element_id: protection_status} with relay check results
             """
        self.time_seconds += dt
        
        # 1. Update governors (primary frequency response)
        for gen_id, gov in self.governors.items():
            gen = self.generators[gen_id]
            Pm_pu = gov.update(dt, gen.omega)
            gen.set_mechanical_power(Pm_pu * gen.params.MVA_base)
            
        # 2. Update PSS (if enabled)
        pss_signals = {}
        for gen_id, pss in self.pss_units.items():
            gen = self.generators[gen_id]
            pss_signals[gen_id] = pss.update(dt, gen.get_speed_deviation_pu())
            
        # 3. Update exciters (AVR)
        for gen_id, exc in self.exciters.items():
            gen = self.generators[gen_id]
            Vt = bus_voltages.get(gen.bus, 1.0)
            Vpss = pss_signals.get(gen_id, 0.0)
            exc.update(dt, Vt, Vpss=Vpss)
            
        # 4. Update generator dynamics (swing equation)
        frequencies = []
        for gen_id, gen in self.generators.items():
            Vt = bus_voltages.get(gen.bus, 1.0)
            P_elec = gen_powers.get(gen_id, gen.P_elec)
            freq, angle = gen.update(dt, Vt, P_elec)
            frequencies.append(freq)
            
        # 5. Calculate system frequency (weighted average)
        if frequencies:
            self.system_frequency_hz = np.mean(frequencies)
            
        # 6. Update AGC (secondary frequency response)
        agc_adjustments = {}
        for area_id, agc in self.agc_controllers.items():
            adjustments = agc.update(dt, self.system_frequency_hz)
            agc_adjustments[area_id] = adjustments
            
        # 7. Update loads
        load_updates = {}
        for bus, load in self.loads.items():
            V = bus_voltages.get(bus, 1.0)
            P, Q = load.update(V, self.system_frequency_hz)
            load_updates[bus] = {"P_mw": P, "Q_mvar": Q}
            
        # 8. Check protection
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
        """
        Retrieve the current system frequency.
        
        Returns:
            system_frequency_hz (float): Current system frequency in hertz.
        """
        return self.system_frequency_hz
    
    def get_generator_states(self) -> Dict:
        """
        Snapshot of each generator's dynamic state.
        
        Returns:
            states (Dict[str, Dict]): Mapping from generator ID to a dictionary with keys:
                - frequency_hz: generator electrical frequency in Hz
                - angle_deg: rotor angle in degrees
                - omega_pu: per-unit rotor speed
                - P_mech_mw: mechanical power in MW
                - P_elec_mw: electrical output power in MW
        """
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

# Standard IEEE 39-bus generator parameters (exact IEEE values)
IEEE39_GENERATOR_DATA = {
    30: {"H": 500.0, "MVA": 250, "type": "Hydro", "D": 0.1},    # Gen 1 (very large H = aggregated system)
    31: {"H": 30.3, "MVA": 520, "type": "Steam", "D": 2.0},     # Gen 2
    32: {"H": 35.8, "MVA": 650, "type": "Steam", "D": 2.0},     # Gen 3
    33: {"H": 28.6, "MVA": 632, "type": "Steam", "D": 2.0},     # Gen 4
    34: {"H": 26.0, "MVA": 508, "type": "Steam", "D": 2.0},     # Gen 5
    35: {"H": 34.8, "MVA": 650, "type": "Steam", "D": 2.0},     # Gen 6
    36: {"H": 26.4, "MVA": 560, "type": "Steam", "D": 2.0},     # Gen 7
    37: {"H": 24.3, "MVA": 540, "type": "Steam", "D": 2.0},     # Gen 8
    38: {"H": 34.5, "MVA": 830, "type": "Steam", "D": 2.0},     # Gen 9
    39: {"H": 42.0, "MVA": 1000, "type": "Slack", "D": 2.0}     # Gen 10 (slack/infinite bus)
}


def create_ieee39_dynamics(area_offset: int = 0, area_id: str = "A") -> DynamicsCoordinator:
    """
    Constructs a DynamicsCoordinator populated with IEEE 39-bus generator models and an area AGC.
    
    Parameters:
        area_offset (int): Bus index offset applied to each IEEE39 bus number (e.g., 0, 39, 78) to create a merged supergrid.
        area_id (str): Area identifier used when naming generators (e.g., "A", "B", "C").
    
    Returns:
        coordinator (DynamicsCoordinator): Coordinator containing one generator per IEEE39 entry (generator IDs formatted as "Gen_{area_id}_{bus}"), AGC configured for the area, PSS attached to units with MVA >= 600, and participation factors for AGC proportional to generator MVA (MVA/5000). Slack machines are excluded from AGC participation.
    """
    coordinator = DynamicsCoordinator()
    
    participating_gens = []
    
    for bus, data in IEEE39_GENERATOR_DATA.items():
        global_bus = area_offset + bus
        gen_id = f"Gen_{area_id}_{bus}"
        
        params = GeneratorDynamicParams(
            H=data["H"],
            D=data["D"],
            MVA_base=data["MVA"],
            frequency=50.0
        )
        
        # Add generator with AVR and governor (PSS for larger units)
        with_pss = data["MVA"] >= 600  # PSS for larger units
        coordinator.add_generator(gen_id, global_bus, params,
                                 with_avr=True, with_governor=True, with_pss=with_pss)
        
        # Add to AGC (except slack)
        if data["type"] != "Slack":
            pf = data["MVA"] / 5000  # Participation proportional to size
            participating_gens.append((gen_id, pf))
            
    # Create AGC for area
    coordinator.add_agc(area_id, participating_gens)
    
    logger.info(f"Created dynamics for Area {area_id}: {len(coordinator.generators)} generators")
    
    return coordinator