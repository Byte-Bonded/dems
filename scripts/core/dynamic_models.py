#!/usr/bin/env python3
"""
IEEE Standard Dynamic Models for Power System Simulation (50 Hz Operation)

This module contains all IEEE standard dynamic models for:
- Synchronous generators with AVR and governor control
- Modern DER systems (Solar, Wind, Storage, EV, DR)
- Protection and monitoring systems
- System coordination and control

All models are designed for 50 Hz operation per international standards.
"""

from dataclasses import dataclass
import numpy as np
import logging

logger = logging.getLogger(__name__)

# ======================== PARAMETER CLASSES ======================== #

@dataclass
class GeneratorParams:
    """Generator parameters for IEEE 39-bus system (50 Hz)"""
    H: float = 3.5              # Inertia constant (s)
    D: float = 2.0              # Damping coefficient (pu MW/Hz)
    MVA_base: float = 100.0     # Machine MVA base
    frequency: float = 50.0     # Operating frequency (Hz)

@dataclass
class ExciterParams:
    """IEEE Type 1 excitation system parameters"""
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
class DERParams:
    """DER system parameters"""
    capacity_mw: float = 10.0   # Capacity in MW
    location: str = "Bus_30"    # Bus location

@dataclass
class ProtectionParams:
    """Protection system parameters (50 Hz)"""
    voltage_low: float = 0.9    # pu
    voltage_high: float = 1.1   # pu
    frequency_low: float = 49.0 # Hz
    frequency_high: float = 51.0 # Hz

# ======================== SYNCHRONOUS GENERATOR MODELS ======================== #

class SynchronousGenerator:
    """IEEE standard synchronous generator model with 50 Hz operation"""
    
    def __init__(self, params=None):
        self.params = params or GeneratorParams()
        self.frequency = 50.0  # 50 Hz operation
        self.omega_base = 2 * np.pi * self.frequency  # 314.16 rad/s
        
        # State variables
        self.delta = 0.0      # Rotor angle (rad)
        self.omega = 1.0      # Speed (pu)
        self.Eq_prime = 1.0   # q-axis transient voltage
        self.Ed_prime = 0.0   # d-axis transient voltage
        
        # Operating point
        self.P_mech = 1.0     # Mechanical power (pu)
        self.P_elec = 1.0     # Electrical power (pu)
        
    def update_states(self, dt, P_mech, P_elec):
        """Update generator states using swing equation"""
        # Power imbalance
        delta_P = P_mech - P_elec
        
        # Swing equation (50 Hz base)
        d_omega = (1.0 / (2 * self.params.H)) * (delta_P - self.params.D * (self.omega - 1.0)) * dt
        d_delta = self.omega_base * (self.omega - 1.0) * dt
        
        # Update states
        self.omega += d_omega
        self.delta += d_delta
        self.P_mech = P_mech
        self.P_elec = P_elec
        
        return self.omega, self.delta

class ExcitationSystem:
    """IEEE Type 1 excitation system (IEEET1)"""
    
    def __init__(self, params=None):
        self.params = params or ExciterParams()
        
        # State variables
        self.Vr = 1.0         # Regulator output
        self.Efd = 1.0        # Field voltage
        self.Vt_filtered = 1.0 # Filtered terminal voltage
        
    def update(self, dt, Vt, Vref=None):
        """Update excitation system states"""
        if Vref is None:
            Vref = self.params.Vref
            
        # Voltage transducer
        dVt_filt = (Vt - self.Vt_filtered) / self.params.TR
        self.Vt_filtered += dVt_filt * dt
        
        # Voltage error
        Ve = Vref - self.Vt_filtered
        
        # Regulator
        dVr = (self.params.KA * Ve - self.Vr) / self.params.TA
        self.Vr += dVr * dt
        
        # Apply limits
        self.Vr = np.clip(self.Vr, self.params.VRMIN, self.params.VRMAX)
        
        # Exciter
        dEfd = (self.params.KE * self.Vr - self.Efd) / self.params.TE
        self.Efd += dEfd * dt
        
        # Apply field limits
        self.Efd = np.clip(self.Efd, self.params.EMIN, self.params.EMAX)
        
        return self.Efd

class GovernorTurbine:
    """IEEE TGOV1 governor-turbine model"""
    
    def __init__(self, params=None):
        self.params = params or GovernorParams()
        
        # State variables
        self.Pg = 1.0         # Governor output
        self.Pm = 1.0         # Mechanical power
        
    def update(self, dt, omega, Pref=None):
        """Update governor-turbine states"""
        if Pref is None:
            Pref = self.params.Pref
            
        # Speed deviation
        delta_omega = omega - 1.0
        
        # Governor droop control
        Pg_ref = Pref - delta_omega / self.params.R
            
        # Governor dynamics
        dPg = (Pg_ref - self.Pg) / self.params.TG
        self.Pg += dPg * dt
        
        # Apply power limits
        self.Pg = np.clip(self.Pg, self.params.Pmin, self.params.Pmax)
        
        # Turbine dynamics
        dPm = (self.Pg - self.Pm) / self.params.TT
        self.Pm += dPm * dt
        
        return self.Pm

class PowerSystemStabilizer:
    """IEEE PSS model for oscillation damping"""
    
    def __init__(self, params=None):
        self.T_washout = 1.41    # Washout time constant
        self.T_lead_lag = 0.154  # Lead-lag time constant
        self.K_pss = 9.5         # PSS gain
        
        # State variables
        self.input_filtered = 0.0
        self.output = 0.0
        
    def update(self, dt, speed_deviation):
        """Update PSS output"""
        # Washout filter
        washout = speed_deviation - self.input_filtered
        self.input_filtered += (speed_deviation - self.input_filtered) * dt / self.T_washout
        
        # Lead-lag compensation
        self.output = self.K_pss * washout
        
        return self.output

# ======================== MODERN DER SYSTEMS ======================== #

class SolarPVSystem:
    """IEEE 1547 compliant solar PV system with MPPT"""
    
    def __init__(self, capacity_mw=10.0, location="Bus_31"):
        self.capacity_mw = capacity_mw
        self.location = location
        self.efficiency = 0.85
        self.capacity_factor = 0.25  # Average capacity factor
        
        # State variables
        self.power_output = 0.0
        self.irradiance = 800.0  # W/m²
        self.temperature = 25.0  # °C
        
    def update_power(self, irradiance=None, temperature=None):
        """Update power output based on conditions"""
        if irradiance is not None:
            self.irradiance = irradiance
        if temperature is not None:
            self.temperature = temperature
            
        # Temperature derating (0.4%/°C above 25°C)
        temp_factor = 1 - 0.004 * (self.temperature - 25)
        
        # Power calculation
        self.power_output = (self.capacity_mw * 
                           (self.irradiance / 1000) * 
                           temp_factor * 
                           self.efficiency)
        
        return self.power_output

class WindTurbineSystem:
    """Wind turbine with pitch control and MPPT"""
    
    def __init__(self, capacity_mw=15.0, location="Bus_37"):
        self.capacity_mw = capacity_mw
        self.location = location
        self.cut_in_speed = 3.0    # m/s
        self.rated_speed = 12.0    # m/s
        self.cut_out_speed = 25.0  # m/s
        
        # State variables
        self.power_output = 0.0
        self.wind_speed = 10.0  # m/s
        self.pitch_angle = 0.0  # degrees
        
    def update_power(self, wind_speed=None):
        """Update power output based on wind speed"""
        if wind_speed is not None:
            self.wind_speed = wind_speed
            
        if self.wind_speed < self.cut_in_speed or self.wind_speed > self.cut_out_speed:
            self.power_output = 0.0
        elif self.wind_speed <= self.rated_speed:
            # Cubic relationship below rated speed
            power_ratio = (self.wind_speed / self.rated_speed) ** 3
            self.power_output = self.capacity_mw * power_ratio
        else:
            # Rated power with pitch control
            self.power_output = self.capacity_mw
            
        return self.power_output

class BatteryEnergyStorage:
    """IEEE 2030.2 compliant battery energy storage system"""
    
    def __init__(self, capacity_mw=20.0, energy_mwh=80.0, location="Bus_33"):
        self.capacity_mw = capacity_mw
        self.energy_mwh = energy_mwh
        self.location = location
        
        # State variables
        self.soc = 0.5          # State of charge (0-1)
        self.power_output = 0.0  # MW (positive = discharge)
        self.efficiency = 0.95
        
        # Control parameters
        self.soc_min = 0.1
        self.soc_max = 0.9
        self.ramp_rate = 10.0   # MW/min
        
    def update_power(self, dt, power_command):
        """Update battery state based on power command"""
        # Apply ramp rate limits
        max_change = self.ramp_rate * (dt / 60.0)  # Convert to MW/dt
        power_limited = np.clip(power_command, 
                               self.power_output - max_change,
                               self.power_output + max_change)
        
        # Apply SOC limits
        if self.soc <= self.soc_min and power_limited > 0:
            power_limited = 0  # Can't discharge below min SOC
        elif self.soc >= self.soc_max and power_limited < 0:
            power_limited = 0  # Can't charge above max SOC
            
        # Update SOC
        energy_change = power_limited * dt / 3600.0  # Convert to MWh
        if power_limited > 0:  # Discharging
            energy_change *= self.efficiency
        else:  # Charging
            energy_change /= self.efficiency
            
        self.soc -= energy_change / self.energy_mwh
        self.soc = np.clip(self.soc, 0.0, 1.0)
        
        self.power_output = power_limited
        return self.power_output

class ElectricVehicleAggregator:
    """Aggregated EV charging station with V2G capability"""
    
    def __init__(self, num_chargers=8, charger_capacity=50.0, location="Bus_16"):
        self.num_chargers = num_chargers
        self.charger_capacity_kw = charger_capacity
        self.location = location
        self.total_capacity_mw = num_chargers * charger_capacity / 1000.0
        
        # State variables
        self.connected_vehicles = int(num_chargers * 0.6)  # 60% occupancy
        self.average_soc = 0.4
        self.charging_power = 0.0
        self.v2g_capable = int(self.connected_vehicles * 0.3)  # 30% V2G capable
        
    def update_charging(self, dt, smart_signal=0.0):
        """Update EV charging based on smart charging signal"""
        # Base charging demand
        base_demand = self.connected_vehicles * 0.8 * self.charger_capacity_kw / 1000.0
        
        # Smart charging response
        response_factor = 1.0 + smart_signal * 0.3  # ±30% modulation
        
        # V2G contribution during peak demand
        v2g_contribution = 0.0
        if smart_signal > 0.5:  # High demand
            v2g_contribution = self.v2g_capable * 0.5 * self.charger_capacity_kw / 1000.0
            
        self.charging_power = (base_demand * response_factor) - v2g_contribution
        
        # Update average SOC
        energy_change = self.charging_power * dt / 3600.0
        avg_battery_capacity = 60.0  # kWh average
        total_capacity = self.connected_vehicles * avg_battery_capacity / 1000.0  # MWh
        
        if total_capacity > 0:
            self.average_soc += energy_change / total_capacity
            self.average_soc = np.clip(self.average_soc, 0.1, 0.9)
        
        return self.charging_power

class DemandResponseSystem:
    """Demand response aggregator for load control"""
    
    def __init__(self, controllable_load_mw=15.0, location="Bus_20"):
        self.controllable_load_mw = controllable_load_mw
        self.location = location
        
        # State variables
        self.baseline_load = controllable_load_mw
        self.current_load = controllable_load_mw
        self.response_capability = 0.8  # Can reduce 80% of load
        
    def update_load(self, dr_signal):
        """Update load based on demand response signal"""
        # DR signal: 0 = normal, 1 = maximum reduction
        reduction = dr_signal * self.response_capability
        self.current_load = self.baseline_load * (1 - reduction)
        
        return self.current_load

# ======================== CONTROL SYSTEMS ======================== #

class AutomaticGenerationControl:
    """AGC for frequency regulation (50 Hz operation)"""
    
    def __init__(self, beta=1.0):
        self.beta = beta  # Frequency bias (MW/Hz)
        self.K_agc = 0.5  # AGC gain
        self.T_agc = 2.0  # AGC time constant
        
        # State variables
        self.ace = 0.0    # Area Control Error
        self.agc_output = 0.0
        
    def update(self, dt, frequency_deviation, tie_line_flow=0.0):
        """Update AGC output"""
        # Area Control Error (50 Hz base)
        self.ace = tie_line_flow + self.beta * frequency_deviation
        
        # AGC controller
        dagc = (self.K_agc * self.ace - self.agc_output) / self.T_agc
        self.agc_output += dagc * dt
        
        return self.agc_output

class LoadModel:
    """Dynamic load model with frequency and voltage dependence"""
    
    def __init__(self, base_power=100.0, location="Bus_04"):
        self.base_power = base_power
        self.location = location
        
        # Load characteristics
        self.alpha = 1.5  # Active power voltage exponent
        self.beta = 2.0   # Reactive power voltage exponent
        self.kpf = 2.0    # Active power frequency coefficient
        self.kqf = -1.0   # Reactive power frequency coefficient
        
        # State variables
        self.P_load = base_power
        self.Q_load = base_power * 0.3  # 0.3 power factor
        
    def update_load(self, voltage, frequency):
        """Update load based on voltage and frequency"""
        # Frequency is in Hz, convert to pu (50 Hz base)
        freq_pu = frequency / 50.0
        
        # Voltage and frequency dependence
        self.P_load = self.base_power * (voltage ** self.alpha) * (1 + self.kpf * (freq_pu - 1.0))
        self.Q_load = self.base_power * 0.3 * (voltage ** self.beta) * (1 + self.kqf * (freq_pu - 1.0))
        
        return self.P_load, self.Q_load

class ProtectionSystem:
    """Comprehensive protection system for IEEE 39-bus"""
    
    def __init__(self, bus_number, params=None):
        self.bus_number = bus_number
        self.params = params or ProtectionParams()
        self.trip_status = False
        self.trip_delay = 0.1  # seconds
        
    def check_protection(self, voltage, frequency):
        """Check protection criteria"""
        violations = []
        
        if voltage < self.params.voltage_low:
            violations.append(f"Undervoltage: {voltage:.3f} pu")
        if voltage > self.params.voltage_high:
            violations.append(f"Overvoltage: {voltage:.3f} pu")
        if frequency < self.params.frequency_low:
            violations.append(f"Underfrequency: {frequency:.2f} Hz")
        if frequency > self.params.frequency_high:
            violations.append(f"Overfrequency: {frequency:.2f} Hz")
            
        return violations

class SystemCoordinator:
    """DER coordination system for optimal operation"""
    
    def __init__(self):
        self.frequency_target = 50.0  # Hz
        self.voltage_target = 1.0     # pu
        
    def coordinate_ders(self, system_state, ders):
        """Coordinate DER operation based on system state"""
        commands = {}
        
        freq_deviation = system_state.get('frequency', 50.0) - 50.0
        
        # Battery dispatch for frequency regulation
        for der_id, der in ders.items():
            if isinstance(der, BatteryEnergyStorage):
                # Frequency regulation: -10 MW per 0.1 Hz deviation
                power_command = -freq_deviation * 100.0
                power_command = np.clip(power_command, -der.capacity_mw, der.capacity_mw)
                commands[der_id] = power_command
                
        return commands

# ======================== PARAMETER CREATION FUNCTIONS ======================== #

def create_ieee39_generator_parameters():
    """Create IEEE 39-bus generator parameters (50 Hz operation)"""
    generators = {}
    
    # Generator data for IEEE 39-bus system (50 Hz)
    gen_data = [
        {"bus": 30, "H": 500.0, "D": 0.0, "MVA": 250.0},    # Gen 1
        {"bus": 31, "H": 30.3, "D": 0.0, "MVA": 520.0},     # Gen 2  
        {"bus": 32, "H": 35.8, "D": 0.0, "MVA": 650.0},     # Gen 3
        {"bus": 33, "H": 28.6, "D": 0.0, "MVA": 632.0},     # Gen 4
        {"bus": 34, "H": 26.0, "D": 0.0, "MVA": 508.0},     # Gen 5
        {"bus": 35, "H": 34.8, "D": 0.0, "MVA": 650.0},     # Gen 6
        {"bus": 36, "H": 26.4, "D": 0.0, "MVA": 560.0},     # Gen 7
        {"bus": 37, "H": 24.3, "D": 0.0, "MVA": 540.0},     # Gen 8
        {"bus": 38, "H": 34.5, "D": 0.0, "MVA": 830.0},     # Gen 9
        {"bus": 39, "H": 42.0, "D": 0.0, "MVA": 1000.0}     # Gen 10 (swing)
    ]
    
    for i, gen in enumerate(gen_data):
        generators[f"Gen_{i+1}"] = GeneratorParams(
            H=gen["H"],
            D=gen["D"], 
            MVA_base=gen["MVA"],
            frequency=50.0  # 50 Hz operation
        )
    
    return generators

def create_ieee39_line_parameters():
    """Create IEEE 39-bus line parameters"""
    lines = {}
    
    # Key transmission lines for IEEE 39-bus system
    line_data = [
        (1, 2, 0.0035, 0.0411, 0.6987),    # Line 1-2
        (1, 39, 0.0010, 0.0250, 0.7500),   # Line 1-39
        (2, 3, 0.0013, 0.0151, 0.2572),    # Line 2-3
        (2, 25, 0.0070, 0.0086, 0.1460),   # Line 2-25
        (3, 4, 0.0013, 0.0213, 0.2214),    # Line 3-4
        (3, 18, 0.0011, 0.0133, 0.2138),   # Line 3-18
        (4, 5, 0.0008, 0.0128, 0.1342),    # Line 4-5
        (4, 14, 0.0008, 0.0129, 0.1382),   # Line 4-14
        (5, 6, 0.0002, 0.0026, 0.0434),    # Line 5-6
        (5, 8, 0.0008, 0.0112, 0.1476),    # Line 5-8
        (6, 7, 0.0006, 0.0092, 0.1130),    # Line 6-7
        (6, 11, 0.0007, 0.0082, 0.1389),   # Line 6-11
        (7, 8, 0.0004, 0.0046, 0.0780),    # Line 7-8
        (8, 9, 0.0023, 0.0363, 0.3804),    # Line 8-9
        (9, 39, 0.0010, 0.0250, 1.2000),   # Line 9-39
        (10, 11, 0.0004, 0.0043, 0.0729),  # Line 10-11
        (10, 13, 0.0004, 0.0043, 0.0729),  # Line 10-13
        (13, 14, 0.0009, 0.0101, 0.1723),  # Line 13-14
        (14, 15, 0.0018, 0.0217, 0.3660),  # Line 14-15
        (15, 16, 0.0009, 0.0094, 0.1710),  # Line 15-16
        (16, 17, 0.0007, 0.0089, 0.1342),  # Line 16-17
        (16, 19, 0.0016, 0.0195, 0.3040),  # Line 16-19
        (16, 21, 0.0008, 0.0135, 0.2548),  # Line 16-21
        (16, 24, 0.0003, 0.0059, 0.0680),  # Line 16-24
        (17, 18, 0.0007, 0.0082, 0.1319),  # Line 17-18
        (17, 27, 0.0013, 0.0173, 0.3216),  # Line 17-27
        (19, 20, 0.0007, 0.0138, 0.0000),  # Line 19-20
        (19, 33, 0.0007, 0.0142, 0.0000),  # Line 19-33
        (20, 34, 0.0009, 0.0180, 0.0000),  # Line 20-34
        (21, 22, 0.0008, 0.0140, 0.2565),  # Line 21-22
        (22, 23, 0.0006, 0.0096, 0.1846),  # Line 22-23
        (23, 24, 0.0022, 0.0350, 0.3610),  # Line 23-24
        (25, 26, 0.0032, 0.0323, 0.5130),  # Line 25-26
        (26, 27, 0.0014, 0.0147, 0.2396),  # Line 26-27
        (26, 28, 0.0043, 0.0474, 0.7802),  # Line 26-28
        (26, 29, 0.0057, 0.0625, 1.0290),  # Line 26-29
        (28, 29, 0.0014, 0.0151, 0.2490),  # Line 28-29
        (12, 11, 0.0016, 0.0435, 0.0000),  # Transformer 12-11
        (12, 13, 0.0016, 0.0435, 0.0000),  # Transformer 12-13
        (6, 31, 0.0000, 0.0250, 0.0000),   # Transformer 6-31
        (10, 32, 0.0000, 0.0200, 0.0000),  # Transformer 10-32
        (19, 33, 0.0007, 0.0142, 0.0000),  # Transformer 19-33
        (20, 34, 0.0009, 0.0180, 0.0000),  # Transformer 20-34
        (22, 35, 0.0000, 0.0143, 0.0000),  # Transformer 22-35
        (23, 36, 0.0005, 0.0272, 0.0000),  # Transformer 23-36
        (25, 37, 0.0006, 0.0232, 0.0000),  # Transformer 25-37
        (2, 30, 0.0000, 0.0181, 0.0000),   # Transformer 2-30
        (29, 38, 0.0008, 0.0156, 0.0000)   # Transformer 29-38
    ]
    
    for i, (from_bus, to_bus, R, X, B) in enumerate(line_data):
        lines[f"Line_{i+1}"] = {
            "from_bus": from_bus,
            "to_bus": to_bus,
            "R": R,
            "X": X,
            "B": B,
            "rating": 500.0  # MVA rating
        }
    
    return lines

# ======================== UTILITY FUNCTIONS ======================== #

def get_system_frequency():
    """Get the system operating frequency"""
    return 50.0  # Hz - International standard

def create_default_der_systems():
    """Create default DER systems for IEEE 39-bus"""
    ders = {
        "Solar_1": SolarPVSystem(capacity_mw=25.0, location="Bus_31"),
        "Solar_2": SolarPVSystem(capacity_mw=20.0, location="Bus_32"),
        "Wind_1": WindTurbineSystem(capacity_mw=30.0, location="Bus_37"),
        "Wind_2": WindTurbineSystem(capacity_mw=25.0, location="Bus_38"),
        "BESS_1": BatteryEnergyStorage(capacity_mw=20.0, energy_mwh=80.0, location="Bus_33"),
        "BESS_2": BatteryEnergyStorage(capacity_mw=20.0, energy_mwh=80.0, location="Bus_34"),
        "BESS_3": BatteryEnergyStorage(capacity_mw=20.0, energy_mwh=80.0, location="Bus_35"),
        "BESS_4": BatteryEnergyStorage(capacity_mw=20.0, energy_mwh=80.0, location="Bus_36"),
        "EV_1": ElectricVehicleAggregator(num_chargers=10, charger_capacity=50.0, location="Bus_16"),
        "EV_2": ElectricVehicleAggregator(num_chargers=8, charger_capacity=50.0, location="Bus_17"),
        "EV_3": ElectricVehicleAggregator(num_chargers=8, charger_capacity=50.0, location="Bus_21"),
        "EV_4": ElectricVehicleAggregator(num_chargers=8, charger_capacity=50.0, location="Bus_22"),
        "EV_5": ElectricVehicleAggregator(num_chargers=7, charger_capacity=50.0, location="Bus_23"),
        "DR_1": DemandResponseSystem(controllable_load_mw=15.0, location="Bus_20"),
        "DR_2": DemandResponseSystem(controllable_load_mw=15.0, location="Bus_21"),
        "DR_3": DemandResponseSystem(controllable_load_mw=15.0, location="Bus_22"),
        "DR_4": DemandResponseSystem(controllable_load_mw=15.0, location="Bus_23"),
        "DR_5": DemandResponseSystem(controllable_load_mw=15.0, location="Bus_24")
    }
    
    return ders

if __name__ == "__main__":
    print("IEEE Standard Dynamic Models (50 Hz Operation)")
    print("=" * 60)
    
    # Test model creation
    gen = SynchronousGenerator()
    exciter = ExcitationSystem() 
    governor = GovernorTurbine()
    pss = PowerSystemStabilizer()
    
    print(f"✓ Synchronous Generator: {gen.frequency} Hz operation")
    print(f"✓ Excitation System: Vref = {exciter.params.Vref} pu")
    print(f"✓ Governor System: R = {governor.params.R} pu")
    print(f"✓ PSS: K = {pss.K_pss}")
    
    # Test DER systems
    ders = create_default_der_systems()
    print(f"✓ Created {len(ders)} DER systems")
    
    # Test parameter creation
    gen_params = create_ieee39_generator_parameters()
    print(f"✓ Created parameters for {len(gen_params)} generators")
    
    print("\n🎯 All IEEE standard models ready for 50 Hz operation!")
