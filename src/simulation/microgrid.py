"""
Microgrid Simulation Module
Integrated from files/ and somu_test/ into dems framework

Provides:
- PyPower-compatible microgrid case structure
- Newton-Raphson power flow solver (standalone, no pandapower dependency)
- DER component models (Solar, Wind, Battery, Diesel)
- Microgrid controller with droop and SOC management
- 24-hour time-series simulation capability

This module complements the pandapower-based SuperGrid simulation
by providing a lightweight, self-contained microgrid simulator
for smaller distribution networks and islanded microgrids.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import logging

try:
    from scipy.sparse import csc_matrix
    from scipy.sparse.linalg import spsolve
    _HAS_SCIPY_SPARSE = True
except ImportError:
    _HAS_SCIPY_SPARSE = False

logger = logging.getLogger(__name__)

# ======================== PYPOWER-COMPATIBLE CONSTANTS ======================== #

# Bus array columns
BUS_I = 0       # bus number
BUS_TYPE = 1    # bus type (1=PQ, 2=PV, 3=ref, 4=isolated)
PD = 2          # real power demand (MW)
QD = 3          # reactive power demand (MVAr)
GS = 4          # shunt conductance
BS = 5          # shunt susceptance
BUS_AREA = 6    # area number
VM = 7          # voltage magnitude (p.u.)
VA = 8          # voltage angle (degrees)
BASE_KV = 9     # base voltage (kV)
ZONE = 10       # loss zone
VMAX = 11       # maximum voltage magnitude (p.u.)
VMIN = 12       # minimum voltage magnitude (p.u.)

# Generator array columns
GEN_BUS = 0
PG = 1
QG = 2
QMAX = 3
QMIN = 4
VG = 5
MBASE = 6
GEN_STATUS = 7
PMAX = 8
PMIN = 9

# Branch array columns
F_BUS = 0
T_BUS = 1
BR_R = 2
BR_X = 3
BR_B = 4
RATE_A = 5
RATE_B = 6
RATE_C = 7
TAP = 8
SHIFT = 9
BR_STATUS = 10
ANGMIN = 11
ANGMAX = 12

# Bus types
PQ_BUS = 1
PV_BUS = 2
REF_BUS = 3
ISOLATED = 4


# ======================== MICROGRID CASE ======================== #

class MicrogridCase:
    """
    PyPower-compatible case structure for microgrid power flow.

    Stores bus, generator, and branch arrays in numpy format compatible
    with standard power system analysis tools.
    """

    def __init__(self, baseMVA: float = 100.0):
        self.baseMVA = baseMVA
        self.bus = np.array([])
        self.gen = np.array([])
        self.branch = np.array([])
        self.gencost = np.array([])
        self.version = '2'

    @property
    def bus_count(self) -> int:
        return len(self.bus)

    @property
    def gen_count(self) -> int:
        return len(self.gen)

    @property
    def branch_count(self) -> int:
        return len(self.branch)


# ======================== DER COMPONENT MODELS ======================== #

class DERComponent:
    """Base class for Distributed Energy Resources"""

    def __init__(self, name: str, bus: int, rated_power: float):
        self.name = name
        self.bus = bus
        self.rated_power = rated_power  # MW
        self.status = 1  # 1 = online, 0 = offline

    def get_output(self, time_step: float) -> Tuple[float, float]:
        """Return (P_MW, Q_MVAr) at given time step"""
        raise NotImplementedError


class SolarPV(DERComponent):
    """Solar PV system with irradiance-based output model"""

    def __init__(self, name: str, bus: int, rated_power: float,
                 efficiency: float = 0.95, power_factor: float = 0.95):
        super().__init__(name, bus, rated_power)
        self.efficiency = efficiency
        self.power_factor = power_factor
        self.irradiance_profile: Optional[np.ndarray] = None

    def set_irradiance_profile(self, profile: np.ndarray):
        """Set solar irradiance profile (0-1 normalized)"""
        self.irradiance_profile = profile

    def get_output(self, time_step: float) -> Tuple[float, float]:
        """Calculate PV output based on irradiance"""
        if self.irradiance_profile is None:
            hour = time_step % 24
            irradiance = np.sin(np.pi * (hour - 6) / 12) if 6 <= hour <= 18 else 0.0
        else:
            idx = int(time_step) % len(self.irradiance_profile)
            irradiance = self.irradiance_profile[idx]

        P = self.rated_power * max(0, irradiance) * self.efficiency * self.status
        Q = P * np.tan(np.arccos(self.power_factor))
        return P, Q


class WindTurbine(DERComponent):
    """Wind turbine with cubic power curve model"""

    def __init__(self, name: str, bus: int, rated_power: float,
                 cut_in_speed: float = 3.0, rated_speed: float = 12.0,
                 cut_out_speed: float = 25.0, power_factor: float = 0.95):
        super().__init__(name, bus, rated_power)
        self.cut_in_speed = cut_in_speed
        self.rated_speed = rated_speed
        self.cut_out_speed = cut_out_speed
        self.power_factor = power_factor
        self.wind_speed_profile: Optional[np.ndarray] = None

    def set_wind_speed_profile(self, profile: np.ndarray):
        """Set wind speed profile (m/s)"""
        self.wind_speed_profile = profile

    def get_output(self, time_step: float) -> Tuple[float, float]:
        """Calculate wind output based on wind speed"""
        if self.wind_speed_profile is None:
            wind_speed = 8 + 4 * np.sin(2 * np.pi * time_step / 24)
        else:
            idx = int(time_step) % len(self.wind_speed_profile)
            wind_speed = self.wind_speed_profile[idx]

        if wind_speed < self.cut_in_speed or wind_speed > self.cut_out_speed:
            P = 0.0
        elif wind_speed >= self.rated_speed:
            P = self.rated_power
        else:
            # FIX BUG-16: Correct IEC 61400 cubic formula
            v, v_ci, v_r = wind_speed, self.cut_in_speed, self.rated_speed
            P = self.rated_power * (v**3 - v_ci**3) / (v_r**3 - v_ci**3)

        P *= self.status
        Q = P * np.tan(np.arccos(self.power_factor))
        return P, Q


class BatteryESS(DERComponent):
    """Battery Energy Storage System with SOC tracking"""

    def __init__(self, name: str, bus: int, rated_power: float,
                 capacity: float, initial_soc: float = 0.5,
                 min_soc: float = 0.1, max_soc: float = 0.9,
                 charge_efficiency: float = 0.95,
                 discharge_efficiency: float = 0.95):
        super().__init__(name, bus, rated_power)
        self.capacity = capacity  # MWh
        self.soc = initial_soc
        self.min_soc = min_soc
        self.max_soc = max_soc
        self.charge_efficiency = charge_efficiency
        self.discharge_efficiency = discharge_efficiency
        self.mode = 'idle'  # 'charging', 'discharging', 'idle'
        self._last_power = 0.0

    def set_power_command(self, P_command: float, dt: float = 1.0) -> float:
        """
        Set power command and update SOC.

        Args:
            P_command: Power in MW (positive=discharge, negative=charge)
            dt: Time step in hours

        Returns:
            Actual power delivered/absorbed (MW)
        """
        if P_command > 0:  # Discharging
            P_actual = min(P_command, self.rated_power)
            P_actual = min(P_actual, (self.soc - self.min_soc) * self.capacity / dt)
            self.soc -= (P_actual * dt) / (self.capacity * self.discharge_efficiency)
            self.mode = 'discharging'
        elif P_command < 0:  # Charging
            P_actual = max(P_command, -self.rated_power)
            P_actual = max(P_actual, -((self.max_soc - self.soc) * self.capacity / dt))
            self.soc -= (P_actual * dt * self.charge_efficiency) / self.capacity
            self.mode = 'charging'
        else:
            P_actual = 0.0
            self.mode = 'idle'

        self.soc = np.clip(self.soc, self.min_soc, self.max_soc)
        self._last_power = P_actual
        return P_actual

    def get_output(self, time_step: float) -> Tuple[float, float]:
        """Get current battery power output"""
        return self._last_power * self.status, 0.0


class DieselGenerator(DERComponent):
    """Diesel/backup generator with ramp rate limits"""

    def __init__(self, name: str, bus: int, rated_power: float,
                 min_power_frac: float = 0.3, ramp_rate: float = 0.5,
                 fuel_curve_coeffs: Optional[List[float]] = None):
        super().__init__(name, bus, rated_power)
        self.min_power = min_power_frac * rated_power
        self.ramp_rate = ramp_rate  # MW/min
        self.current_output = 0.0
        self.fuel_curve = fuel_curve_coeffs or [0.5, 0.3, 0.01]

    def set_output(self, P_target: float, dt: float = 1.0) -> float:
        """Set output considering ramp rate limits.
        
        Args:
            P_target: Target output (MW)
            dt: Time step in hours
            
        FIX BUG-17: ramp_rate is MW/min, dt is in hours.
        Convert dt to minutes for correct ramp calculation.
        """
        if self.status == 0:
            self.current_output = 0.0
            return 0.0

        max_change = self.ramp_rate * (dt * 60.0)  # MW/min × minutes
        P_new = np.clip(P_target,
                        self.current_output - max_change,
                        self.current_output + max_change)

        if P_new > 0:
            P_new = np.clip(P_new, self.min_power, self.rated_power)
        else:
            P_new = 0.0

        self.current_output = P_new
        return P_new

    def get_output(self, time_step: float) -> Tuple[float, float]:
        P = self.current_output * self.status
        Q = P * 0.1
        return P, Q

    def get_fuel_consumption(self) -> float:
        """Calculate fuel consumption (L/h)"""
        P_norm = self.current_output / self.rated_power if self.rated_power > 0 else 0
        a, b, c = self.fuel_curve
        return (a + b * P_norm + c * P_norm ** 2) * self.rated_power


# ======================== POWER FLOW SOLVER ======================== #

class PowerFlowSolver:
    """
    Newton-Raphson power flow solver (PyPower-compatible).

    Solves AC power flow for MicrogridCase objects using full
    Newton-Raphson with Jacobian construction.
    """

    def __init__(self, case: MicrogridCase, tolerance: float = 1e-6,
                 max_iter: int = 20):
        self.case = case
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.results: Optional[Dict] = None

    def run(self) -> Dict:
        """Run Newton-Raphson power flow"""
        bus = self.case.bus.copy()
        gen = self.case.gen.copy()
        branch = self.case.branch.copy()
        baseMVA = self.case.baseMVA

        nb = len(bus)
        ng = len(gen)

        # Build admittance matrix
        Ybus = self._build_ybus(bus, branch, baseMVA)

        # Initialize voltage
        V = bus[:, VM] * np.exp(1j * np.deg2rad(bus[:, VA]))

        # Bus classification
        ref_bus = np.where(bus[:, BUS_TYPE] == REF_BUS)[0]
        pv_buses = np.where(bus[:, BUS_TYPE] == PV_BUS)[0]
        pq_buses = np.where(bus[:, BUS_TYPE] == PQ_BUS)[0]

        if len(ref_bus) == 0:
            raise ValueError("No reference bus defined")
        ref_bus = ref_bus[0]

        # Generator bus mapping
        gen_bus_map = {int(gen[i, GEN_BUS]): i for i in range(ng)}

        # Net injections (load convention: negative = load)
        Sbus = np.zeros(nb, dtype=complex)
        Sbus -= (bus[:, PD] + 1j * bus[:, QD]) / baseMVA

        for i in range(ng):
            bus_idx = int(gen[i, GEN_BUS])
            if gen[i, GEN_STATUS] > 0:
                Sbus[bus_idx] += (gen[i, PG] + 1j * gen[i, QG]) / baseMVA

        # Newton-Raphson iteration
        converged = False
        iteration = 0
        for iteration in range(self.max_iter):
            Scalc = V * np.conj(Ybus @ V)
            mis = Sbus - Scalc

            pv_pq = np.concatenate([pv_buses, pq_buses])
            P_mis = mis[pv_pq].real
            Q_mis = mis[pq_buses].imag

            normP = np.linalg.norm(P_mis, np.inf)
            normQ = np.linalg.norm(Q_mis, np.inf) if len(Q_mis) > 0 else 0.0

            if normP < self.tolerance and normQ < self.tolerance:
                converged = True
                break

            J = self._build_jacobian(Ybus, V, pv_pq, pq_buses)

            mismatch = np.concatenate([P_mis, Q_mis])
            if J.shape[0] != len(mismatch):
                break
            try:
                # FIX BUG-20: Use sparse solver when available for O(n) scalability
                if _HAS_SCIPY_SPARSE and J.shape[0] > 10:
                    J_sparse = csc_matrix(J)
                    dx = spsolve(J_sparse, mismatch)
                else:
                    dx = np.linalg.solve(J, mismatch)
            except (np.linalg.LinAlgError, Exception):
                break

            npv_pq = len(pv_pq)
            dVa = np.zeros(nb)
            dVm = np.zeros(nb)

            dVa[pv_pq] = dx[:npv_pq]
            dVm[pq_buses] = dx[npv_pq:]

            # Update voltage from current V (not initial bus values)
            Va_new = np.angle(V) + dVa
            Vm_new = np.abs(V) + dVm

            # PV buses: hold voltage magnitude at setpoint
            for pv in pv_buses:
                Vm_new[pv] = bus[pv, VM]
            # Ref bus: hold both magnitude and angle
            Vm_new[ref_bus] = bus[ref_bus, VM]
            Va_new[ref_bus] = np.deg2rad(bus[ref_bus, VA])

            V = Vm_new * np.exp(1j * Va_new)

        # Update results in bus array
        bus[:, VM] = np.abs(V)
        bus[:, VA] = np.rad2deg(np.angle(V))

        # Update generator reactive power
        Sbus_final = V * np.conj(Ybus @ V) * baseMVA
        for i in pv_buses:
            if i in gen_bus_map:
                gen[gen_bus_map[i], QG] = Sbus_final[i].imag + bus[i, QD]

        if ref_bus in gen_bus_map:
            gen[gen_bus_map[ref_bus], PG] = Sbus_final[ref_bus].real + bus[ref_bus, PD]
            gen[gen_bus_map[ref_bus], QG] = Sbus_final[ref_bus].imag + bus[ref_bus, QD]

        # Branch flows
        branch_flows = self._calculate_branch_flows(branch, bus, V, baseMVA)

        self.results = {
            'converged': converged,
            'iterations': iteration + 1,
            'bus': bus,
            'gen': gen,
            'branch_flows': branch_flows,
            'V': V,
            'Ybus': Ybus
        }
        return self.results

    def _build_ybus(self, bus: np.ndarray, branch: np.ndarray,
                    baseMVA: float) -> np.ndarray:
        """Build bus admittance matrix"""
        nb = len(bus)
        Ybus = np.zeros((nb, nb), dtype=complex)

        for k in range(len(branch)):
            if branch[k, BR_STATUS] == 0:
                continue
            f = int(branch[k, F_BUS])
            t = int(branch[k, T_BUS])
            r, x, b = branch[k, BR_R], branch[k, BR_X], branch[k, BR_B]

            if r == 0 and x == 0:
                continue

            y = 1 / (r + 1j * x)
            ysh = 1j * b / 2
            tap = branch[k, TAP] if branch[k, TAP] != 0 else 1.0
            shift = np.deg2rad(branch[k, SHIFT])
            tap_complex = tap * np.exp(1j * shift)

            Ytt = y + ysh
            Yff = Ytt / (tap * tap)
            Yft = -y / np.conj(tap_complex)
            Ytf = -y / tap_complex

            Ybus[f, f] += Yff
            Ybus[t, t] += Ytt
            Ybus[f, t] += Yft
            Ybus[t, f] += Ytf

        for i in range(nb):
            Ybus[i, i] += (bus[i, GS] + 1j * bus[i, BS]) / baseMVA

        return Ybus

    def _build_jacobian(self, Ybus: np.ndarray, V: np.ndarray,
                        pv_pq: np.ndarray, pq: np.ndarray) -> np.ndarray:
        """Build Jacobian matrix for Newton-Raphson"""
        Vm = np.abs(V)
        Va = np.angle(V)
        G = Ybus.real
        B = Ybus.imag

        n_pvpq = len(pv_pq)
        n_pq = len(pq)

        J11 = np.zeros((n_pvpq, n_pvpq))
        J12 = np.zeros((n_pvpq, n_pq))
        J21 = np.zeros((n_pq, n_pvpq))
        J22 = np.zeros((n_pq, n_pq))

        for i, bi in enumerate(pv_pq):
            for j, bj in enumerate(pv_pq):
                if i == j:
                    J11[i, j] = -np.sum(
                        Vm[bi] * Vm * (G[bi, :] * np.sin(Va[bi] - Va) -
                                       B[bi, :] * np.cos(Va[bi] - Va)))
                    J11[i, j] -= Vm[bi] ** 2 * B[bi, bi]
                else:
                    J11[i, j] = Vm[bi] * Vm[bj] * (
                        G[bi, bj] * np.sin(Va[bi] - Va[bj]) -
                        B[bi, bj] * np.cos(Va[bi] - Va[bj]))

        for i, bi in enumerate(pv_pq):
            for j, bj in enumerate(pq):
                if bi == bj:
                    J12[i, j] = np.sum(
                        Vm * (G[bi, :] * np.cos(Va[bi] - Va) +
                              B[bi, :] * np.sin(Va[bi] - Va)))
                    J12[i, j] += Vm[bi] * G[bi, bi]
                else:
                    J12[i, j] = Vm[bi] * (
                        G[bi, bj] * np.cos(Va[bi] - Va[bj]) +
                        B[bi, bj] * np.sin(Va[bi] - Va[bj]))

        for i, bi in enumerate(pq):
            for j, bj in enumerate(pv_pq):
                if bi == bj:
                    J21[i, j] = np.sum(
                        Vm[bi] * Vm * (G[bi, :] * np.cos(Va[bi] - Va) +
                                       B[bi, :] * np.sin(Va[bi] - Va)))
                    J21[i, j] -= Vm[bi] ** 2 * G[bi, bi]
                else:
                    J21[i, j] = -Vm[bi] * Vm[bj] * (
                        G[bi, bj] * np.cos(Va[bi] - Va[bj]) +
                        B[bi, bj] * np.sin(Va[bi] - Va[bj]))

        for i, bi in enumerate(pq):
            for j, bj in enumerate(pq):
                if i == j:
                    J22[i, j] = np.sum(
                        Vm * (G[bi, :] * np.sin(Va[bi] - Va) -
                              B[bi, :] * np.cos(Va[bi] - Va)))
                    J22[i, j] -= Vm[bi] * B[bi, bi]
                else:
                    J22[i, j] = Vm[bi] * (
                        G[bi, bj] * np.sin(Va[bi] - Va[bj]) -
                        B[bi, bj] * np.cos(Va[bi] - Va[bj]))

        return np.vstack([
            np.hstack([J11, J12]),
            np.hstack([J21, J22])
        ])

    def _calculate_branch_flows(self, branch: np.ndarray, bus: np.ndarray,
                                V: np.ndarray, baseMVA: float) -> np.ndarray:
        """Calculate power flows in branches"""
        nl = len(branch)
        flows = np.zeros((nl, 8))

        for k in range(nl):
            if branch[k, BR_STATUS] == 0:
                continue
            f, t = int(branch[k, F_BUS]), int(branch[k, T_BUS])
            r, x, b = branch[k, BR_R], branch[k, BR_X], branch[k, BR_B]

            if r == 0 and x == 0:
                continue

            y = 1 / (r + 1j * x)
            ysh = 1j * b / 2
            tap = branch[k, TAP] if branch[k, TAP] != 0 else 1.0
            shift = np.deg2rad(branch[k, SHIFT])

            Vf, Vt_bus = V[f], V[t]
            If = y * (Vf / tap - Vt_bus / tap * np.exp(-1j * shift)) + ysh * Vf / tap
            Sf = Vf * np.conj(If) * baseMVA
            It = y * (Vt_bus - Vf / tap * np.exp(1j * shift)) + ysh * Vt_bus
            St = Vt_bus * np.conj(It) * baseMVA

            flows[k, :] = [Sf.real, Sf.imag, St.real, St.imag,
                           (Sf + St).real, (Sf + St).imag,
                           np.abs(Sf), np.abs(St)]
        return flows


# ======================== MICROGRID CONTROLLER ======================== #

class MicrogridController:
    """
    Centralized microgrid controller with droop, voltage control,
    and battery SOC management.
    """

    def __init__(self, case: MicrogridCase, ders: Dict[str, DERComponent]):
        self.case = case
        self.ders = ders
        self.mode = 'grid_connected'
        self.frequency = 50.0  # Hz (50 Hz for Indian grid)
        self.voltage_ref = 1.0

    def droop_control(self, P_measured: float, f_measured: float,
                      P_ref: float, f_ref: float, droop_coeff: float) -> float:
        """P-f droop control: P_new = P_ref - droop × (f_measured - f_ref)"""
        return P_ref - droop_coeff * (f_measured - f_ref)

    def voltage_control(self, Q_measured: float, V_measured: float,
                        Q_ref: float, V_ref: float, droop_coeff: float) -> float:
        """Q-V droop control"""
        return Q_ref - droop_coeff * (V_measured - V_ref)

    def battery_soc_control(self, battery: BatteryESS, P_net: float,
                            dt: float = 1.0) -> float:
        """
        Battery SOC management.
        Positive P_net = excess generation (charge battery)
        Negative P_net = deficit (discharge battery)
        """
        if P_net > 0.1:
            P_command = -min(P_net, battery.rated_power)
        elif P_net < -0.1:
            P_command = min(abs(P_net), battery.rated_power)
        else:
            if battery.soc < 0.45:
                P_command = -battery.rated_power * 0.2
            elif battery.soc > 0.55:
                P_command = battery.rated_power * 0.2
            else:
                P_command = 0.0
        return battery.set_power_command(P_command, dt)

    def update_generation(self, time_step: float):
        """Update all DER outputs and apply to case gen array"""
        gen = self.case.gen
        for name, der in self.ders.items():
            P, Q = der.get_output(time_step)
            gen_idx = np.where(gen[:, GEN_BUS] == der.bus)[0]
            if len(gen_idx) > 0:
                gen[gen_idx[0], PG] = P
                gen[gen_idx[0], QG] = Q
                gen[gen_idx[0], GEN_STATUS] = der.status


# ======================== CASE FACTORY ======================== #

def create_example_microgrid() -> Tuple[MicrogridCase, Dict[str, DERComponent]]:
    """
    Create an example 5-bus microgrid with DERs.

    Topology:
    - Bus 0: Main grid connection (slack bus)
    - Bus 1: Solar PV + residential load
    - Bus 2: Wind turbine + commercial load
    - Bus 3: Battery ESS + industrial load
    - Bus 4: Diesel generator + load
    """
    case = MicrogridCase(baseMVA=100.0)

    case.bus = np.array([
        [0, REF_BUS, 0.0, 0.0, 0, 0, 1, 1.00, 0.0, 11.0, 1, 1.05, 0.95],
        [1, PQ_BUS,  2.0, 0.5, 0, 0, 1, 1.00, 0.0, 11.0, 1, 1.05, 0.95],
        [2, PQ_BUS,  3.0, 0.8, 0, 0, 1, 1.00, 0.0, 11.0, 1, 1.05, 0.95],
        [3, PQ_BUS,  4.0, 1.0, 0, 0, 1, 1.00, 0.0, 11.0, 1, 1.05, 0.95],
        [4, PQ_BUS,  1.5, 0.4, 0, 0, 1, 1.00, 0.0, 11.0, 1, 1.05, 0.95],
    ])

    case.gen = np.array([
        [0, 0.0, 0.0, 10.0, -10.0, 1.00, 100, 1, 50.0, 0.0],
        [1, 2.5, 0.5,  2.0,  -1.0, 1.00, 100, 1,  3.0, 0.0],
        [2, 2.0, 0.4,  1.5,  -1.0, 1.00, 100, 1,  2.5, 0.0],
        [3, 0.0, 0.0,  1.0,  -1.0, 1.00, 100, 1,  2.0, -2.0],
        [4, 0.0, 0.0,  1.5,  -0.5, 1.00, 100, 1,  3.0, 0.5],
    ])

    case.branch = np.array([
        [0, 1, 0.001, 0.005, 0.002, 10, 12, 15, 0, 0, 1, -360, 360],
        [0, 2, 0.002, 0.006, 0.003, 10, 12, 15, 0, 0, 1, -360, 360],
        [1, 3, 0.0015, 0.0055, 0.0025, 8, 10, 12, 0, 0, 1, -360, 360],
        [2, 4, 0.0012, 0.0048, 0.002, 8, 10, 12, 0, 0, 1, -360, 360],
        [3, 4, 0.001, 0.004, 0.0015, 6, 8, 10, 0, 0, 1, -360, 360],
    ])

    ders = {
        'solar_pv': SolarPV('Solar_PV_1', bus=1, rated_power=3.0, efficiency=0.95),
        'wind_turbine': WindTurbine('Wind_1', bus=2, rated_power=2.5),
        'battery': BatteryESS('BESS_1', bus=3, rated_power=2.0, capacity=4.0),
        'diesel': DieselGenerator('Diesel_1', bus=4, rated_power=3.0)
    }

    return case, ders


def create_ieee14_case() -> MicrogridCase:
    """Create simplified IEEE 14-bus case"""
    case = MicrogridCase(baseMVA=100.0)

    case.bus = np.array([
        [0, REF_BUS, 0.0, 0.0, 0, 0, 1, 1.06, 0, 138, 1, 1.1, 0.9],
        [1, PV_BUS, 21.7, 12.7, 0, 0, 1, 1.045, 0, 138, 1, 1.1, 0.9],
        [2, PQ_BUS, 94.2, 19.0, 0, 0, 1, 1.0, 0, 138, 1, 1.1, 0.9],
        [3, PQ_BUS, 47.8, -3.9, 0, 0, 1, 1.0, 0, 138, 1, 1.1, 0.9],
        [4, PQ_BUS, 7.6, 1.6, 0, 0, 1, 1.0, 0, 138, 1, 1.1, 0.9],
    ])

    case.gen = np.array([
        [0, 232.4, -16.9, 10, -40, 1.06, 100, 1, 332.4, 0],
        [1, 40.0, 42.4, 50, -40, 1.045, 100, 1, 140.0, 0],
    ])

    case.branch = np.array([
        [0, 1, 0.01938, 0.05917, 0.0528, 250, 250, 250, 0, 0, 1, -360, 360],
        [0, 4, 0.05403, 0.22304, 0.0492, 150, 150, 150, 0, 0, 1, -360, 360],
        [1, 2, 0.04699, 0.19797, 0.0438, 180, 180, 180, 0, 0, 1, -360, 360],
        [1, 3, 0.05811, 0.17632, 0.034, 130, 130, 130, 0, 0, 1, -360, 360],
        [2, 3, 0.06701, 0.17103, 0.0346, 150, 150, 150, 0, 0, 1, -360, 360],
    ])

    return case
