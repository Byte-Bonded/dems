"""
DEMS Simulation Engine
Wraps the grid orchestrator, DER manager, and dynamics coordinator
to provide a unified stepping interface for the Streamlit dashboard.

Supports two modes:
  - "embedded": Imports src modules directly and runs simulation in-process
  - "api": Connects to a running FastAPI backend via HTTP
"""

import sys
import os
import time
import math
import random
import logging
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is on path
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.grid import DEMSGrid
from src.simulation.supergrid import SuperGridConfig, AreaID
from src.simulation.power_flow import PowerFlowRunner, PowerFlowResult
from src.simulation.der import DERManager, DERType, DERState
from src.simulation.dynamics import DynamicsCoordinator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class SimSnapshot:
    """A single point-in-time capture of the full simulation state."""
    step: int = 0
    timestamp: float = 0.0  # wall-clock seconds since engine start
    sim_hour: float = 0.0   # simulated hour of day (0-23.99)

    # Power flow
    converged: bool = False
    total_generation_mw: float = 0.0
    total_load_mw: float = 0.0
    total_losses_mw: float = 0.0
    min_voltage_pu: float = 1.0
    max_voltage_pu: float = 1.0
    avg_voltage_pu: float = 1.0
    num_voltage_violations: int = 0
    num_line_overloads: int = 0
    is_secure: bool = True

    # Frequency / dynamics
    system_frequency_hz: float = 50.0
    generator_frequencies: Dict[str, float] = field(default_factory=dict)
    generator_angles: Dict[str, float] = field(default_factory=dict)
    agc_adjustments: Dict[str, Dict[str, float]] = field(default_factory=dict)
    protection_status: Dict[str, Dict] = field(default_factory=dict)

    # Area metrics
    area_states: Dict[str, Dict] = field(default_factory=dict)

    # Tie-line flows
    tie_line_flows: List[Dict] = field(default_factory=list)

    # DER
    der_states: List[Dict] = field(default_factory=list)
    der_summary: Dict[str, Dict] = field(default_factory=dict)

    # Bus-level detail (kept lightweight – only voltages & angles)
    bus_voltages: Dict[int, float] = field(default_factory=dict)
    bus_angles: Dict[int, float] = field(default_factory=dict)

    # Line-level detail
    line_results: List[Dict] = field(default_factory=list)

    # Generator-level detail
    gen_results: List[Dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Weather / scenario helpers
# ---------------------------------------------------------------------------
def _solar_irradiance_profile(hour: float) -> float:
    """Simple bell-curve irradiance peaking at noon, 0 outside 6–18h."""
    if hour < 6.0 or hour > 18.0:
        return 0.0
    norm = (hour - 6.0) / 12.0  # 0..1
    return 1000.0 * math.sin(math.pi * norm)  # W/m² peak ~1000


def _wind_speed_profile(hour: float) -> float:
    """Synthetic diurnal wind profile – stronger at night."""
    base = 8.0
    variation = 4.0 * math.sin(2 * math.pi * (hour / 24.0 - 0.25))
    noise = random.gauss(0, 0.8)
    return max(0.0, base + variation + noise)


# ---------------------------------------------------------------------------
# Main simulation engine
# ---------------------------------------------------------------------------
class SimEngine:
    """Drives the DEMS grid simulation, exposing a simple step() interface."""

    MAX_HISTORY = 500

    def __init__(self) -> None:
        logger.info("SimEngine: building grid …")
        self.grid = DEMSGrid()
        self.grid.supergrid.initialize_der(add_default=True)

        # Dynamics (optional – may fail on limited envs)
        self._dynamics_ok = False
        try:
            self.grid.supergrid.initialize_dynamics()
            self._dynamics_ok = True
            logger.info("SimEngine: dynamics initialised")
        except Exception as exc:
            logger.warning("SimEngine: dynamics init failed – %s", exc)

        # Run initial power flow
        self._pf_runner = PowerFlowRunner()
        self._last_pf: Optional[PowerFlowResult] = None
        self._run_pf()

        # Time tracking
        self._start_wall = time.time()
        self._step = 0
        self._sim_hour = 8.0  # start at 8 AM

        # Weather overrides (None = auto from 24h profile)
        self._irr_override: Optional[float] = None
        self._ws_override: Optional[float] = None

        # Base loads (for perturbation)
        self._base_loads: Optional[Any] = None
        if not self.grid.supergrid.net.load.empty:
            self._base_loads = self.grid.supergrid.net.load["p_mw"].copy()

        # History ring buffer
        self.history: deque[SimSnapshot] = deque(maxlen=self.MAX_HISTORY)

        # Capture initial snapshot
        snap = self._capture()
        self.history.append(snap)

    # ----- public interface ------------------------------------------------

    def step(self, dt_hours: float = 1.0) -> SimSnapshot:
        """Advance the simulation by *dt_hours*, return the new snapshot."""
        self._step += 1
        self._sim_hour = (self._sim_hour + dt_hours) % 24.0
        hour = self._sim_hour

        net = self.grid.supergrid.net
        dm = self.grid.supergrid.der_manager

        # 0. Apply random load perturbations to make each step dynamic --
        self._apply_load_perturbations()

        # 1. Update weather-driven DERs --------------------------------
        if dm is not None:
            irr = self._irr_override if self._irr_override is not None else _solar_irradiance_profile(hour)
            ws = self._ws_override if self._ws_override is not None else _wind_speed_profile(hour)
            for spec in dm.der_specs:
                try:
                    if spec.der_type == DERType.SOLAR_PV:
                        dm.set_solar_output(spec.name, irr)
                    elif spec.der_type == DERType.WIND:
                        dm.set_wind_output(spec.name, ws)
                except Exception:
                    pass
            # EV profile
            try:
                dm.update_ev_charging_by_hour(int(hour) % 24)
            except Exception:
                pass
            # Battery SOC
            try:
                dm.update_battery_soc(dt_hours)
            except Exception:
                pass
            # Clear overrides after use (so next step uses auto profile)
            self._irr_override = None
            self._ws_override = None

        # 2. Power flow -------------------------------------------------
        self._run_pf()

        # 3. Dynamics: run multiple sub-steps for realistic behaviour ---
        dyn_result: Dict[str, Any] = {}
        if self._dynamics_ok and self.grid.supergrid.dynamics is not None:
            try:
                # Re-sync governor/LFC state to the new PF operating point.
                # Each step jumps ~1 hour ahead; without re-sync the governor
                # and LFC carry stale transient state → frequency runaway.
                self.grid.supergrid.sync_dynamics_to_pf()
                n_substeps = max(1, int(dt_hours / 0.02))
                n_substeps = min(n_substeps, 10)  # cap at 10 sub-steps
                for _ in range(n_substeps):
                    dyn_result = self.grid.supergrid.step_dynamics(dt=0.02)
            except Exception:
                pass

        # 3b. Synthetic frequency perturbation (makes chart dynamic) ----
        if not dyn_result:
            dyn_result = self._synthetic_dynamics()

        # 4. Capture & store --------------------------------------------
        snap = self._capture(dyn_result)
        self.history.append(snap)
        return snap

    @property
    def current(self) -> SimSnapshot:
        return self.history[-1] if self.history else SimSnapshot()

    def get_history_list(self, n: int = 100) -> List[SimSnapshot]:
        return list(self.history)[-n:]

    # ----- DER control helpers -----------------------------------------

    def set_irradiance(self, irradiance: float) -> None:
        self._irr_override = irradiance

    def set_wind_speed(self, speed: float) -> None:
        self._ws_override = speed

    def set_battery_power(self, name: str, power_mw: float) -> None:
        dm = self.grid.supergrid.der_manager
        if dm:
            dm.set_battery_power(name, power_mw)

    def set_dr_curtailment(self, name: str, fraction: float) -> None:
        dm = self.grid.supergrid.der_manager
        if dm:
            dm.set_demand_response_curtailment(name, fraction)

    # ----- grid control helpers ----------------------------------------

    def scale_area_load(self, area: str, factor: float) -> None:
        self.grid.scale_area_load(area, factor)

    # ----- dynamic / perturbation helpers ------------------------------

    def _apply_load_perturbations(self) -> None:
        """Add ±2-5 % random per-bus load noise to make each step visibly different."""
        net = self.grid.supergrid.net
        if net.load.empty or self._base_loads is None:
            return
        hour = self._sim_hour
        # Diurnal load curve (peaks at ~18h, trough at ~4h)
        diurnal = 0.85 + 0.30 * math.sin(math.pi * (hour - 4.0) / 14.0)
        diurnal = max(0.7, min(1.15, diurnal))
        for idx in net.load.index:
            noise = random.gauss(1.0, 0.025)  # ±2.5 % noise
            net.load.at[idx, "p_mw"] = float(self._base_loads.iloc[idx]) * diurnal * noise

    def _synthetic_dynamics(self) -> Dict[str, Any]:
        """Generate plausible synthetic frequency/angle data when real dynamics are not available."""
        sg = self.grid.supergrid
        # Frequency: small perturbation around 50 Hz
        freq_dev = random.gauss(0, 0.04)  # std=40 mHz
        sys_freq = 50.0 + freq_dev

        gen_freqs: Dict[str, float] = {}
        gen_angles: Dict[str, float] = {}
        protection: Dict[str, Dict] = {}

        # Per-generator frequencies + angles
        if not sg.net.gen.empty:
            for idx in sg.net.gen.index:
                bus = int(sg.net.gen.at[idx, "bus"])
                # Determine area
                area = "A" if bus < 39 else ("B" if bus < 78 else "C")
                gid = f"gen_{area}_{idx}"
                gen_freqs[gid] = sys_freq + random.gauss(0, 0.015)
                gen_angles[gid] = random.gauss(0, 8)  # degrees

                # Protection relay status (normally healthy)
                gf = gen_freqs[gid]
                tripped = abs(gf - 50.0) > 0.45
                protection[gid] = {
                    "tripped": tripped,
                    "trip_reason": f"Frequency {gf:.3f} Hz" if tripped else "—",
                }

        return {
            "system_frequency_hz": sys_freq,
            "generator_frequencies": gen_freqs,
            "generator_angles": gen_angles,
            "protection": protection,
        }

    # ----- internal ----------------------------------------------------

    def _run_pf(self) -> None:
        try:
            self._last_pf = self._pf_runner.run(self.grid.supergrid.net)
        except Exception as exc:
            logger.error("Power flow failed: %s", exc)

    def _capture(self, dyn_result: Optional[Dict] = None) -> SimSnapshot:
        """Build a SimSnapshot from current grid + dynamics state."""
        pf = self._last_pf
        sg = self.grid.supergrid
        net = sg.net

        snap = SimSnapshot(
            step=self._step,
            timestamp=time.time() - self._start_wall,
            sim_hour=self._sim_hour,
        )

        # --- power flow results ---
        if pf:
            snap.converged = pf.converged
            snap.total_generation_mw = pf.total_generation_mw
            snap.total_load_mw = pf.total_load_mw
            snap.total_losses_mw = pf.total_losses_mw
            snap.min_voltage_pu = pf.min_voltage_pu
            snap.max_voltage_pu = pf.max_voltage_pu
            snap.avg_voltage_pu = pf.avg_voltage_pu
            snap.num_voltage_violations = pf.num_voltage_violations
            snap.num_line_overloads = pf.num_line_overloads
            snap.is_secure = pf.is_secure

        # --- bus voltages & angles ---
        if not net.res_bus.empty:
            for bus in net.res_bus.index:
                snap.bus_voltages[int(bus)] = float(net.res_bus.at[bus, "vm_pu"])
                snap.bus_angles[int(bus)] = float(net.res_bus.at[bus, "va_degree"])

        # --- line results ---
        if not net.res_line.empty:
            for idx in net.res_line.index:
                snap.line_results.append({
                    "index": int(idx),
                    "from_bus": int(net.line.at[idx, "from_bus"]),
                    "to_bus": int(net.line.at[idx, "to_bus"]),
                    "name": str(net.line.at[idx, "name"]) if "name" in net.line.columns else "",
                    "p_from_mw": float(net.res_line.at[idx, "p_from_mw"]),
                    "loading_percent": float(net.res_line.at[idx, "loading_percent"]),
                })

        # --- generator results ---
        if not net.res_gen.empty:
            for idx in net.res_gen.index:
                snap.gen_results.append({
                    "index": int(idx),
                    "bus": int(net.gen.at[idx, "bus"]),
                    "p_mw": float(net.res_gen.at[idx, "p_mw"]),
                    "q_mvar": float(net.res_gen.at[idx, "q_mvar"]),
                    "vm_pu": float(net.gen.at[idx, "vm_pu"]),
                })

        # --- area states ---
        try:
            for aid in AreaID:
                snap.area_states[aid.value] = sg.get_area_state(aid)
        except Exception:
            pass

        # --- tie lines ---
        try:
            snap.tie_line_flows = sg.get_tie_line_flows()
        except Exception:
            pass

        # --- DER states ---
        dm = sg.der_manager
        if dm is not None:
            try:
                all_states = dm.get_all_der_states()
                snap.der_states = [
                    {
                        "name": s.name,
                        "der_type": s.der_type.value if hasattr(s.der_type, "value") else str(s.der_type),
                        "bus": s.bus,
                        "current_output_mw": s.current_output_mw,
                        "capacity_mw": s.capacity_mw,
                        "availability": s.availability,
                        "soc": getattr(s, "soc", None),
                        "charging": getattr(s, "charging", None),
                        "utilization": getattr(s, "utilization", None),
                        "curtailment_level": getattr(s, "curtailment_level", None),
                        "curtailed_load_mw": getattr(s, "curtailed_load_mw", None),
                        "num_active_chargers": getattr(s, "num_active_chargers", None),
                    }
                    for s in all_states
                ]
            except Exception:
                pass
            try:
                snap.der_summary = dm.get_status()
            except Exception:
                pass

        # --- dynamics ---
        snap.system_frequency_hz = sg.system_frequency_hz
        if dyn_result:
            snap.system_frequency_hz = dyn_result.get("system_frequency_hz", 50.0)
            snap.generator_frequencies = dyn_result.get("generator_frequencies", {})
            snap.generator_angles = dyn_result.get("generator_angles", {})
            prot = dyn_result.get("protection", {})
            if prot:
                snap.protection_status = prot
        # NaN guard: never expose NaN to the UI
        import math
        if math.isnan(snap.system_frequency_hz):
            snap.system_frequency_hz = 50.0

        return snap
