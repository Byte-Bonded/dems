"""
Kundur Two-Area System Implementation
Classic benchmark system for inter-area oscillation studies and PSS testing

System Architecture:
===================
Area 1: G1, G2 (700 MW each) ─── L1 (967 MW)
Area 2: G3, G4 (719 MW + 700 MW) ─── L2 (1767 MW)

Connected by two parallel 230 kV tie-lines (220 km, X=0.025 pu each)
Total generation: 2719 MW, Total load: 2734 MW
Nominal tie-line flow: ~400 MW (Area 1 → Area 2)

Modal Characteristics:
=====================
- Inter-area mode: ~0.6 Hz (rotor angle oscillation between areas)
- Local modes: ~1.2-1.5 Hz (generators within same area)
- Excellent frequency separation for PSS tuning
- All generators strongly participate in inter-area mode

Reference:
=========
Kundur, P. "Power System Stability and Control" (1994)
WSCC system parameters (Western Systems Coordinating Council)
"""

import pandapower as pp
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging
from .der import DERManager, DERType, DERSpec

logger = logging.getLogger(__name__)


class AreaID(Enum):
    """Two-area system identifiers"""
    AREA_1 = "Area1"
    AREA_2 = "Area2"


@dataclass
class KundurConfig:
    """Configuration for Kundur Two-Area System"""
    nominal_frequency_hz: float = 60.0  # Original Kundur uses 60 Hz (WSCC)
    base_mva: float = 100.0
    base_voltage_kv: float = 230.0
    
    # Tie-line parameters
    tie_line_length_km: float = 220.0
    tie_line_rating_mva: float = 400.0
    num_parallel_lines: int = 2
    
    # Voltage limits (per-unit)
    v_min_pu: float = 0.95
    v_max_pu: float = 1.05
    
    # Frequency limits (Hz)
    f_min_hz: float = 59.5
    f_max_hz: float = 60.5
    
    # Load parameters
    load_area1_mw: float = 967.0
    load_area2_mw: float = 1767.0
    load_power_factor: float = 0.9  # lagging


@dataclass
class GeneratorParams:
    """Generator parameters for Kundur system"""
    name: str
    bus: int
    p_mw: float
    vm_pu: float
    area: AreaID
    
    # Dynamic parameters
    H: float  # Inertia constant (seconds)
    D: float = 2.0  # Damping coefficient (pu)
    Xd: float = 1.8  # d-axis synchronous reactance (pu)
    Xq: float = 1.7  # q-axis synchronous reactance (pu)
    Xd_prime: float = 0.3  # d-axis transient reactance (pu)
    Xq_prime: float = 0.55  # q-axis transient reactance (pu)
    Td0_prime: float = 8.0  # d-axis open-circuit time constant (s)
    Tq0_prime: float = 0.4  # q-axis open-circuit time constant (s)
    Ra: float = 0.0025  # Armature resistance (pu)
    
    # Governor parameters (TGOV1)
    R_droop: float = 0.05  # 5% droop
    Tg: float = 0.2  # Governor time constant (s)
    Tt: float = 0.5  # Turbine time constant (s)
    Pmax: float = 1.2  # Max power (pu)
    Pmin: float = 0.0  # Min power (pu)
    
    # Exciter parameters (IEEE Type 1)
    KA: float = 200.0  # AVR gain
    TA: float = 0.02  # AVR time constant (s)
    TR: float = 0.02  # Voltage transducer time constant (s)
    KE: float = 1.0  # Exciter gain
    TE: float = 0.5  # Exciter time constant (s)
    VRMAX: float = 5.0  # Max regulator output (pu)
    VRMIN: float = -5.0  # Min regulator output (pu)


# Standard Kundur generator parameters
KUNDUR_GENERATORS = [
    GeneratorParams(
        name="G1", bus=0, p_mw=700.0, vm_pu=1.03, area=AreaID.AREA_1,
        H=6.5, Xd=1.8, Xd_prime=0.3
    ),
    GeneratorParams(
        name="G2", bus=1, p_mw=700.0, vm_pu=1.01, area=AreaID.AREA_1,
        H=6.5, Xd=1.8, Xd_prime=0.3
    ),
    GeneratorParams(
        name="G3", bus=10, p_mw=719.0, vm_pu=1.03, area=AreaID.AREA_2,
        H=6.175, Xd=1.8, Xd_prime=0.3
    ),
    GeneratorParams(
        name="G4", bus=11, p_mw=700.0, vm_pu=1.01, area=AreaID.AREA_2,
        H=6.175, Xd=1.8, Xd_prime=0.3
    ),
]


class KundurTwoAreaSystem:
    """
    Kundur Two-Area System for inter-area oscillation studies
    
    This is the industry-standard benchmark for:
    - Power System Stabilizer (PSS) design and testing
    - Wide-area damping controller development
    - Inter-area oscillation analysis (~0.6 Hz mode)
    - Small-signal stability studies
    - RL-based control algorithm validation
    
    Advantages over larger systems (e.g., IEEE 39-bus):
    - Clear modal structure with well-separated frequencies
    - All generators participate strongly in inter-area mode
    - Fast simulation (40 states vs 300+)
    - Analytical tractability for eigenvalue analysis
    - Direct transfer to real WSCC system behavior
    """
    
    def __init__(self, config: Optional[KundurConfig] = None, enable_der: bool = True):
        """
        Initialize Kundur Two-Area System
        
        Args:
            config: System configuration (uses defaults if None)
            enable_der: Enable DER (Solar, Wind, Battery, EV) integration
        """
        self.config = config or KundurConfig()
        self.net = None
        self.generator_params = KUNDUR_GENERATORS.copy()
        self.enable_der = enable_der
        self.der_manager: Optional[DERManager] = None
        self._create_network()
        
        # Initialize DER manager if enabled
        if self.enable_der:
            self.der_manager = DERManager(self.net)
            self._install_default_ders()
            
        logger.info("Kundur Two-Area System initialized")
        
    def _create_network(self) -> None:
        """Create the PandaPower network model"""
        self.net = pp.create_empty_network(
            name="Kundur Two-Area System",
            f_hz=self.config.nominal_frequency_hz,
            sn_mva=self.config.base_mva
        )
        
        # Create buses
        self._create_buses()
        
        # Create generators
        self._create_generators()
        
        # Create loads
        self._create_loads()
        
        # Create transmission lines between areas
        self._create_transmission_system()
        
        # Create tie-lines
        self._create_tie_lines()
        
        logger.info(
            f"Created Kundur system: {len(self.net.bus)} buses, "
            f"{len(self.net.gen)} generators, {len(self.net.line)} lines"
        )
        
    def _create_buses(self) -> None:
        """Create the 11-bus system"""
        # Area 1: Buses 0-5
        # Bus 0-1: Generator buses
        pp.create_bus(self.net, vn_kv=20.0, name="G1", type="b", zone="Area1")
        pp.create_bus(self.net, vn_kv=20.0, name="G2", type="b", zone="Area1")
        
        # Bus 2-3: Step-up transformer buses
        pp.create_bus(self.net, vn_kv=230.0, name="B1_HV", type="b", zone="Area1")
        pp.create_bus(self.net, vn_kv=230.0, name="B2_HV", type="b", zone="Area1")
        
        # Bus 4: Area 1 interconnection bus
        pp.create_bus(self.net, vn_kv=230.0, name="B5_Area1", type="b", zone="Area1")
        
        # Bus 5: Load bus Area 1
        pp.create_bus(self.net, vn_kv=230.0, name="L1", type="b", zone="Area1")
        
        # Area 2: Buses 6-10
        # Bus 6: Area 2 interconnection bus
        pp.create_bus(self.net, vn_kv=230.0, name="B6_Area2", type="b", zone="Area2")
        
        # Bus 7: Load bus Area 2
        pp.create_bus(self.net, vn_kv=230.0, name="L2", type="b", zone="Area2")
        
        # Bus 8-9: Step-up transformer buses
        pp.create_bus(self.net, vn_kv=230.0, name="B3_HV", type="b", zone="Area2")
        pp.create_bus(self.net, vn_kv=230.0, name="B4_HV", type="b", zone="Area2")
        
        # Bus 10-11: Generator buses
        pp.create_bus(self.net, vn_kv=20.0, name="G3", type="b", zone="Area2")
        pp.create_bus(self.net, vn_kv=20.0, name="G4", type="b", zone="Area2")
        
    def _create_generators(self) -> None:
        """Create the 4 generators with standard Kundur parameters"""
        for i, gen_params in enumerate(self.generator_params):
            # First generator is the slack bus
            is_slack = (i == 0)
            
            pp.create_gen(
                self.net,
                bus=gen_params.bus,
                p_mw=gen_params.p_mw,
                vm_pu=gen_params.vm_pu,
                name=gen_params.name,
                slack=is_slack,  # G1 is slack bus
                controllable=True,
                min_p_mw=gen_params.p_mw * 0.1,
                max_p_mw=gen_params.p_mw * gen_params.Pmax,
                min_q_mvar=-gen_params.p_mw * 0.6,
                max_q_mvar=gen_params.p_mw * 0.6,
            )
            
    def _create_loads(self) -> None:
        """Create loads at designated load buses"""
        # Area 1 load (Bus 5)
        pp.create_load(
            self.net,
            bus=5,
            p_mw=self.config.load_area1_mw,
            q_mvar=self.config.load_area1_mw * np.tan(np.arccos(self.config.load_power_factor)),
            name="Load_Area1",
            controllable=False
        )
        
        # Area 2 load (Bus 7)
        pp.create_load(
            self.net,
            bus=7,
            p_mw=self.config.load_area2_mw,
            q_mvar=self.config.load_area2_mw * np.tan(np.arccos(self.config.load_power_factor)),
            name="Load_Area2",
            controllable=False
        )
        
    def _create_transmission_system(self) -> None:
        """Create transmission lines within each area"""
        # Area 1 internal lines
        # Generator step-up transformers (20 kV → 230 kV)
        pp.create_transformer_from_parameters(
            self.net, hv_bus=2, lv_bus=0, sn_mva=900.0, vn_hv_kv=230.0,
            vn_lv_kv=20.0, vk_percent=15.0, vkr_percent=0.5,
            pfe_kw=0, i0_percent=0, name="Trafo_G1"
        )
        pp.create_transformer_from_parameters(
            self.net, hv_bus=3, lv_bus=1, sn_mva=900.0, vn_hv_kv=230.0,
            vn_lv_kv=20.0, vk_percent=15.0, vkr_percent=0.5,
            pfe_kw=0, i0_percent=0, name="Trafo_G2"
        )
        
        # Area 1: HV buses to interconnection bus (short lines, 25 km)
        self._create_line(from_bus=2, to_bus=4, length_km=25.0, name="L1-5_A")
        self._create_line(from_bus=3, to_bus=4, length_km=25.0, name="L2-5_A")
        
        # Area 1: Interconnection to load bus (10 km)
        self._create_line(from_bus=4, to_bus=5, length_km=10.0, name="L5-Load1")
        
        # Area 2 internal lines
        # Area 2: Interconnection to load bus (10 km)
        self._create_line(from_bus=6, to_bus=7, length_km=10.0, name="L6-Load2")
        
        # Area 2: HV buses to interconnection bus (25 km)
        self._create_line(from_bus=8, to_bus=6, length_km=25.0, name="L3-6_B")
        self._create_line(from_bus=9, to_bus=6, length_km=25.0, name="L4-6_B")
        
        # Generator step-up transformers (20 kV → 230 kV)
        pp.create_transformer_from_parameters(
            self.net, hv_bus=8, lv_bus=10, sn_mva=900.0, vn_hv_kv=230.0,
            vn_lv_kv=20.0, vk_percent=15.0, vkr_percent=0.5,
            pfe_kw=0, i0_percent=0, name="Trafo_G3"
        )
        pp.create_transformer_from_parameters(
            self.net, hv_bus=9, lv_bus=11, sn_mva=900.0, vn_hv_kv=230.0,
            vn_lv_kv=20.0, vk_percent=15.0, vkr_percent=0.5,
            pfe_kw=0, i0_percent=0, name="Trafo_G4"
        )
        
    def _create_tie_lines(self) -> None:
        """
        Create the critical tie-lines between areas
        Two parallel 230 kV lines, 220 km each
        These lines carry inter-area power flow and are key to inter-area oscillations
        """
        for i in range(self.config.num_parallel_lines):
            self._create_line(
                from_bus=4,  # Area 1 interconnection bus
                to_bus=6,    # Area 2 interconnection bus
                length_km=self.config.tie_line_length_km,
                name=f"TieLine_{i+1}",
                max_loading_percent=100.0,
                is_tie_line=True
            )
            
    def _create_line(
        self,
        from_bus: int,
        to_bus: int,
        length_km: float,
        name: str,
        max_loading_percent: float = 100.0,
        is_tie_line: bool = False
    ) -> None:
        """
        Create a transmission line with standard 230 kV parameters
        
        Standard parameters for 230 kV lines:
        - r = 0.0001 Ohm/km (low resistance for EHV)
        - x = 0.001 Ohm/km (inductive reactance)
        - c = 10 nF/km (capacitance)
        - max_i_ka = Rating based on thermal limits
        """
        # Calculate line rating
        if is_tie_line:
            max_i_ka = self.config.tie_line_rating_mva / (np.sqrt(3) * self.config.base_voltage_kv)
        else:
            max_i_ka = 2.0  # 2 kA for internal lines
        
        pp.create_line_from_parameters(
            self.net,
            from_bus=from_bus,
            to_bus=to_bus,
            length_km=length_km,
            r_ohm_per_km=0.0001,
            x_ohm_per_km=0.001,
            c_nf_per_km=10.0,
            max_i_ka=max_i_ka,
            name=name,
            max_loading_percent=max_loading_percent
        )
        
    def get_generator_params(self, gen_name: str) -> Optional[GeneratorParams]:
        """Get dynamic parameters for a specific generator"""
        for gen in self.generator_params:
            if gen.name == gen_name:
                return gen
        return None
        
    def get_tie_line_flow(self) -> Dict[str, float]:
        """
        Get power flow on tie-lines (critical for inter-area oscillation monitoring)
        
        Returns:
            Dict with tie-line flows in MW and MVAR
        """
        if self.net.res_line is None or len(self.net.res_line) == 0:
            logger.warning("No power flow results available. Run power flow first.")
            return {}
            
        tie_line_flows = {}
        for idx, row in self.net.line.iterrows():
            if "TieLine" in row['name']:
                tie_line_flows[row['name']] = {
                    'p_from_mw': self.net.res_line.at[idx, 'p_from_mw'],
                    'q_from_mvar': self.net.res_line.at[idx, 'q_from_mvar'],
                    'p_to_mw': self.net.res_line.at[idx, 'p_to_mw'],
                    'q_to_mvar': self.net.res_line.at[idx, 'q_to_mvar'],
                    'loading_percent': self.net.res_line.at[idx, 'loading_percent'],
                }
                
        return tie_line_flows
        
    def get_area_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get aggregated metrics per area"""
        if self.net.res_bus is None or len(self.net.res_bus) == 0:
            logger.warning("No power flow results available. Run power flow first.")
            return {}
            
        metrics = {
            "Area1": {"generation_mw": 0.0, "load_mw": 0.0, "avg_voltage_pu": 0.0},
            "Area2": {"generation_mw": 0.0, "load_mw": 0.0, "avg_voltage_pu": 0.0},
        }
        
        # Aggregate by area
        for area_name in ["Area1", "Area2"]:
            area_buses = self.net.bus[self.net.bus.zone == area_name].index
            
            # Generation
            area_gens = self.net.gen[self.net.gen.bus.isin(area_buses)]
            for idx in area_gens.index:
                metrics[area_name]["generation_mw"] += self.net.res_gen.at[idx, 'p_mw']
                
            # Load
            area_loads = self.net.load[self.net.load.bus.isin(area_buses)]
            for idx in area_loads.index:
                metrics[area_name]["load_mw"] += self.net.res_load.at[idx, 'p_mw']
                
            # Average voltage
            voltages = [self.net.res_bus.at[bus, 'vm_pu'] for bus in area_buses]
            metrics[area_name]["avg_voltage_pu"] = np.mean(voltages)
            
        return metrics
        
    def get_state(self) -> Dict:
        """
        Get complete system state for RL environment
        
        Returns:
            Dict with generators, loads, tie-lines, and area-level metrics
        """
        state = {
            "generators": [],
            "loads": [],
            "tie_lines": self.get_tie_line_flow(),
            "areas": self.get_area_metrics(),
            "system": {
                "total_generation_mw": 0.0,
                "total_load_mw": 0.0,
                "tie_line_flow_mw": 0.0,
            }
        }
        
        if self.net.res_gen is not None and len(self.net.res_gen) > 0:
            for idx, row in self.net.gen.iterrows():
                gen_state = {
                    "name": row['name'],
                    "bus": int(row['bus']),
                    "p_mw": float(self.net.res_gen.at[idx, 'p_mw']),
                    "q_mvar": float(self.net.res_gen.at[idx, 'q_mvar']),
                    "vm_pu": float(self.net.res_gen.at[idx, 'vm_pu']),
                }
                state["generators"].append(gen_state)
                state["system"]["total_generation_mw"] += gen_state["p_mw"]
                
        if self.net.res_load is not None and len(self.net.res_load) > 0:
            for idx, row in self.net.load.iterrows():
                load_state = {
                    "name": row['name'],
                    "bus": int(row['bus']),
                    "p_mw": float(self.net.res_load.at[idx, 'p_mw']),
                    "q_mvar": float(self.net.res_load.at[idx, 'q_mvar']),
                }
                state["loads"].append(load_state)
                state["system"]["total_load_mw"] += load_state["p_mw"]
                
        # Calculate net tie-line flow (Area 1 → Area 2)
        for tie_name, flow in state["tie_lines"].items():
            state["system"]["tie_line_flow_mw"] += flow['p_from_mw']
            
        return state
        
    def set_generator_setpoint(self, gen_name: str, p_mw: float) -> bool:
        """
        Set generator active power setpoint (for control actions)
        
        Args:
            gen_name: Generator name (G1, G2, G3, G4)
            p_mw: Active power setpoint in MW
            
        Returns:
            True if successful, False otherwise
        """
        for idx, row in self.net.gen.iterrows():
            if row['name'] == gen_name:
                # Enforce limits
                p_mw = np.clip(p_mw, row['min_p_mw'], row['max_p_mw'])
                self.net.gen.at[idx, 'p_mw'] = p_mw
                logger.debug(f"Set {gen_name} setpoint to {p_mw:.2f} MW")
                return True
                
        logger.warning(f"Generator {gen_name} not found")
        return False
        
    def apply_load_perturbation(self, area: AreaID, delta_mw: float) -> None:
        """
        Apply load step change (for transient stability testing)
        
        Args:
            area: Target area for perturbation
            delta_mw: Load change in MW (positive = increase)
        """
        area_name = area.value
        for idx, row in self.net.load.iterrows():
            bus = row['bus']
            zone = self.net.bus.at[bus, 'zone']
            if zone == area_name:
                current_p = self.net.load.at[idx, 'p_mw']
                new_p = current_p + delta_mw
                self.net.load.at[idx, 'p_mw'] = max(0, new_p)
                logger.info(f"Applied {delta_mw:+.2f} MW perturbation to {row['name']}")
                
    def _install_default_ders(self) -> None:
        """
        Install default DER portfolio for testing and demonstration
        
        Distributes DERs across both areas:
        - Solar PV: Near load centers (high local consumption)
        - Wind: At transmission level (bulk power injection)
        - Battery: At load centers (peak shaving, frequency support)
        - EV Charging: Load buses (smart charging capability)
        """
        if self.der_manager is None:
            return
            
        # Area 1 DERs (buses 0-5)
        # Solar at load bus (distributed generation)
        self.der_manager.add_solar_pv(
            bus=5, capacity_mw=50.0, name="Solar_A1", area_id="Area1",
            panel_area_m2=250000, efficiency=0.20
        )
        
        # Wind at transmission level
        self.der_manager.add_wind_turbine(
            bus=4, capacity_mw=100.0, name="Wind_A1", area_id="Area1",
            num_turbines=50
        )
        
        # Battery storage at load bus
        self.der_manager.add_battery(
            bus=5, power_mw=30.0, energy_mwh=120.0,
            name="BESS_A1", area_id="Area1", initial_soc=0.6
        )
        
        # EV charging at load bus
        self.der_manager.add_ev_charging_station(
            bus=5, num_chargers=100, charger_power_kw=200,
            name="EV_A1", area_id="Area1"
        )
        
        # Area 2 DERs (buses 6-11)
        # Larger solar deployment (higher load)
        self.der_manager.add_solar_pv(
            bus=7, capacity_mw=80.0, name="Solar_A2", area_id="Area2",
            panel_area_m2=400000, efficiency=0.20
        )
        
        # Wind farm at transmission level
        self.der_manager.add_wind_turbine(
            bus=6, capacity_mw=150.0, name="Wind_A2", area_id="Area2",
            num_turbines=75
        )
        
        # Battery storage at load bus
        self.der_manager.add_battery(
            bus=7, power_mw=50.0, energy_mwh=200.0,
            name="BESS_A2", area_id="Area2", initial_soc=0.5
        )
        
        # EV charging at load bus
        self.der_manager.add_ev_charging_station(
            bus=7, num_chargers=175, charger_power_kw=200,
            name="EV_A2", area_id="Area2"
        )
        
        logger.info(
            f"Installed {len(self.der_manager.der_specs)} DER units: "
            f"Solar={sum(1 for d in self.der_manager.der_specs if d.der_type == DERType.SOLAR_PV)}, "
            f"Wind={sum(1 for d in self.der_manager.der_specs if d.der_type == DERType.WIND)}, "
            f"BESS={sum(1 for d in self.der_manager.der_specs if d.der_type == DERType.BESS)}, "
            f"EV={sum(1 for d in self.der_manager.der_specs if d.der_type == DERType.EV_CHARGING)}"
        )
        
    def get_der_state(self) -> List[Dict]:
        """Get current state of all DER units"""
        if self.der_manager is None:
            return []
            
        return [
            {
                "name": der.name,
                "type": der.der_type.value,
                "bus": der.bus,
                "capacity_mw": der.capacity_mw,
                "area": der.area_id,
            }
            for der in self.der_manager.der_specs
        ]
        
    def update_der_conditions(
        self,
        solar_irradiance_w_m2: float = 800.0,
        wind_speed_m_s: float = 12.0,
        temperature_c: float = 25.0
    ) -> None:
        """
        Update DER output based on environmental conditions
        
        Args:
            solar_irradiance_w_m2: Solar irradiance (0-1000 W/m²)
            wind_speed_m_s: Wind speed (0-25 m/s)
            temperature_c: Ambient temperature (°C)
        """
        if self.der_manager is None:
            return
            
        # Update solar PV outputs
        for der in self.der_manager.der_specs:
            if der.der_type == DERType.SOLAR_PV:
                self.der_manager.set_solar_output(der.name, solar_irradiance_w_m2)
            elif der.der_type == DERType.WIND:
                self.der_manager.set_wind_output(der.name, wind_speed_m_s)
        
    def dispatch_battery(self, battery_name: str, power_mw: float, duration_hours: float = 1.0) -> bool:
        """
        Dispatch battery storage (positive = discharge, negative = charge)
        
        Args:
            battery_name: Name of battery to dispatch
            power_mw: Power setpoint (positive=discharge, negative=charge)
            duration_hours: Duration for energy calculation
            
        Returns:
            True if successful, False otherwise
        """
        if self.der_manager is None:
            return False
            
        return self.der_manager.dispatch_battery(battery_name, power_mw, duration_hours)
        
    def __repr__(self) -> str:
        der_str = ""
        if self.der_manager:
            num_ders = len(self.der_manager.der_specs)
            der_str = f", DERs={num_ders}"
            
        return (
            f"KundurTwoAreaSystem("
            f"buses={len(self.net.bus)}, "
            f"generators={len(self.net.gen)}, "
            f"lines={len(self.net.line)}"
            f"{der_str}, "
            f"f={self.config.nominal_frequency_hz} Hz)"
        )
                
    def __repr__(self) -> str:
        return (
            f"KundurTwoAreaSystem("
            f"buses={len(self.net.bus)}, "
            f"generators={len(self.net.gen)}, "
            f"lines={len(self.net.line)}, "
            f"f={self.config.nominal_frequency_hz} Hz)"
        )
