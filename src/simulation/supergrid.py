"""
Tri-Area Super-Grid Implementation
Creates a 117-bus system by merging three IEEE 39-Bus (New England) systems

The IEEE 39-bus system represents the New England power system with:
- 10 generators
- 39 buses
- 46 transmission lines
- 12 transformers

By merging 3 instances, we get:
- 30 generators (controllable units)
- 117 buses
- 138+ lines (plus tie-lines between areas)

Tie-Line Strategy:
==================
The IEEE 39-bus has key interconnection buses:
- Bus 1: Major load center, high connectivity
- Bus 2: Connected to Bus 1, good for parallel paths  
- Bus 3: Load bus with good connectivity
- Bus 9: Strategic position in network
- Bus 14: Central load bus
- Bus 39: Generator bus, represents external grid connection

We create a MESH topology between areas for reliability (N-1 secure):
- Area A ↔ Area B: 3 tie-lines
- Area B ↔ Area C: 3 tie-lines  
- Area A ↔ Area C: 2 tie-lines (direct path)
"""

import pandapower as pp
import pandapower.networks as pn
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging
from .der import DERManager, DERType
from .dynamics import (
    DynamicsCoordinator, SynchronousGeneratorDynamic, ExcitationSystem,
    GovernorTurbine, AutomaticGenerationControl, DynamicLoadModel,
    ProtectionRelay, IEEE39_GENERATOR_DATA, create_ieee39_dynamics
)

logger = logging.getLogger(__name__)


class AreaID(Enum):
    """Identifiers for the three grid areas"""
    AREA_A = "A"
    AREA_B = "B"
    AREA_C = "C"


@dataclass
class AreaConfig:
    """Configuration for a single area in the super-grid"""
    area_id: AreaID
    bus_offset: int  # Offset to add to bus indices when merging
    name: str
    color: str  # For visualization
    
    # After merging, these track the actual bus indices in the super-grid
    bus_range: Tuple[int, int] = field(default=(0, 0))
    generator_indices: List[int] = field(default_factory=list)
    load_indices: List[int] = field(default_factory=list)


@dataclass 
class SuperGridConfig:
    """Configuration for the entire Tri-Area Super-Grid"""
    nominal_frequency_hz: float = 50.0  # 50Hz Indian power system
    base_mva: float = 100.0
    
    # Tie-line thermal limits (MVA)
    tie_line_rating_mva: float = 600.0
    
    # Voltage limits (per-unit)
    v_min_pu: float = 0.95
    v_max_pu: float = 1.05
    
    # Frequency limits (Hz) - Indian Grid Code (IEGC) standards
    f_min_hz: float = 49.5
    f_max_hz: float = 50.5


@dataclass
class TieLineSpec:
    """Specification for a tie-line between areas"""
    name: str
    from_area: AreaID
    from_bus_local: int  # Bus index within the area (0-38)
    to_area: AreaID
    to_bus_local: int
    length_km: float
    rating_mva: float
    r_ohm_per_km: float = 0.02
    x_ohm_per_km: float = 0.25
    c_nf_per_km: float = 12.0


class SuperGrid:
    """
    Tri-Area Super-Grid: 117-bus system from 3x IEEE 39-bus networks
    
    This class manages:
    1. Grid topology creation and merging
    2. Area-based organization of components
    3. State extraction for RL agent
    4. Control interface for generators
    
    Topology Overview (8 Tie-Lines for N-1 Security):
    =================================================
    
              AREA A (North)
             /      |      \
        TL1/    TL2|    TL3\
           /        |        \
      AREA B ----TL4---- AREA C
      (West)  ----TL5----  (East)
              ----TL6----
                  |
              TL7,TL8 (A↔C direct)
    """
    
    def __init__(self, config: Optional[SuperGridConfig] = None):
        """
        Constructs the merged 117-bus Tri-Area Super-Grid and applies initial configuration.
        
        Builds three IEEE 39-bus areas, assigns per-area offsets and visual metadata, creates the predefined tie-line topology for N-1 security, merges the networks into a single pandapower net, and runs initial convergence-improvement steps. Leaves DER manager and dynamics coordinator uninitialized (placeholders), and initializes the system frequency to the configured nominal value.
        
        Parameters:
            config (Optional[SuperGridConfig]): Optional grid-wide configuration; when omitted a default SuperGridConfig is used.
        """
        self.config = config or SuperGridConfig()
        self.net: Optional[pp.pandapowerNet] = None
        
        # Area configurations
        self.areas: Dict[AreaID, AreaConfig] = {
            AreaID.AREA_A: AreaConfig(
                area_id=AreaID.AREA_A,
                bus_offset=0,
                name="Area A (North)",
                color="#FF6B6B"  # Pinkish-red (matches UI theme)
            ),
            AreaID.AREA_B: AreaConfig(
                area_id=AreaID.AREA_B,
                bus_offset=39,
                name="Area B (West)",
                color="#4ECDC4"  # Teal
            ),
            AreaID.AREA_C: AreaConfig(
                area_id=AreaID.AREA_C,
                bus_offset=78,
                name="Area C (East)",
                color="#FFE66D"  # Yellow
            ),
        }
        
        # Strategic tie-line connections for N-1 security
        # Using key buses in IEEE 39-bus: 1, 2, 3, 9, 14, 26, 39
        self.tie_line_specs: List[TieLineSpec] = [
            # === AREA A ↔ AREA B (3 parallel paths) ===
            TieLineSpec(
                name="TL_AB_1",
                from_area=AreaID.AREA_A, from_bus_local=1,
                to_area=AreaID.AREA_B, to_bus_local=1,
                length_km=100, rating_mva=700
            ),
            TieLineSpec(
                name="TL_AB_2", 
                from_area=AreaID.AREA_A, from_bus_local=2,
                to_area=AreaID.AREA_B, to_bus_local=2,
                length_km=120, rating_mva=600
            ),
            TieLineSpec(
                name="TL_AB_3",
                from_area=AreaID.AREA_A, from_bus_local=39,
                to_area=AreaID.AREA_B, to_bus_local=9,
                length_km=150, rating_mva=500
            ),
            
            # === AREA B ↔ AREA C (3 parallel paths) ===
            TieLineSpec(
                name="TL_BC_1",
                from_area=AreaID.AREA_B, from_bus_local=3,
                to_area=AreaID.AREA_C, to_bus_local=3,
                length_km=110, rating_mva=650
            ),
            TieLineSpec(
                name="TL_BC_2",
                from_area=AreaID.AREA_B, from_bus_local=14,
                to_area=AreaID.AREA_C, to_bus_local=14,
                length_km=130, rating_mva=550
            ),
            TieLineSpec(
                name="TL_BC_3",
                from_area=AreaID.AREA_B, from_bus_local=26,
                to_area=AreaID.AREA_C, to_bus_local=26,
                length_km=140, rating_mva=500
            ),
            
            # === AREA A ↔ AREA C (2 direct paths - longest distance) ===
            TieLineSpec(
                name="TL_AC_1",
                from_area=AreaID.AREA_A, from_bus_local=9,
                to_area=AreaID.AREA_C, to_bus_local=9,
                length_km=200, rating_mva=600  # Increased for margin
            ),
            TieLineSpec(
                name="TL_AC_2",
                from_area=AreaID.AREA_A, from_bus_local=14,
                to_area=AreaID.AREA_C, to_bus_local=1,
                length_km=220, rating_mva=500  # Increased for margin
            ),
        ]
        
        # Build the grid
        self._build_supergrid()
        
        # Initialize DER manager
        self.der_manager: Optional[DERManager] = None
        
        # Initialize dynamics coordinator (for frequency simulation)
        self.dynamics: Optional[DynamicsCoordinator] = None
        
        # System frequency (updated by dynamics simulation)
        self.system_frequency_hz = 50.0
        
        # Improve convergence settings
        self._improve_convergence()
        
    def _build_supergrid(self) -> None:
        """
        Construct the 117-bus super-grid by merging three IEEE 39-bus systems.
        
        Builds the combined network and applies post-merge configuration: assigns the merged Network object to self.net, updates per-area bus/generator/load mappings, adds inter-area tie-lines, and configures the system slack/reference bus for power flow.
        """
        logger.info("Building Tri-Area Super-Grid (117 buses)...")
        
        # Create three separate IEEE 39-bus networks
        net_a = pn.case39()
        net_b = pn.case39()
        net_c = pn.case39()
        
        # Add area zone information to each network before merging
        net_a.bus["zone"] = 1
        net_b.bus["zone"] = 2
        net_c.bus["zone"] = 3
        
        # Merge networks: A + B first, then + C
        logger.info("Merging Area A and Area B...")
        merged_ab = pp.merge_nets(net_a, net_b, validate=False)
        
        logger.info("Merging with Area C...")
        self.net = pp.merge_nets(merged_ab, net_c, validate=False)
        
        # Update area configurations with actual bus ranges
        self._update_area_mappings()
        
        # Add tie-lines between areas
        self._add_tie_lines()
        
        # Set slack bus (generator at bus 30 in Area A - the main slack in IEEE 39)
        self._configure_slack_bus()
        
        logger.info(f"Super-Grid built successfully:")
        logger.info(f"  - Total buses: {len(self.net.bus)}")
        logger.info(f"  - Total generators: {len(self.net.gen)}")
        logger.info(f"  - Total lines: {len(self.net.line)}")
        logger.info(f"  - Total loads: {len(self.net.load)}")
        logger.info(f"  - Tie-lines: {len(self.tie_line_specs)}")
    
    def _improve_convergence(self) -> None:
        """
        Improve the network power flow convergence by adjusting reactive limits and adding shunt compensation.
        
        This method modifies the internal pandapower network to help achieve a bus voltage profile within 0.95–1.05 pu. Changes and effects:
        - Ensures generators can absorb reactive power by relaxing positive min Q limits and expanding max Q capability.
        - Sets generator and slack voltage setpoints to 1.00 pu.
        - Runs up to five Newton–Raphson power-flow attempts and stops early if all bus voltages are within 0.95–1.05 pu.
        - Adds shunt elements at buses with persistently low or high voltages to provide capacitive or inductive reactive compensation respectively.
        - Logs progress and final counts of shunts and net reactive injection.
        
        Note: In pandapower, positive shunt q_mvar is inductive (absorbs reactive power, lowers voltage) and negative q_mvar is capacitive (supplies reactive power, raises voltage).
        """
        logger.info("Applying voltage corrections...")
        
        # 1. Fix generator reactive power limits
        # Some IEEE 39-bus generators have positive min Q limits which prevents 
        # them from absorbing reactive power - this causes high voltage
        # Allow all generators to absorb Q (negative min_q_mvar)
        for idx in self.net.gen.index:
            if self.net.gen.at[idx, 'min_q_mvar'] > 0:
                # Set to negative value to allow Q absorption
                max_q = self.net.gen.at[idx, 'max_q_mvar']
                self.net.gen.at[idx, 'min_q_mvar'] = -max_q * 0.5  # 50% absorption capability
        
        # 2. Expand max Q capability
        self.net.gen['max_q_mvar'] = self.net.gen['max_q_mvar'] * 1.5
        
        # 3. Set generator voltage setpoints at 1.0 pu (nominal)
        self.net.gen['vm_pu'] = 1.00
        
        # 4. Set slack bus voltage to 1.0 pu
        ext_grid_idx = self.net.ext_grid.index[0]
        self.net.ext_grid.at[ext_grid_idx, 'vm_pu'] = 1.00
        
        # 5. Run iterative voltage correction (up to 5 iterations)
        for iteration in range(5):
            try:
                pp.runpp(self.net, algorithm='nr', max_iteration=50)
                
                vm_min = self.net.res_bus['vm_pu'].min()
                vm_max = self.net.res_bus['vm_pu'].max()
                
                logger.info(f"  Iteration {iteration+1}: V={vm_min:.3f}-{vm_max:.3f} pu")
                
                # Check if within acceptable range
                if vm_min >= 0.95 and vm_max <= 1.05:
                    logger.info("  ✓ Voltage profile within limits")
                    break
                
                # Add targeted compensation (tighter thresholds for better final result)
                buses_fixed = 0
                for bus_idx in self.net.res_bus.index:
                    vm = self.net.res_bus.at[bus_idx, 'vm_pu']
                    
                    # Low voltage buses - add capacitors
                    if vm < 0.97:
                        q_cap = -30 * (0.97 - vm) / 0.01  # 30 MVAR per 0.01 pu
                        pp.create_shunt(self.net, bus_idx, q_mvar=q_cap, p_mw=0)
                        buses_fixed += 1
                    
                    # High voltage buses - add inductors
                    elif vm > 1.03:
                        q_ind = 30 * (vm - 1.03) / 0.01  # 30 MVAR per 0.01 pu
                        pp.create_shunt(self.net, bus_idx, q_mvar=q_ind, p_mw=0)
                        buses_fixed += 1
                
                if buses_fixed == 0:
                    break
                    
            except pp.powerflow.LoadflowNotConverged:
                logger.warning(f"  Iteration {iteration+1}: Power flow did not converge")
                break
        
        shunt_count = len(self.net.shunt)
        total_q = self.net.shunt.q_mvar.sum() if shunt_count > 0 else 0
        logger.info(f"  - Total: {shunt_count} shunts, {total_q:.0f} MVAR net")
    
    def initialize_dynamics(self) -> DynamicsCoordinator:
        """
        Initialize IEEE-39-standard dynamic models for all generators and area-level AGC.
        
        Creates per-generator dynamic models (swing dynamics) with automatic voltage regulators (AVR) and governor-turbine controllers; units with MVA >= 600 also receive a power system stabilizer (PSS). For each area, an AGC controller is created with participation factors proportional to generator MVA.
        
        Returns:
            DynamicsCoordinator: coordinator managing the added generator dynamics and AGC controllers.
        """
        logger.info("Initializing dynamic models (IEEE 39-bus standard)...")
        
        self.dynamics = DynamicsCoordinator()
        
        # Add generators for each area with IEEE 39-bus parameters
        for area_id, area_config in self.areas.items():
            offset = area_config.bus_offset
            participating_gens = []
            
            for local_bus, data in IEEE39_GENERATOR_DATA.items():
                global_bus = offset + local_bus
                gen_id = f"Gen_{area_id.value}_{local_bus}"
                
                from .dynamics import GeneratorDynamicParams
                params = GeneratorDynamicParams(
                    H=data["H"],
                    D=data["D"],
                    MVA_base=data["MVA"],
                    frequency=50.0
                )
                
                # Add generator with AVR and governor
                # PSS for larger units (>600 MVA)
                with_pss = data["MVA"] >= 600
                self.dynamics.add_generator(
                    gen_id, global_bus, params,
                    with_avr=True, with_governor=True, with_pss=with_pss
                )
                
                # Add to AGC (except slack generators)
                if data["type"] != "Slack":
                    pf = data["MVA"] / 5000  # Participation proportional to size
                    participating_gens.append((gen_id, pf))
            
            # Create AGC for area
            self.dynamics.add_agc(area_id.value, participating_gens)
            logger.info(f"  - Area {area_id.value}: 10 generators, AGC enabled")
        
        logger.info(f"✓ Initialized {len(self.dynamics.generators)} dynamic generators")
        logger.info(f"✓ AGC controllers: {len(self.dynamics.agc_controllers)}")
        
        return self.dynamics
    
    def step_dynamics(self, dt: float = 0.01) -> Dict:
        """
        Advance the dynamic simulation by a single integration step.
        
        Parameters:
            dt (float): Time step in seconds (e.g., 0.01 for 10 ms).
        
        Returns:
            result (dict): Simulation result dictionary containing dynamics outputs. Includes at minimum
                'system_frequency_hz' with the updated system frequency in Hz; may include additional
                per-generator or per-bus dynamic state provided by the DynamicsCoordinator.
        
        Raises:
            RuntimeError: If dynamics have not been initialized via initialize_dynamics().
            RuntimeError: If a steady-state power flow has not been solved (net.res_bus is empty).
        """
        if self.dynamics is None:
            raise RuntimeError("Dynamics not initialized. Call initialize_dynamics() first.")
        
        if self.net.res_bus.empty:
            raise RuntimeError("Power flow not solved. Run power flow first.")
        
        # Get bus voltages from power flow results
        bus_voltages = {int(bus): float(self.net.res_bus.at[bus, 'vm_pu']) 
                       for bus in self.net.res_bus.index}
        
        # Get generator powers from power flow results
        gen_powers = {}
        for gen_id in self.dynamics.generators:
            # Extract area and bus from gen_id (format: Gen_A_30)
            parts = gen_id.split('_')
            area = parts[1]
            local_bus = int(parts[2])
            
            # Find area offset
            for area_id, area_config in self.areas.items():
                if area_id.value == area:
                    global_bus = area_config.bus_offset + local_bus
                    break
            
            # Find generator at this bus
            gen_mask = self.net.gen.bus == global_bus
            if gen_mask.any():
                gen_idx = self.net.gen.index[gen_mask][0]
                gen_powers[gen_id] = float(self.net.res_gen.at[gen_idx, 'p_mw'])
        
        # Step dynamics
        result = self.dynamics.step(dt, bus_voltages, gen_powers)
        
        # Update system frequency
        self.system_frequency_hz = result['system_frequency_hz']
        
        return result
    
    def get_system_frequency(self) -> float:
        """
        Return the current system frequency of the super-grid.
        
        Returns:
            current_frequency_hz (float): Current system frequency in hertz.
        """
        return self.system_frequency_hz
    
    def initialize_der(self, add_default: bool = True) -> DERManager:
        """
        Create and register a DERManager for the current network.
        
        Parameters:
            add_default (bool): If True, populate the manager with the module's default DER configuration.
        
        Returns:
            DERManager: The manager instance used to control and query distributed energy resources for this network.
        """
        self.der_manager = DERManager(self.net)
        
        if add_default:
            self._add_default_der()
            
        return self.der_manager

    
    def _add_default_der(self) -> None:
        """
        Populate the DERManager with a predefined set of distributed energy resources for all three areas.
        
        Adds the following default DERs:
        - Area A (bus offsets applied): solar PV at buses 3, 7, 15; a battery at bus 20; EV charging stations at buses 4 and 12; demand response at bus 8.
        - Area B (39-bus offset): wind farms at buses 39+3 and 39+8; a battery at bus 39+15; an EV charging station at bus 39+7; demand response at buses 39+20 and 39+18.
        - Area C (78-bus offset): solar PV at bus 78+4; a wind farm at bus 78+12; a battery at bus 78+25; EV charging stations at buses 78+8 and 78+16; demand response at bus 78+15.
        
        After adding resources, logs a short summary of the number of DER units and aggregated capacity by DER type.
        """
        logger.info("Adding default DER configuration...")
        
        # Area A (North) - Solar dominated with EV infrastructure
        # Add solar PV at load buses (where there's demand)
        self.der_manager.add_solar_pv(bus=3, capacity_mw=50, name="Solar_A1", area_id="A")
        self.der_manager.add_solar_pv(bus=7, capacity_mw=40, name="Solar_A2", area_id="A")
        self.der_manager.add_solar_pv(bus=15, capacity_mw=60, name="Solar_A3", area_id="A")
        self.der_manager.add_battery(bus=20, power_mw=30, energy_mwh=120, name="BESS_A1", area_id="A")
        
        # EV charging stations in Area A
        self.der_manager.add_ev_charging_station(bus=4, num_chargers=50, charger_power_kw=50, name="EV_Station_A1", area_id="A")
        self.der_manager.add_ev_charging_station(bus=12, num_chargers=30, charger_power_kw=150, name="EV_FastCharge_A1", area_id="A")
        
        # Demand response in Area A
        self.der_manager.add_demand_response(bus=8, baseline_load_mw=25, curtailable_fraction=0.3, name="DR_Industrial_A1", area_id="A")
        
        # Area B (West) - Wind dominated with commercial DR
        self.der_manager.add_wind_turbine(bus=39+3, capacity_mw=80, name="Wind_B1", area_id="B", num_turbines=20)
        self.der_manager.add_wind_turbine(bus=39+8, capacity_mw=100, name="Wind_B2", area_id="B", num_turbines=25)
        self.der_manager.add_battery(bus=39+15, power_mw=40, energy_mwh=160, name="BESS_B1", area_id="B")
        
        # EV charging in Area B
        self.der_manager.add_ev_charging_station(bus=39+7, num_chargers=40, charger_power_kw=50, name="EV_Station_B1", area_id="B")
        
        # Demand response in Area B
        self.der_manager.add_demand_response(bus=39+20, baseline_load_mw=30, curtailable_fraction=0.25, name="DR_Commercial_B1", area_id="B")
        self.der_manager.add_demand_response(bus=39+18, baseline_load_mw=20, curtailable_fraction=0.4, name="DR_Industrial_B1", area_id="B")
        
        # Area C (East) - Hybrid with residential DR
        self.der_manager.add_solar_pv(bus=78+4, capacity_mw=55, name="Solar_C1", area_id="C")
        self.der_manager.add_wind_turbine(bus=78+12, capacity_mw=70, name="Wind_C1", area_id="C", num_turbines=18)
        self.der_manager.add_battery(bus=78+25, power_mw=50, energy_mwh=200, name="BESS_C1", area_id="C")
        
        # EV charging in Area C
        self.der_manager.add_ev_charging_station(bus=78+8, num_chargers=60, charger_power_kw=50, name="EV_Station_C1", area_id="C")
        self.der_manager.add_ev_charging_station(bus=78+16, num_chargers=20, charger_power_kw=150, name="EV_FastCharge_C1", area_id="C")
        
        # Demand response in Area C
        self.der_manager.add_demand_response(bus=78+15, baseline_load_mw=35, curtailable_fraction=0.2, name="DR_Residential_C1", area_id="C")
        
        # Log summary
        der_by_type = {}
        total_capacity = {}
        for spec in self.der_manager.der_specs:
            type_name = spec.der_type.value
            der_by_type[type_name] = der_by_type.get(type_name, 0) + 1
            total_capacity[type_name] = total_capacity.get(type_name, 0) + spec.capacity_mw
        
        logger.info(f"  - Added {len(self.der_manager.der_specs)} DER units")
        for der_type, count in der_by_type.items():
            logger.info(f"  - {der_type.capitalize()}: {count} units, {total_capacity[der_type]:.1f} MW")
        
    def _update_area_mappings(self) -> None:
        """
        Populate each area's mapping information after the merged network is constructed.
        
        For each configured area, sets `bus_range` to the area's global bus index span (offset to offset+38) and fills:
        - `generator_indices` with indices of generators whose `bus` lies within that range.
        - `load_indices` with indices of loads whose `bus` lies within that range.
        """
        for area_id, area_config in self.areas.items():
            offset = area_config.bus_offset
            
            # Bus range for this area
            area_config.bus_range = (offset, offset + 38)
            
            # Find generators in this area (by their bus location)
            gen_mask = (self.net.gen.bus >= offset) & (self.net.gen.bus <= offset + 38)
            area_config.generator_indices = list(self.net.gen.index[gen_mask])
            
            # Find loads in this area
            load_mask = (self.net.load.bus >= offset) & (self.net.load.bus <= offset + 38)
            area_config.load_indices = list(self.net.load.index[load_mask])
            
            logger.debug(f"{area_config.name}: buses {area_config.bus_range}, "
                        f"{len(area_config.generator_indices)} gens, "
                        f"{len(area_config.load_indices)} loads")
    
    def _add_tie_lines(self) -> None:
        """
        Add the configured inter-area high-voltage tie-lines to the merged network.
        
        Creates overhead transmission lines on self.net for each TieLineSpec in self.tie_line_specs by computing global bus indices from area offsets and the spec's local bus indices, applying the spec's electrical parameters (length, r/x/c per km) and a 345 kV-based current rating.
        """
        logger.info(f"Adding {len(self.tie_line_specs)} inter-area tie-lines...")
        
        for spec in self.tie_line_specs:
            # Calculate global bus indices
            from_bus = self.areas[spec.from_area].bus_offset + spec.from_bus_local
            to_bus = self.areas[spec.to_area].bus_offset + spec.to_bus_local
            
            # Add tie-line with appropriate impedance for long-distance transmission
            # Using typical 345kV transmission line parameters
            pp.create_line_from_parameters(
                self.net,
                from_bus=from_bus,
                to_bus=to_bus,
                length_km=spec.length_km,
                r_ohm_per_km=spec.r_ohm_per_km,
                x_ohm_per_km=spec.x_ohm_per_km,
                c_nf_per_km=spec.c_nf_per_km,
                max_i_ka=spec.rating_mva / (345 * np.sqrt(3)),  # Current rating
                name=spec.name,
                type="ol",  # Overhead line
            )
            
            logger.debug(f"Added tie-line: {spec.name} (Bus {from_bus} <-> Bus {to_bus})")
    
    def _configure_slack_bus(self) -> None:
        """
        Configure the network slack/reference bus to Area A's bus 30 and set its voltage.
        
        Keeps only the first external grid element, assigns it to bus 30 (Area A slack) and sets its voltage setpoint to 1.03 pu.
        """
        # In IEEE 39-bus, bus 30 (index 30) is typically the slack
        # After merging, Area A's bus 30 becomes the system slack
        slack_bus = 30  # Area A's slack bus
        
        # Ensure the external grid element exists and is at the slack bus
        if len(self.net.ext_grid) > 0:
            # Keep only the first external grid (Area A's slack)
            # Remove others that came from merging
            self.net.ext_grid = self.net.ext_grid.iloc[:1].copy()
            self.net.ext_grid.at[0, 'bus'] = slack_bus
            self.net.ext_grid.at[0, 'vm_pu'] = 1.03  # Typical slack voltage
            
    def get_area_state(self, area_id: AreaID) -> Dict:
        """
        Return aggregated operational metrics for the specified area.
        
        Returns:
            dict: Aggregated area metrics with keys:
                - area_id (str): Area identifier value (e.g., "A", "B", "C").
                - area_name (str): Human-readable area name.
                - total_generation_mw (float): Sum of active power produced by generators in the area.
                - total_load_mw (float): Sum of active power consumed by loads in the area.
                - net_interchange_mw (float): Generation minus load (positive = net export).
                - avg_voltage_pu (float): Mean per-unit bus voltage across the area's buses.
                - min_voltage_pu (float): Minimum per-unit bus voltage in the area.
                - max_voltage_pu (float): Maximum per-unit bus voltage in the area.
                - num_generators (int): Number of generators in the area.
                - num_loads (int): Number of loads in the area.
        
        Raises:
            RuntimeError: If the network is uninitialized or a power flow solution is not available.
        """
        if self.net is None or self.net.res_bus.empty:
            raise RuntimeError("Power flow has not been run yet")
            
        area = self.areas[area_id]
        bus_start, bus_end = area.bus_range
        
        # Get bus results for this area
        area_buses = self.net.res_bus.loc[bus_start:bus_end]
        
        # Get generator results for this area
        area_gen_results = self.net.res_gen.loc[area.generator_indices]
        
        # Get load results for this area  
        area_load_results = self.net.res_load.loc[area.load_indices]
        
        # Calculate net interchange (generation - load = export)
        total_gen = area_gen_results.p_mw.sum()
        total_load = area_load_results.p_mw.sum()
        
        return {
            "area_id": area_id.value,
            "area_name": area.name,
            "total_generation_mw": float(total_gen),
            "total_load_mw": float(total_load),
            "net_interchange_mw": float(total_gen - total_load),
            "avg_voltage_pu": float(area_buses.vm_pu.mean()),
            "min_voltage_pu": float(area_buses.vm_pu.min()),
            "max_voltage_pu": float(area_buses.vm_pu.max()),
            "num_generators": len(area.generator_indices),
            "num_loads": len(area.load_indices),
        }
    
    def get_tie_line_flows(self) -> List[Dict]:
        """
        Return flow measurements and status for each inter-area tie-line.
        
        Returns:
            tie_line_flows (List[Dict]): A list where each dictionary contains metrics for one tie-line:
                - name: Tie-line identifier (str).
                - from_bus: Sending-end bus index (int).
                - to_bus: Receiving-end bus index (int).
                - p_from_mw: Active power injected at the sending end in MW (float).
                - p_to_mw: Active power at the receiving end in MW (float).
                - q_from_mvar: Reactive power injected at the sending end in MVAR (float).
                - loading_percent: Line loading as a percentage of its thermal limit (float).
                - is_overloaded: `True` if loading_percent > 100.0, `False` otherwise.
        """
        if self.net is None or self.net.res_line.empty:
            raise RuntimeError("Power flow has not been run yet")
            
        tie_line_flows = []
        
        # Tie-lines are the last N lines added to the network
        num_tie_lines = len(self.tie_line_specs)
        tie_line_indices = self.net.line.index[-num_tie_lines:]
        
        for idx in tie_line_indices:
            line_result = self.net.res_line.loc[idx]
            line_data = self.net.line.loc[idx]
            
            tie_line_flows.append({
                "name": line_data["name"],
                "from_bus": int(line_data["from_bus"]),
                "to_bus": int(line_data["to_bus"]),
                "p_from_mw": float(line_result.p_from_mw),
                "p_to_mw": float(line_result.p_to_mw),
                "q_from_mvar": float(line_result.q_from_mvar),
                "loading_percent": float(line_result.loading_percent),
                "is_overloaded": bool(line_result.loading_percent > 100.0),
            })
            
        return tie_line_flows
    
    def get_global_state(self) -> Dict:
        """
        Return an observation dictionary representing the current super-grid state for RL agents.
        
        The returned structure includes per-area aggregations, tie-line flows, and system-wide metrics derived from the last solved power flow and current dynamics state.
        
        Returns:
            global_state (dict): A dictionary with keys:
                - "areas": mapping of area id (str) to area state dict (contains generation, load, voltages, counts).
                - "tie_lines": list of tie-line flow dicts (power, reactive, loading, overload flag).
                - "global_metrics": dict containing:
                    - "total_generation_mw": sum of area generation (MW).
                    - "total_load_mw": sum of area loads (MW).
                    - "total_losses_mw": generation minus load (MW).
                    - "system_frequency_hz": current system frequency (Hz).
                    - "has_voltage_violations": `true` if any area voltage is outside configured limits, `false` otherwise.
                    - "has_thermal_violations": `true` if any tie-line is overloaded, `false` otherwise.
                    - "frequency_deviation_hz": deviation from nominal 50.0 Hz.
                    - "dynamics_enabled": `true` if dynamic simulation is initialized, `false` otherwise.
        
        Raises:
            RuntimeError: If the super-grid network is not initialized.
        """
        if self.net is None:
            raise RuntimeError("Grid not initialized")
            
        # Get state for each area
        area_states = {
            area_id.value: self.get_area_state(area_id)
            for area_id in AreaID
        }
        
        # Get tie-line flows
        tie_line_flows = self.get_tie_line_flows()
        
        # Calculate global metrics
        total_gen = sum(a["total_generation_mw"] for a in area_states.values())
        total_load = sum(a["total_load_mw"] for a in area_states.values())
        total_losses = total_gen - total_load
        
        # Check for any violations
        voltage_violations = any(
            a["min_voltage_pu"] < self.config.v_min_pu or 
            a["max_voltage_pu"] > self.config.v_max_pu
            for a in area_states.values()
        )
        thermal_violations = any(tl["is_overloaded"] for tl in tie_line_flows)
        
        return {
            "areas": area_states,
            "tie_lines": tie_line_flows,
            "global_metrics": {
                "total_generation_mw": total_gen,
                "total_load_mw": total_load,
                "total_losses_mw": total_losses,
                "system_frequency_hz": self.system_frequency_hz,  # From dynamics
                "has_voltage_violations": voltage_violations,
                "has_thermal_violations": thermal_violations,
                "frequency_deviation_hz": self.system_frequency_hz - 50.0,
                "dynamics_enabled": self.dynamics is not None,
            }
        }
    
    def set_generator_setpoint(self, gen_idx: int, p_mw: float) -> None:
        """
        Set the active power setpoint for a generator, clamping it to the generator's minimum and maximum limits.
        
        Parameters:
            gen_idx (int): Index of the generator in self.net.gen.
            p_mw (float): Desired active power setpoint in MW.
        
        Raises:
            ValueError: If the specified generator index does not exist in the network.
        """
        if gen_idx not in self.net.gen.index:
            raise ValueError(f"Generator {gen_idx} does not exist")
            
        # Respect generator limits
        p_max = self.net.gen.at[gen_idx, 'max_p_mw']
        p_min = self.net.gen.at[gen_idx, 'min_p_mw']
        
        self.net.gen.at[gen_idx, 'p_mw'] = np.clip(p_mw, p_min, p_max)
        
    def set_area_generation(self, area_id: AreaID, target_mw: float) -> None:
        """
        Distribute a target generation level across all generators in an area
        Uses economic dispatch (proportional to capacity)
        
        Args:
            area_id: Which area to control
            target_mw: Total active power target for the area
        """
        area = self.areas[area_id]
        gen_indices = area.generator_indices
        
        if not gen_indices:
            logger.warning(f"No generators in {area.name}")
            return
            
        # Get generator capacities
        capacities = self.net.gen.loc[gen_indices, 'max_p_mw'].values
        total_capacity = capacities.sum()
        
        if total_capacity == 0:
            logger.warning(f"Total capacity is zero in {area.name}")
            return
            
        # Proportional dispatch based on capacity
        dispatch_ratios = capacities / total_capacity
        setpoints = dispatch_ratios * target_mw
        
        # Apply setpoints
        for gen_idx, setpoint in zip(gen_indices, setpoints):
            self.set_generator_setpoint(gen_idx, setpoint)
            
        logger.debug(f"Set {area.name} generation to {target_mw:.1f} MW "
                    f"(distributed across {len(gen_indices)} generators)")
    
    def scale_loads(self, area_id: AreaID, scale_factor: float) -> None:
        """
        Scale all loads in the specified area by a multiplicative factor.
        
        Parameters:
            area_id (AreaID): The area whose loads will be scaled.
            scale_factor (float): Multiplicative factor applied to each load's active power (`p_mw`) and reactive power (`q_mvar`) (1.0 = no change).
        """
        area = self.areas[area_id]
        
        for load_idx in area.load_indices:
            current_p = self.net.load.at[load_idx, 'p_mw']
            current_q = self.net.load.at[load_idx, 'q_mvar']
            
            self.net.load.at[load_idx, 'p_mw'] = current_p * scale_factor
            self.net.load.at[load_idx, 'q_mvar'] = current_q * scale_factor
            
    def get_controllable_generators(self) -> Dict[str, List[Dict]]:
        """
        Collects controllable generator metadata grouped by area.
        
        Returns:
            dict: Mapping from area ID string to a list of generator info dictionaries. Each generator dictionary contains:
                - `index` (int): generator table index
                - `bus` (int): associated bus index
                - `p_mw` (float): current active power output in MW
                - `max_p_mw` (float): maximum active power in MW
                - `min_p_mw` (float): minimum active power in MW
                - `vm_pu` (float): voltage setpoint in per unit
                - `in_service` (bool): whether the generator is in service
        """
        result = {}
        
        for area_id, area in self.areas.items():
            generators = []
            
            for gen_idx in area.generator_indices:
                gen_data = self.net.gen.loc[gen_idx]
                
                generators.append({
                    "index": int(gen_idx),
                    "bus": int(gen_data["bus"]),
                    "p_mw": float(gen_data["p_mw"]),
                    "max_p_mw": float(gen_data["max_p_mw"]),
                    "min_p_mw": float(gen_data["min_p_mw"]),
                    "vm_pu": float(gen_data["vm_pu"]),
                    "in_service": bool(gen_data["in_service"]),
                })
                
            result[area_id.value] = generators
            
        return result
    
    def reset_to_base_case(self) -> None:
        """Reset the grid to the original IEEE 39-bus base case values"""
        logger.info("Resetting grid to base case...")
        
        # Rebuild the grid from scratch
        self._build_supergrid()
        
        # Reset instance variables to initial state
        self.der_manager = None
        self.dynamics = None
        self.system_frequency_hz = 50.0
        
        # Ensure grid state is consistent after reset
        self._improve_convergence()
        
    def __repr__(self) -> str:
        """
        Provide a concise string describing the SuperGrid's initialization state and key component counts.
        
        Returns:
            A string describing whether the super-grid is initialized. If initialized, includes the number of buses, generators, lines, and configured tie-lines; otherwise returns "SuperGrid(not initialized)".
        """
        if self.net is None:
            return "SuperGrid(not initialized)"
        return (f"SuperGrid(buses={len(self.net.bus)}, "
                f"generators={len(self.net.gen)}, "
                f"lines={len(self.net.line)}, "
                f"tie_lines={len(self.tie_line_specs)})")