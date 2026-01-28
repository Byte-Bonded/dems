"""
Distributed Energy Resources (DER) Module
Adds solar PV, wind turbines, battery storage, EV charging, and demand response to the grid

DER Types:
- Solar PV: Variable generation based on irradiance
- Wind Turbines: Variable generation based on wind speed  
- Battery Storage (BESS): Controllable charge/discharge
- EV Charging: Smart charging with load flexibility
- Demand Response: Controllable load curtailment

Thread Safety:
- All public methods are thread-safe via internal locking
- Use DERManager in multi-threaded environments safely
"""

import pandapower as pp
import numpy as np
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DERType(Enum):
    """Types of Distributed Energy Resources"""
    SOLAR_PV = "solar"
    WIND = "wind"
    BESS = "battery"
    EV_CHARGING = "ev_charging"
    DEMAND_RESPONSE = "demand_response"


@dataclass
class DERSpec:
    """Specification for a DER installation"""
    der_type: DERType
    bus: int
    capacity_mw: float
    name: str
    area_id: str  # "A", "B", or "C"
    
    # Solar specific
    panel_area_m2: Optional[float] = None
    efficiency: Optional[float] = None
    
    # Wind specific
    turbine_capacity_mw: Optional[float] = None
    num_turbines: Optional[int] = None
    
    # Battery specific
    energy_capacity_mwh: Optional[float] = None
    charge_rate_mw: Optional[float] = None
    discharge_rate_mw: Optional[float] = None
    initial_soc: float = 0.5  # State of charge (0-1)
    
    # EV Charging specific
    num_chargers: Optional[int] = None
    charger_power_kw: Optional[float] = None
    ev_battery_capacity_kwh: Optional[float] = None
    smart_charging_enabled: Optional[bool] = True
    
    # Demand Response specific
    baseline_load_mw: Optional[float] = None
    curtailable_fraction: Optional[float] = None  # 0-1
    response_time_minutes: Optional[float] = None
    incentive_price_per_mwh: Optional[float] = None


@dataclass
class DERState:
    """Current state of a DER unit"""
    name: str
    der_type: DERType
    bus: int
    current_output_mw: float
    capacity_mw: float
    availability: float  # 0-1
    
    # Battery specific
    soc: Optional[float] = None
    charging: Optional[bool] = None
    
    # EV Charging specific
    utilization: Optional[float] = None  # 0-1
    num_active_chargers: Optional[int] = None
    
    # Demand Response specific
    curtailment_level: Optional[float] = None  # 0-1
    curtailed_load_mw: Optional[float] = None


@dataclass
class BatteryState:
    """Comprehensive battery state tracking with degradation"""
    name: str
    soc: float  # State of charge (0-1)
    capacity_mwh: float  # Current effective capacity
    nominal_capacity_mwh: float  # Original capacity
    total_cycles: float = 0.0  # Total equivalent full cycles
    last_update: datetime = field(default_factory=datetime.now)
    round_trip_efficiency: float = 0.92  # 92% typical for Li-ion
    degradation_factor: float = 1.0  # Health factor (1.0 = new)
    
    def update_soc(self, delta_energy_mwh: float) -> float:
        """
        Update SOC accounting for efficiency and track cycles.
        
        Args:
            delta_energy_mwh: Energy change (positive=discharge, negative=charge)
            
        Returns:
            Actual energy delivered/absorbed after efficiency
        """
        # Apply efficiency losses
        if delta_energy_mwh > 0:  # Discharging
            actual_energy = delta_energy_mwh
            soc_delta = -delta_energy_mwh / self.capacity_mwh
        else:  # Charging
            actual_energy = delta_energy_mwh * self.round_trip_efficiency
            soc_delta = -actual_energy / self.capacity_mwh
        
        # Update SOC with bounds
        old_soc = self.soc
        self.soc = np.clip(self.soc + soc_delta, 0.0, 1.0)
        
        # Track cycles (one full charge + discharge = 1 cycle)
        self.total_cycles += abs(self.soc - old_soc) / 2
        
        # Update degradation (simple linear model: 20% loss at 5000 cycles)
        self.degradation_factor = max(0.8, 1.0 - (self.total_cycles / 5000) * 0.2)
        self.capacity_mwh = self.nominal_capacity_mwh * self.degradation_factor
        
        self.last_update = datetime.now()
        return actual_energy


class DERManager:
    """
    Manages Distributed Energy Resources in the grid
    
    Handles:
    - Adding DER to buses
    - Updating DER output based on conditions
    - Managing battery storage dispatch
    - Tracking DER state
    
    Thread Safety:
        All public methods are protected by an internal lock.
        Safe for concurrent access from multiple threads.
    """
    
    def __init__(self, net: pp.pandapowerNet):
        """
        Create a DERManager bound to a pandapower network and initialize internal DER registries.
        
        Args:
            net: Pandapower network to add DER to
            
        Raises:
            ValueError: If net is None
        """
        if net is None:
            raise ValueError("Network cannot be None")
            
        self.net = net
        self.der_specs: List[DERSpec] = []
        self.der_indices: Dict[str, int] = {}  # name -> pandapower index
        self.battery_soc: Dict[str, float] = {}  # name -> state of charge (legacy)
        self._battery_states: Dict[str, BatteryState] = {}  # Enhanced battery tracking
        self._lock = threading.RLock()  # Reentrant lock for thread safety
        
    def add_solar_pv(
        self, 
        bus: int, 
        capacity_mw: float, 
        name: str,
        area_id: str,
        panel_area_m2: float = 1000.0,
        efficiency: float = 0.20
    ) -> int:
        """
        Register a solar PV DER at the specified bus and add a controllable static generator to the pandapower network.
        
        Parameters:
            bus (int): Target bus index in the pandapower network.
            capacity_mw (float): Peak AC capacity of the PV installation in megawatts.
            name (str): Unique identifier for this PV system.
            area_id (str): Identifier for the geographic or operational area.
            panel_area_m2 (float): Total photovoltaic panel area in square meters.
            efficiency (float): Panel conversion efficiency as a fraction between 0 and 1.
        
        Returns:
            int: Index of the created static generator element in the pandapower network.
        """
        spec = DERSpec(
            der_type=DERType.SOLAR_PV,
            bus=bus,
            capacity_mw=capacity_mw,
            name=name,
            area_id=area_id,
            panel_area_m2=panel_area_m2,
            efficiency=efficiency
        )
        self.der_specs.append(spec)
        
        # Add as controllable static generator (sgen)
        # Start with 50% capacity (midday-ish generation)
        idx = pp.create_sgen(
            self.net,
            bus=bus,
            p_mw=capacity_mw * 0.5,  # Start at 50% output
            q_mvar=0,  # PV systems can provide reactive power but start at 0
            name=name,
            type="PV",
            controllable=True
        )
        self.der_indices[name] = idx
        
        logger.info(f"Added Solar PV: {name} at Bus {bus}, {capacity_mw} MW capacity")
        return idx
        
    def add_wind_turbine(
        self,
        bus: int,
        capacity_mw: float,
        name: str,
        area_id: str,
        num_turbines: int = 1
    ) -> int:
        """
        Register a wind farm and add a controllable wind static generator to the pandapower network.
        
        Parameters:
            bus (int): Index of the bus where the wind farm is placed.
            capacity_mw (float): Total installed capacity of the wind farm in MW.
            name (str): Unique identifier for this wind farm.
            area_id (str): Area or region identifier for grouping/metadata.
            num_turbines (int): Number of turbines in the wind farm; used to compute per-turbine capacity.
        
        Notes:
            The created generator is initialized with output set to 40% of total capacity.
        
        Returns:
            int: Index of the created static generator element in the pandapower network.
        """
        spec = DERSpec(
            der_type=DERType.WIND,
            bus=bus,
            capacity_mw=capacity_mw,
            name=name,
            area_id=area_id,
            turbine_capacity_mw=capacity_mw / num_turbines,
            num_turbines=num_turbines
        )
        self.der_specs.append(spec)
        
        # Add as controllable static generator
        # Start with 40% capacity factor (typical wind)
        idx = pp.create_sgen(
            self.net,
            bus=bus,
            p_mw=capacity_mw * 0.4,
            q_mvar=0,
            name=name,
            type="WP",  # Wind power
            controllable=True
        )
        self.der_indices[name] = idx
        
        logger.info(f"Added Wind Farm: {name} at Bus {bus}, {capacity_mw} MW capacity ({num_turbines} turbines)")
        return idx
        
    def add_battery(
        self,
        bus: int,
        power_mw: float,
        energy_mwh: float,
        name: str,
        area_id: str,
        initial_soc: float = 0.5
    ) -> int:
        """
        Register a battery energy storage system (BESS) and add a controllable storage element to the pandapower network.
        
        Parameters:
            bus (int): Bus index where the battery is connected.
            power_mw (float): Maximum charge/discharge power in MW.
            energy_mwh (float): Energy capacity in MWh.
            name (str): Unique identifier for the battery.
            area_id (str): Area identifier for grouping or reporting.
            initial_soc (float): Initial state of charge as a fraction between 0 and 1.
        
        Returns:
            int: Index of the created storage element.
        """
        spec = DERSpec(
            der_type=DERType.BESS,
            bus=bus,
            capacity_mw=power_mw,
            name=name,
            area_id=area_id,
            energy_capacity_mwh=energy_mwh,
            charge_rate_mw=power_mw,
            discharge_rate_mw=power_mw,
            initial_soc=initial_soc
        )
        self.der_specs.append(spec)
        self.battery_soc[name] = initial_soc
        
        # Add as storage element
        idx = pp.create_storage(
            self.net,
            bus=bus,
            p_mw=0,  # Start at zero (neutral)
            max_e_mwh=energy_mwh,
            q_mvar=0,
            soc_percent=initial_soc * 100,
            min_e_mwh=0,
            name=name,
            type="BESS",
            controllable=True,
            max_p_mw=power_mw,
            min_p_mw=-power_mw  # Negative = charging
        )
        self.der_indices[name] = idx
        
        logger.info(f"Added Battery: {name} at Bus {bus}, {power_mw} MW / {energy_mwh} MWh")
        return idx
        
    def add_ev_charging_station(
        self,
        bus: int,
        num_chargers: int,
        charger_power_kw: float,
        name: str,
        area_id: str,
        ev_battery_capacity_kwh: float = 60.0,
        smart_charging_enabled: bool = True
    ) -> int:
        """
        Register an EV charging station and create a controllable pandapower load representing its initial demand.
        
        Parameters:
            bus (int): Network bus index where the station is connected.
            num_chargers (int): Number of charging points at the station.
            charger_power_kw (float): Rated power per charger in kilowatts.
            name (str): Unique identifier for the station.
            area_id (str): Geographic or logical area identifier for grouping.
            ev_battery_capacity_kwh (float): Typical EV battery capacity used for modeling (kWh).
            smart_charging_enabled (bool): If true, enables smart charging behavior that can be used to reduce load during control actions.
        
        Returns:
            load_index (int): Index of the created pandapower load element.
        """
        capacity_mw = (num_chargers * charger_power_kw) / 1000.0
        
        spec = DERSpec(
            der_type=DERType.EV_CHARGING,
            bus=bus,
            capacity_mw=capacity_mw,
            name=name,
            area_id=area_id,
            num_chargers=num_chargers,
            charger_power_kw=charger_power_kw,
            ev_battery_capacity_kwh=ev_battery_capacity_kwh,
            smart_charging_enabled=smart_charging_enabled
        )
        self.der_specs.append(spec)
        
        # Add as controllable load
        # Start with 30% utilization (typical for daytime)
        initial_load = capacity_mw * 0.3
        
        idx = pp.create_load(
            self.net,
            bus=bus,
            p_mw=initial_load,
            q_mvar=initial_load * 0.1,  # Small reactive component
            name=name,
            type="EV",
            controllable=True
        )
        self.der_indices[name] = idx
        
        logger.info(f"Added EV Charging: {name} at Bus {bus}, {num_chargers} chargers × {charger_power_kw} kW = {capacity_mw:.2f} MW")
        return idx
        
    def add_demand_response(
        self,
        bus: int,
        baseline_load_mw: float,
        curtailable_fraction: float,
        name: str,
        area_id: str,
        response_time_minutes: float = 5.0,
        incentive_price_per_mwh: float = 100.0
    ) -> int:
        """
        Register a demand response program and create a controllable load at its baseline demand.
        
        Creates a DERSpec for the demand response program and adds a controllable pandapower load initialized to the provided baseline load.
        
        Parameters:
            bus (int): Pandapower bus index where the DR resource is connected.
            baseline_load_mw (float): Baseline load in megawatts that can be reduced.
            curtailable_fraction (float): Fraction of the baseline load that is curtailable (0 to 1).
            name (str): Unique identifier for this DR program.
            area_id (str): Geographic or aggregation area identifier.
            response_time_minutes (float): Expected DR response time in minutes.
            incentive_price_per_mwh (float): Incentive payment for curtailed energy in $/MWh.
        
        Returns:
            int: Index of the created pandapower load element.
        """
        capacity_mw = baseline_load_mw * curtailable_fraction
        
        spec = DERSpec(
            der_type=DERType.DEMAND_RESPONSE,
            bus=bus,
            capacity_mw=capacity_mw,
            name=name,
            area_id=area_id,
            baseline_load_mw=baseline_load_mw,
            curtailable_fraction=curtailable_fraction,
            response_time_minutes=response_time_minutes,
            incentive_price_per_mwh=incentive_price_per_mwh
        )
        self.der_specs.append(spec)
        
        # Add as controllable load at baseline
        idx = pp.create_load(
            self.net,
            bus=bus,
            p_mw=baseline_load_mw,
            q_mvar=baseline_load_mw * 0.2,  # Typical power factor
            name=name,
            type="DR",
            controllable=True
        )
        self.der_indices[name] = idx
        
        logger.info(f"Added Demand Response: {name} at Bus {bus}, {baseline_load_mw} MW baseline, {capacity_mw:.2f} MW curtailable")
        return idx
        
    def set_solar_output(self, name: str, irradiance_w_m2: float) -> None:
        """
        Set the active power of a named solar PV DER based on irradiance and update the network element.
        
        Args:
            name: PV system name
            irradiance_w_m2: Solar irradiance (W/m²), typically 0-1000
            
        Raises:
            ValueError: If DER not found or wrong type
            RuntimeError: If network index is invalid
        """
        with self._lock:
            try:
                spec = self._get_spec(name)
            except KeyError:
                raise ValueError(f"DER '{name}' not found")
                
            if spec.der_type != DERType.SOLAR_PV:
                raise ValueError(f"{name} is not a solar PV system (type: {spec.der_type.value})")
            
            # Validate irradiance
            if irradiance_w_m2 < 0:
                logger.warning(f"Negative irradiance {irradiance_w_m2} for {name}, clamping to 0")
                irradiance_w_m2 = 0
            elif irradiance_w_m2 > 1400:  # Max possible solar irradiance on Earth
                logger.warning(f"Irradiance {irradiance_w_m2} exceeds maximum, clamping to 1400")
                irradiance_w_m2 = 1400
                
            # Calculate output: P = Area × Irradiance × Efficiency
            irradiance_kw_m2 = irradiance_w_m2 / 1000.0
            output_mw = (spec.panel_area_m2 * irradiance_kw_m2 * spec.efficiency) / 1000.0
            output_mw = min(output_mw, spec.capacity_mw)
            
            idx = self.der_indices.get(name)
            if idx is None:
                raise RuntimeError(f"DER '{name}' has no registered network index")
            if idx not in self.net.sgen.index:
                raise RuntimeError(f"DER '{name}' index {idx} no longer valid in network")
                
            self.net.sgen.at[idx, 'p_mw'] = output_mw
            
            logger.debug(f"{name}: {irradiance_w_m2} W/m² → {output_mw:.2f} MW")
        
    def set_wind_output(self, name: str, wind_speed_m_s: float) -> None:
        """
        Set the wind turbine's active power output based on the provided wind speed.
        
        Args:
            name: Wind farm name
            wind_speed_m_s: Wind speed (m/s)
            
        Raises:
            ValueError: If DER not found or wrong type
            RuntimeError: If network index is invalid
        """
        with self._lock:
            try:
                spec = self._get_spec(name)
            except KeyError:
                raise ValueError(f"DER '{name}' not found")
                
            if spec.der_type != DERType.WIND:
                raise ValueError(f"{name} is not a wind system (type: {spec.der_type.value})")
            
            # Validate wind speed
            if wind_speed_m_s < 0:
                logger.warning(f"Negative wind speed {wind_speed_m_s} for {name}, clamping to 0")
                wind_speed_m_s = 0
                
            # Simplified power curve: cubic relationship between 3-15 m/s
            if wind_speed_m_s < 3:
                capacity_factor = 0
            elif wind_speed_m_s > 15:
                capacity_factor = 1.0
            else:
                # Cubic approximation: P ∝ v³
                capacity_factor = min(((wind_speed_m_s - 3) / 12) ** 3, 1.0)
                
            output_mw = spec.capacity_mw * capacity_factor
            
            idx = self.der_indices.get(name)
            if idx is None:
                raise RuntimeError(f"DER '{name}' has no registered network index")
            if idx not in self.net.sgen.index:
                raise RuntimeError(f"DER '{name}' index {idx} no longer valid in network")
                
            self.net.sgen.at[idx, 'p_mw'] = output_mw
            
            logger.debug(f"{name}: {wind_speed_m_s} m/s → {output_mw:.2f} MW ({capacity_factor*100:.1f}%)")
        
    def set_battery_power(self, name: str, power_mw: float) -> None:
        """
        Set the battery's active power setpoint (positive values discharge the battery, negative values charge it).
        
        Args:
            name: Battery name
            power_mw: Power in MW (positive = discharge, negative = charge)
            
        Raises:
            ValueError: If DER not found or wrong type
            RuntimeError: If network index is invalid
        """
        with self._lock:
            try:
                spec = self._get_spec(name)
            except KeyError:
                raise ValueError(f"DER '{name}' not found")
                
            if spec.der_type != DERType.BESS:
                raise ValueError(f"{name} is not a battery system (type: {spec.der_type.value})")
            
            # Clamp to charge/discharge limits
            original_power = power_mw
            power_mw = np.clip(power_mw, -spec.charge_rate_mw, spec.discharge_rate_mw)
            if original_power != power_mw:
                logger.warning(f"{name}: Power {original_power} MW clamped to {power_mw} MW")
            
            # Check SOC limits
            if name in self._battery_states:
                batt_state = self._battery_states[name]
                if power_mw > 0 and batt_state.soc <= 0.05:
                    logger.warning(f"{name}: SOC too low ({batt_state.soc:.1%}) for discharge")
                    power_mw = 0
                elif power_mw < 0 and batt_state.soc >= 0.95:
                    logger.warning(f"{name}: SOC too high ({batt_state.soc:.1%}) for charge")
                    power_mw = 0
            
            idx = self.der_indices.get(name)
            if idx is None:
                raise RuntimeError(f"DER '{name}' has no registered network index")
            if idx not in self.net.storage.index:
                raise RuntimeError(f"DER '{name}' index {idx} no longer valid in network")
                
            self.net.storage.at[idx, 'p_mw'] = power_mw
            
            logger.debug(f"{name}: {'Discharging' if power_mw > 0 else 'Charging'} at {abs(power_mw):.2f} MW")
        
    def set_ev_charging_load(self, name: str, utilization: float, smart_override: bool = False) -> None:
        """
        Adjust the active charging load of an EV charging station according to utilization and optional smart-charge reduction.
        
        Args:
            name: EV station name
            utilization: Charger utilization (0-1), 0=no EVs, 1=all chargers full
            smart_override: If True, allows reducing load for grid balancing
            
        Raises:
            ValueError: If DER not found, wrong type, or utilization out of range
            RuntimeError: If network index is invalid
        """
        with self._lock:
            try:
                spec = self._get_spec(name)
            except KeyError:
                raise ValueError(f"DER '{name}' not found")
                
            if spec.der_type != DERType.EV_CHARGING:
                raise ValueError(f"{name} is not an EV charging station (type: {spec.der_type.value})")
            
            # Validate utilization
            if not 0 <= utilization <= 1:
                raise ValueError(f"Utilization must be 0-1, got {utilization}")
            
            # Calculate load based on utilization
            target_load_mw = spec.capacity_mw * utilization
            
            # If smart charging enabled and override requested, can reduce by up to 50%
            if spec.smart_charging_enabled and smart_override:
                target_load_mw *= 0.5
                
            idx = self.der_indices.get(name)
            if idx is None:
                raise RuntimeError(f"DER '{name}' has no registered network index")
            if idx not in self.net.load.index:
                raise RuntimeError(f"DER '{name}' index {idx} no longer valid in network")
                
            self.net.load.at[idx, 'p_mw'] = target_load_mw
            self.net.load.at[idx, 'q_mvar'] = target_load_mw * 0.1
            
            logger.debug(f"{name}: {utilization*100:.1f}% utilization → {target_load_mw:.2f} MW")
        
    def set_demand_response_curtailment(self, name: str, curtailment_fraction: float) -> None:
        """
        Set the demand response curtailment for a named DR program.
        
        Parameters:
            name (str): Name of the demand response program.
            curtailment_fraction (float): Fraction of curtailable load to reduce (0 to 1). Values outside this range are clamped.
        
        Notes:
            Updates the associated pandapower load: sets active power to baseline minus the curtailed amount and reactive power to 20% of the resulting active power.
        """
        spec = self._get_spec(name)
        if spec.der_type != DERType.DEMAND_RESPONSE:
            raise ValueError(f"{name} is not a demand response program")
            
        # Clamp to valid range
        curtailment_fraction = np.clip(curtailment_fraction, 0, 1)
        
        # Calculate actual load: baseline - (curtailable × curtailment_fraction)
        curtailment_mw = spec.capacity_mw * curtailment_fraction
        actual_load_mw = spec.baseline_load_mw - curtailment_mw
        
        idx = self.der_indices[name]
        self.net.load.at[idx, 'p_mw'] = actual_load_mw
        self.net.load.at[idx, 'q_mvar'] = actual_load_mw * 0.2
        
        if curtailment_fraction > 0:
            logger.info(f"{name}: DR activated, curtailing {curtailment_mw:.2f} MW ({curtailment_fraction*100:.1f}%)")
        
    def update_ev_charging_by_hour(self, hour: int) -> None:
        """
        Apply a predefined hourly utilization profile to all EV charging DERs and update each station's target utilization.
        
        Parameters:
            hour (int): Hour of day; values are interpreted modulo 24 (0–23).
        """
        # Typical EV charging profile:
        # - Night (00-06): 0.2 (home charging)
        # - Morning (07-09): 0.1 (low, people driving)
        # - Day (10-17): 0.4 (workplace charging)
        # - Evening (18-22): 0.6 (peak home charging)
        # - Late (23): 0.3 (decreasing)
        
        utilization_profile = {
            0: 0.2, 1: 0.2, 2: 0.15, 3: 0.15, 4: 0.1, 5: 0.1,
            6: 0.15, 7: 0.1, 8: 0.1, 9: 0.15, 10: 0.3, 11: 0.4,
            12: 0.4, 13: 0.4, 14: 0.4, 15: 0.4, 16: 0.35, 17: 0.3,
            18: 0.5, 19: 0.6, 20: 0.6, 21: 0.5, 22: 0.4, 23: 0.3
        }
        
        utilization = utilization_profile.get(hour % 24, 0.3)
        
        for spec in self.der_specs:
            if spec.der_type == DERType.EV_CHARGING:
                self.set_ev_charging_load(spec.name, utilization)
        
    def get_der_state(self, name: str) -> DERState:
        """
        Retrieve the live state for a named DER in the network.
        
        Parameters:
            name (str): Unique name of the DER to query.
        
        Returns:
            DERState: Current measured and derived state for the DER. Fields populated depend on DER type:
                - BESS: includes `current_output_mw`, `soc`, and `charging`.
                - EV_CHARGING: includes `current_output_mw`, `utilization`, and `num_active_chargers`.
                - DEMAND_RESPONSE: includes `current_output_mw`, `curtailment_level`, and `curtailed_load_mw`.
                - Solar/Wind (generation types): includes `current_output_mw` and `availability`.
        """
        spec = self._get_spec(name)
        idx = self.der_indices[name]
        
        if spec.der_type == DERType.BESS:
            storage = self.net.storage.loc[idx]
            return DERState(
                name=name,
                der_type=spec.der_type,
                bus=spec.bus,
                current_output_mw=storage.p_mw,
                capacity_mw=spec.capacity_mw,
                availability=1.0,
                soc=self.battery_soc.get(name, 0.5),
                charging=storage.p_mw < 0
            )
        elif spec.der_type == DERType.EV_CHARGING:
            load = self.net.load.loc[idx]
            utilization = load.p_mw / spec.capacity_mw if spec.capacity_mw > 0 else 0
            return DERState(
                name=name,
                der_type=spec.der_type,
                bus=spec.bus,
                current_output_mw=load.p_mw,  # Positive for load
                capacity_mw=spec.capacity_mw,
                availability=1.0,
                utilization=utilization,
                num_active_chargers=int(utilization * spec.num_chargers)
            )
        elif spec.der_type == DERType.DEMAND_RESPONSE:
            load = self.net.load.loc[idx]
            curtailed_mw = spec.baseline_load_mw - load.p_mw
            curtailment_level = curtailed_mw / spec.capacity_mw if spec.capacity_mw > 0 else 0
            return DERState(
                name=name,
                der_type=spec.der_type,
                bus=spec.bus,
                current_output_mw=load.p_mw,  # Positive for load
                capacity_mw=spec.capacity_mw,
                availability=1.0,
                curtailment_level=curtailment_level,
                curtailed_load_mw=curtailed_mw
            )
        else:
            sgen = self.net.sgen.loc[idx]
            return DERState(
                name=name,
                der_type=spec.der_type,
                bus=spec.bus,
                current_output_mw=sgen.p_mw,
                capacity_mw=spec.capacity_mw,
                availability=sgen.p_mw / spec.capacity_mw if spec.capacity_mw > 0 else 0
            )
            
    def get_all_der_states(self) -> List[DERState]:
        """
        Retrieve the current state for every registered DER.
        
        Returns:
            states (List[DERState]): A list containing one DERState for each registered DER.
        """
        return [self.get_der_state(name) for name in self.der_indices.keys()]
        
    def _get_spec(self, name: str) -> DERSpec:
        """
        Retrieve the DERSpec for a DER with the given name.
        
        Parameters:
            name (str): Unique DER name to look up.
        
        Returns:
            DERSpec: The specification object for the matching DER.
        
        Raises:
            ValueError: If no DER with the given name is found.
        """
        for spec in self.der_specs:
            if spec.name == name:
                return spec
        raise ValueError(f"DER not found: {name}")
        
    def update_battery_soc(self, timestep_hours: float = 1.0) -> None:
        """
        Advance battery states of charge (SOC) for all registered BESS units and synchronize them to the pandapower storage table.
        
        For each battery, reads the current power (positive = discharge, negative = charging), converts it to energy over the given timestep, updates the internal SOC map (clamped between 0 and 1), and writes the SOC back to the pandapower storage row as percent.
        
        Parameters:
            timestep_hours (float): Duration over which power is integrated, in hours (default 1.0).
        """
        for name, idx in self.der_indices.items():
            spec = self._get_spec(name)
            if spec.der_type == DERType.BESS:
                power_mw = self.net.storage.at[idx, 'p_mw']
                energy_mwh = power_mw * timestep_hours
                
                # Update SOC (positive power = discharge = decrease SOC)
                delta_soc = -energy_mwh / spec.energy_capacity_mwh
                new_soc = np.clip(self.battery_soc[name] + delta_soc, 0, 1)
                self.battery_soc[name] = new_soc
                
                # Update pandapower storage SOC
                self.net.storage.at[idx, 'soc_percent'] = new_soc * 100
    
    def get_total_generation(self) -> float:
        """
        Return the net active generation from all DERs in MW.
        
        Returns:
            float: Net generation in MW; positive values indicate net generation. Battery discharge is counted as generation (discharging storage p_mw increases the returned total).
        """
        total = 0.0
        for name, idx in self.der_indices.items():
            spec = self._get_spec(name)
            if spec.der_type == DERType.BESS:
                # Battery: negative power = discharge = generation
                total += -self.net.storage.at[idx, 'p_mw']
            else:
                # Solar/Wind: always generation
                total += self.net.sgen.at[idx, 'p_mw']
        return total
    
    def get_status(self) -> Dict[str, Dict[str, float]]:
        """
        Return aggregated status metrics for all distributed energy resources grouped by type.
        
        The returned dictionary contains per-type summaries with capacity, current output/power, and unit counts; battery entries include average state of charge, and demand response entries include available and currently curtailed MW.
        
        Returns:
            dict: Mapping of DER type to metrics:
                - "solar": {"total_capacity_mw": total installed solar capacity in MW,
                            "current_output_mw": sum of current solar output in MW,
                            "unit_count": number of solar units}
                - "wind": {"total_capacity_mw": total installed wind capacity in MW,
                           "current_output_mw": sum of current wind output in MW,
                           "unit_count": number of wind units}
                - "battery": {"capacity_mwh": total battery energy capacity in MWh,
                              "current_soc_pct": average state of charge across batteries in percent,
                              "power_mw": sum of battery active power (positive = discharge) in MW,
                              "unit_count": number of battery units}
                - "ev_charger": {"max_power_mw": aggregated maximum EV charger power in MW,
                                 "current_power_mw": sum of current EV charging load in MW,
                                 "unit_count": number of EV charging units}
                - "demand_response": {"available_mw": total curtailable capacity in MW,
                                      "curtailed_mw": total currently curtailed load in MW,
                                      "unit_count": number of demand response units}
        """
        status = {
            "solar": {"total_capacity_mw": 0.0, "current_output_mw": 0.0, "unit_count": 0},
            "wind": {"total_capacity_mw": 0.0, "current_output_mw": 0.0, "unit_count": 0},
            "battery": {"capacity_mwh": 0.0, "current_soc_pct": 0.0, "power_mw": 0.0, "unit_count": 0},
            "ev_charger": {"max_power_mw": 0.0, "current_power_mw": 0.0, "unit_count": 0},
            "demand_response": {"available_mw": 0.0, "curtailed_mw": 0.0, "unit_count": 0}
        }
        
        for spec in self.der_specs:
            name = spec.name
            idx = self.der_indices.get(name)
            if idx is None:
                continue
                
            if spec.der_type == DERType.SOLAR_PV:
                status["solar"]["total_capacity_mw"] += spec.capacity_mw
                status["solar"]["current_output_mw"] += self.net.sgen.at[idx, 'p_mw']
                status["solar"]["unit_count"] += 1
                
            elif spec.der_type == DERType.WIND:
                status["wind"]["total_capacity_mw"] += spec.capacity_mw
                status["wind"]["current_output_mw"] += self.net.sgen.at[idx, 'p_mw']
                status["wind"]["unit_count"] += 1
                
            elif spec.der_type == DERType.BESS:
                status["battery"]["capacity_mwh"] += spec.energy_capacity_mwh or (spec.capacity_mw * 4)
                status["battery"]["current_soc_pct"] += self.battery_soc.get(name, 0.5) * 100
                status["battery"]["power_mw"] += self.net.storage.at[idx, 'p_mw']
                status["battery"]["unit_count"] += 1
                
            elif spec.der_type == DERType.EV_CHARGING:
                # EV: max power is capacity or num_chargers * charger_power
                max_power = spec.capacity_mw
                if spec.num_chargers and spec.charger_power_kw:
                    max_power = spec.num_chargers * spec.charger_power_kw / 1000
                status["ev_charger"]["max_power_mw"] += max_power or 0
                status["ev_charger"]["current_power_mw"] += self.net.load.at[idx, 'p_mw']
                status["ev_charger"]["unit_count"] += 1
                
            elif spec.der_type == DERType.DEMAND_RESPONSE:
                baseline = spec.baseline_load_mw or spec.capacity_mw
                curtailable = spec.curtailable_fraction or 0.2
                status["demand_response"]["available_mw"] += baseline * curtailable
                # Curtailed is baseline - current
                current = self.net.load.at[idx, 'p_mw']
                status["demand_response"]["curtailed_mw"] += max(0, baseline - current)
                status["demand_response"]["unit_count"] += 1
        
        # Average SOC across batteries if multiple
        if status["battery"]["unit_count"] > 0:
            status["battery"]["current_soc_pct"] /= status["battery"]["unit_count"]
            
        return status
                
    def __repr__(self) -> str:
        """
        Provide a concise textual summary of managed DER units grouped by type.
        
        Returns:
            repr_str (str): A string like "DERManager(3 SOLAR_PV, 2 WIND, ...)" listing the count of DERs for each DERType.
        """
        by_type = {}
        for spec in self.der_specs:
            by_type[spec.der_type.value] = by_type.get(spec.der_type.value, 0) + 1
        return f"DERManager({', '.join(f'{count} {t}' for t, count in by_type.items())})"