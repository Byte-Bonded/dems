"""
Tie-Line Configuration and Management
Handles inter-area connections in the Tri-Area Super-Grid

Tie-lines are the critical links between areas that:
1. Allow power transfer between regions
2. Create thermal bottlenecks that must be managed
3. Determine the degree of inter-area coupling
"""

from dataclasses import dataclass
from typing import List
from enum import Enum
import numpy as np
import pandapower as pp


class TieLineType(Enum):
    """Types of inter-area connections"""
    HVAC_345KV = "hvac_345kv"  # High-voltage AC (typical US)
    HVAC_500KV = "hvac_500kv"  # Extra-high voltage AC
    HVDC = "hvdc"              # High-voltage DC (for async areas)


@dataclass
class TieLineConfig:
    """Configuration for a single tie-line"""
    name: str
    from_area: str  # Area ID (A, B, or C)
    to_area: str    # Area ID
    from_bus_local: int  # Bus number within from_area (0-38)
    to_bus_local: int    # Bus number within to_area (0-38)
    
    # Electrical parameters
    line_type: TieLineType = TieLineType.HVAC_345KV
    length_km: float = 150.0
    rating_mva: float = 600.0  # Thermal limit
    
    # Impedance parameters (per km)
    r_ohm_per_km: float = 0.02   # Resistance
    x_ohm_per_km: float = 0.25   # Reactance
    c_nf_per_km: float = 12.0    # Capacitance
    
    # Operating limits
    emergency_rating_mva: float = 720.0  # Short-term overload limit
    
    def get_max_current_ka(self, voltage_kv: float = 345.0) -> float:
        """
        Return the maximum continuous current for the tie-line at a specified line-to-line voltage.
        
        Parameters:
            voltage_kv (float): Line-to-line voltage in kilovolts used to compute the current (default 345.0).
        
        Returns:
            max_current_ka (float): Maximum continuous current in kiloamperes corresponding to the tie-line's MVA rating at the given voltage.
        """
        return self.rating_mva / (voltage_kv * np.sqrt(3))


# Default tie-line configuration for the Tri-Area Super-Grid
DEFAULT_TIE_LINES: List[TieLineConfig] = [
    # Area A <-> Area B connections
    TieLineConfig(
        name="TL_A-B_1",
        from_area="A",
        to_area="B",
        from_bus_local=1,
        to_bus_local=1,
        length_km=120.0,
        rating_mva=600.0,
    ),
    TieLineConfig(
        name="TL_A-B_2",
        from_area="A",
        to_area="B", 
        from_bus_local=26,
        to_bus_local=26,
        length_km=180.0,
        rating_mva=500.0,
    ),
    
    # Area B <-> Area C connections
    TieLineConfig(
        name="TL_B-C_1",
        from_area="B",
        to_area="C",
        from_bus_local=3,
        to_bus_local=3,
        length_km=100.0,
        rating_mva=550.0,
    ),
    TieLineConfig(
        name="TL_B-C_2",
        from_area="B",
        to_area="C",
        from_bus_local=15,
        to_bus_local=15,
        length_km=140.0,
        rating_mva=600.0,
    ),
    
    # Area A <-> Area C connections (direct link)
    TieLineConfig(
        name="TL_A-C_1",
        from_area="A",
        to_area="C",
        from_bus_local=16,
        to_bus_local=16,
        length_km=200.0,
        rating_mva=450.0,
    ),
    TieLineConfig(
        name="TL_A-C_2",
        from_area="A",
        to_area="C",
        from_bus_local=21,
        to_bus_local=21,
        length_km=250.0,
        rating_mva=400.0,
    ),
]


def create_tie_lines(
    net: pp.pandapowerNet,
    tie_line_configs: List[TieLineConfig],
    area_offsets: dict,
    voltage_kv: float = 345.0,
) -> List[int]:
    """
    Add tie-lines to a pandapower network from the provided TieLineConfig entries.
    
    Parameters:
        net (pp.pandapowerNet): The pandapower network to modify.
        tie_line_configs (List[TieLineConfig]): Configurations describing each tie-line to create.
        area_offsets (dict): Mapping from area ID to bus index offset (e.g., {"A": 0, "B": 39, "C": 78}) used to convert local bus indices to global bus indices.
        voltage_kv (float): Nominal line-to-line voltage in kV used to compute each line's maximum current (default 345.0).
    
    Returns:
        List[int]: List of created pandapower line indices.
    """
    created_indices = []
    
    for config in tie_line_configs:
        # Calculate global bus indices
        from_bus = area_offsets[config.from_area] + config.from_bus_local
        to_bus = area_offsets[config.to_area] + config.to_bus_local
        
        # Create the line
        line_idx = pp.create_line_from_parameters(
            net,
            from_bus=from_bus,
            to_bus=to_bus,
            length_km=config.length_km,
            r_ohm_per_km=config.r_ohm_per_km,
            x_ohm_per_km=config.x_ohm_per_km,
            c_nf_per_km=config.c_nf_per_km,
            max_i_ka=config.get_max_current_ka(voltage_kv),
            name=config.name,
            type="ol",  # Overhead line
        )
        
        created_indices.append(line_idx)
        
    return created_indices


def get_tie_line_transfer_limits(
    configs: List[TieLineConfig],
) -> dict:
    """
    Aggregate transfer limits by inter-area pair from a list of tie-line configurations.
    
    Area pair keys are canonicalized by alphabetical order (e.g., "A-B"). Each pair maps to a dictionary with summed thermal ratings in both directions.
    
    Parameters:
        configs (List[TieLineConfig]): Tie-line configurations to aggregate.
    
    Returns:
        dict: Mapping from area-pair string to a dict with keys:
            - "forward_mva" (float): Sum of rating_mva for lines in that area pair.
            - "reverse_mva" (float): Same as forward_mva for AC tie-lines.
    """
    limits = {}
    
    # Group by area pairs
    for config in configs:
        # Create canonical key (alphabetically ordered)
        areas = sorted([config.from_area, config.to_area])
        key = f"{areas[0]}-{areas[1]}"
        
        if key not in limits:
            limits[key] = {"forward_mva": 0.0, "reverse_mva": 0.0}
            
        # Add this line's capacity (same in both directions for AC)
        limits[key]["forward_mva"] += config.rating_mva
        limits[key]["reverse_mva"] += config.rating_mva
        
    return limits


@dataclass
class TieLineStatus:
    """Real-time status of a tie-line"""
    name: str
    p_flow_mw: float      # Active power flow (positive = from -> to)
    q_flow_mvar: float    # Reactive power flow
    loading_percent: float
    is_overloaded: bool
    from_bus: int
    to_bus: int
    
    @property
    def is_critical(self) -> bool:
        """
        Indicates whether the tie-line's loading exceeds 80% of its thermal rating.
        
        Returns:
            bool: `true` if `loading_percent` is greater than 80.0, `false` otherwise.
        """
        return self.loading_percent > 80.0
    
    @property
    def margin_mw(self) -> float:
        """
        Estimate the remaining transferable capacity of the tie-line in megawatts.
        
        Returns:
            float: Remaining transfer margin in MW calculated from current active flow and loading percentage; returns 0.0 if loading_percent is greater than or equal to 100 or less than or equal to 0.01.
        """
        # Simplified calculation assuming constant power factor
        if self.loading_percent >= 100:
            return 0.0
        # Guard against zero or near-zero loading to avoid ZeroDivisionError
        if self.loading_percent <= 0.01:
            return 0.0  # No meaningful margin calculation when line is essentially unloaded
        return abs(self.p_flow_mw) * (100 - self.loading_percent) / self.loading_percent


def analyze_transfer_capability(
    net: pp.pandapowerNet,
    tie_line_indices: List[int],
) -> dict:
    """
    Evaluate current transfer capability of specified tie-lines using power-flow results.
    
    Parameters:
        net (pp.pandapowerNet): Pandapower network containing completed power-flow results; `net.res_line` must be populated.
        tie_line_indices (List[int]): Indices of tie-lines in `net.line` to include in the analysis.
    
    Returns:
        dict: Analysis summary with the following keys:
            - individual_status (List[TieLineStatus]): Per-line status objects for each tie-line.
            - total_transfer_mw (float): Sum of the absolute active power flows (MW) on the provided tie-lines.
            - max_loading_percent (float): Maximum loading percent among the provided tie-lines.
            - num_overloaded (int): Count of tie-lines with loading percent greater than 100.
            - num_critical (int): Count of tie-lines considered critical (loading percent > 80).
            - system_secure (bool): True if no tie-lines are overloaded, False otherwise.
    
    Raises:
        RuntimeError: If power-flow results are not available (i.e., `net.res_line` is empty).
    """
    if net.res_line.empty:
        raise RuntimeError("Power flow results not available")
        
    statuses = []
    
    for idx in tie_line_indices:
        line_data = net.line.loc[idx]
        line_result = net.res_line.loc[idx]
        
        status = TieLineStatus(
            name=line_data["name"],
            p_flow_mw=line_result.p_from_mw,
            q_flow_mvar=line_result.q_from_mvar,
            loading_percent=line_result.loading_percent,
            is_overloaded=line_result.loading_percent > 100.0,
            from_bus=int(line_data["from_bus"]),
            to_bus=int(line_data["to_bus"]),
        )
        statuses.append(status)
        
    # Aggregate analysis
    total_transfer = sum(abs(s.p_flow_mw) for s in statuses)
    # Handle empty statuses to avoid ValueError from max() on empty sequence
    max_loading = max((s.loading_percent for s in statuses), default=0.0)
    num_overloaded = sum(1 for s in statuses if s.is_overloaded)
    num_critical = sum(1 for s in statuses if s.is_critical)
    
    return {
        "individual_status": statuses,
        "total_transfer_mw": total_transfer,
        "max_loading_percent": max_loading,
        "num_overloaded": num_overloaded,
        "num_critical": num_critical,
        "system_secure": num_overloaded == 0,
    }