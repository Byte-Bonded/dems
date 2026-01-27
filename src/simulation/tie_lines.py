"""
Tie-Line Configuration and Management
Handles inter-area connections in the Tri-Area Super-Grid

Tie-lines are the critical links between areas that:
1. Allow power transfer between regions
2. Create thermal bottlenecks that must be managed
3. Determine the degree of inter-area coupling
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
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
        """Calculate maximum current based on MVA rating and voltage"""
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
    Create tie-lines in a pandapower network
    
    Args:
        net: The pandapower network to add tie-lines to
        tie_line_configs: List of tie-line configurations
        area_offsets: Dictionary mapping area IDs to bus offsets {"A": 0, "B": 39, "C": 78}
        voltage_kv: Nominal voltage for current calculation
        
    Returns:
        List of created line indices
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
    Calculate aggregate transfer limits between area pairs
    
    Args:
        configs: List of tie-line configurations
        
    Returns:
        Dictionary with transfer limits:
        {
            "A-B": {"forward_mva": 1100, "reverse_mva": 1100},
            "B-C": {"forward_mva": 1150, "reverse_mva": 1150},
            "A-C": {"forward_mva": 850, "reverse_mva": 850},
        }
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
        """Check if line is approaching thermal limit (>80%)"""
        return self.loading_percent > 80.0
    
    @property
    def margin_mw(self) -> float:
        """Approximate remaining transfer margin in MW"""
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
    Analyze the current transfer capability of tie-lines
    
    Args:
        net: Pandapower network with completed power flow results
        tie_line_indices: Indices of tie-lines in the network
        
    Returns:
        Analysis results including:
        - Individual line status
        - Aggregate interface flows
        - System-wide transfer margins
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
