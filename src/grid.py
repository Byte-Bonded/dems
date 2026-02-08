"""
DEMS Grid Module
High-level interface to the Kundur Two-Area System

Provides a simplified API for:
- Running power flow simulations
- Getting system state
- Applying control actions
- Managing DER (Solar, Wind, Battery, EV)
"""

import logging
from typing import Dict, Optional, List, Any
from src.simulation import KundurTwoAreaSystem, KundurConfig, AreaID, PowerFlowRunner, PowerFlowResult
from src.utils.decorators import (
    log_operation,
    measure_performance,
    safe_grid_operation,
)

logger = logging.getLogger(__name__)


class DEMSGrid:
    """
    High-level wrapper for the Kundur Two-Area System
    
    Simplified interface for power system control and RL training.
    Optimized for PSS testing and inter-area oscillation studies.
    
    Example:
        >>> grid = DEMSGrid()
        >>> result = grid.run_power_flow()
        >>> if result.converged:
        ...     state = grid.get_state()
        ...     print(f"Tie-line flow: {state['system']['tie_line_flow_mw']} MW")
    """
    
    def __init__(self, config: Optional[KundurConfig] = None, enable_der: bool = True):
        """
        Create a DEMSGrid instance with Kundur Two-Area System
        
        Args:
            config: System configuration (uses defaults if None)
            enable_der: Enable DER (Solar, Wind, Battery, EV) integration
        """
        self.kundur = KundurTwoAreaSystem(config, enable_der=enable_der)
        self.power_flow_runner = PowerFlowRunner()
        self._last_result: Optional[PowerFlowResult] = None
        logger.info("DEMSGrid initialized with Kundur Two-Area System")
        
    @log_operation
    @measure_performance
    def run_power_flow(self, verbose: bool = False) -> PowerFlowResult:
        """
        Run AC power flow on the grid
        
        Args:
            verbose: Print detailed results
            
        Returns:
            PowerFlowResult with convergence status and metrics
        """
        self._last_result = self.power_flow_runner.run(
            self.kundur.net, 
            verbose=verbose
        )
        return self._last_result
        
    def get_state(self) -> Dict:
        """
        Get complete system state
        
        Returns:
            Dict with generators, loads, tie-lines, areas, and DER states
        """
        return self.kundur.get_state()
        
    def get_area_state(self, area: AreaID) -> Dict:
        """
        Get state for a specific area
        
        Args:
            area: AreaID.AREA_1 or AreaID.AREA_2
            
        Returns:
            Dict with area generation, load, and voltage
        """
        metrics = self.kundur.get_area_metrics()
        return metrics.get(area.value, {})
        
    def get_tie_line_flows(self) -> Dict[str, float]:
        """
        Get power flow on tie-lines (critical for oscillation monitoring)
        
        Returns:
            Dict with tie-line flows in MW and MVAR
        """
        return self.kundur.get_tie_line_flow()
        
    @safe_grid_operation
    @log_operation
    def set_generator_power(self, gen_name: str, p_mw: float) -> bool:
        """
        Set generator active power setpoint
        
        Args:
            gen_name: Generator name (G1, G2, G3, G4)
            p_mw: Power setpoint in MW
            
        Returns:
            True if successful, False otherwise
        """
        return self.kundur.set_generator_setpoint(gen_name, p_mw)
        
    @safe_grid_operation
    @log_operation
    def apply_load_perturbation(self, area: AreaID, delta_mw: float) -> None:
        """
        Apply load step change (for transient stability testing)
        
        Args:
            area: Target area (AREA_1 or AREA_2)
            delta_mw: Load change in MW (positive = increase)
        """
        self.kundur.apply_load_perturbation(area, delta_mw)
        
    def update_der_conditions(
        self,
        solar_irradiance: float = 800.0,
        wind_speed: float = 12.0,
        temperature: float = 25.0
    ) -> None:
        """
        Update DER output based on environmental conditions
        
        Args:
            solar_irradiance: Solar irradiance (0-1000 W/m²)
            wind_speed: Wind speed (0-25 m/s)
            temperature: Ambient temperature (°C)
        """
        self.kundur.update_der_conditions(solar_irradiance, wind_speed, temperature)
        
    def dispatch_battery(self, battery_name: str, power_mw: float) -> bool:
        """
        Dispatch battery storage
        
        Args:
            battery_name: Battery name (BESS_A1, BESS_A2)
            power_mw: Power setpoint (positive=discharge, negative=charge)
            
        Returns:
            True if successful
        """
        return self.kundur.dispatch_battery(battery_name, power_mw)
        
    def get_der_state(self) -> List[Dict]:
        """Get current state of all DER units"""
        return self.kundur.get_der_state()
        
    @log_operation
    def reset(self) -> None:
        """Reset grid to initial state"""
        self.kundur = KundurTwoAreaSystem(
            self.kundur.config, 
            enable_der=self.kundur.enable_der
        )
        self._last_result = None
        
    @property
    def is_converged(self) -> bool:
        """Check if last power flow converged"""
        return self._last_result.converged if self._last_result else False
        
    @property
    def num_buses(self) -> int:
        """Number of buses in the system"""
        return len(self.kundur.net.bus)
        
    @property
    def num_generators(self) -> int:
        """Number of generators in the system"""
        return len(self.kundur.net.gen)
        
    def __repr__(self) -> str:
        return f"DEMSGrid({self.kundur})"


def create_grid(config: Optional[KundurConfig] = None, enable_der: bool = True) -> DEMSGrid:
    """
    Create a DEMSGrid instance
    
    Args:
        config: Kundur system configuration
        enable_der: Enable DER integration
        
    Returns:
        Initialized DEMSGrid
    """
    return DEMSGrid(config, enable_der)
