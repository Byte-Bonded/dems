"""
DEMS Grid Module
High-level interface to the Tri-Area Super-Grid simulation

This module provides a simplified API for interacting with the 
117-bus Pandapower simulation. For direct access to the simulation
engine, use the src.simulation module.
"""

import logging
from typing import Dict, Optional, List, Any
from src.simulation import SuperGrid, PowerFlowRunner, PowerFlowResult
from src.simulation.supergrid import AreaID, SuperGridConfig

# Import decorators from utils (also available locally for backward compat)
from src.utils.decorators import (
    log_operation,
    measure_performance,
    validate_converged,
    safe_grid_operation,
)

logger = logging.getLogger(__name__)


class DEMSGrid:
    """
    High-level wrapper for the DEMS Tri-Area Super-Grid
    
    Provides simplified methods for:
    - Running simulations
    - Getting grid state
    - Applying control actions
    
    Example:
        >>> grid = DEMSGrid()
        >>> result = grid.run_power_flow()
        >>> if result.converged:
        ...     state = grid.get_state()
        ...     print(f"Total load: {state['global_metrics']['total_load_mw']} MW")
    """
    
    def __init__(self, config: Optional[SuperGridConfig] = None):
        """
        Initialize the DEMS Grid
        
        Args:
            config: Optional configuration for the super-grid
        """
        self.supergrid = SuperGrid(config)
        self.power_flow_runner = PowerFlowRunner()
        self._last_result: Optional[PowerFlowResult] = None
        logger.info("DEMSGrid initialized successfully")
        
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
            self.supergrid.net, 
            verbose=verbose
        )
        return self._last_result
        
    @validate_converged
    def get_state(self) -> Dict:
        """
        Get the complete grid state (after running power flow)
        
        Returns:
            Dictionary with area states, tie-line flows, and global metrics
        """
        return self.supergrid.get_global_state()
        
    @validate_converged
    def get_area_state(self, area: str) -> Dict:
        """
        Get state for a specific area
        
        Args:
            area: Area identifier ("A", "B", or "C")
            
        Returns:
            Dictionary with area metrics
            
        Raises:
            ValueError: If area is invalid
            RuntimeError: If power flow has not converged
        """
        try:
            area_id = AreaID(area)
        except ValueError:
            raise ValueError(f"Invalid area '{area}'. Must be 'A', 'B', or 'C'")
        return self.supergrid.get_area_state(area_id)
        
    @safe_grid_operation
    @log_operation
    def set_area_generation(self, area: str, target_mw: float) -> None:
        """
        Set target generation for an area (dispatched across generators)
        
        Args:
            area: Area identifier ("A", "B", or "C")
            target_mw: Total generation target in MW
            
        Raises:
            ValueError: If area is invalid or target_mw is negative
        """
        # Validate inputs
        try:
            area_id = AreaID(area)
        except ValueError:
            raise ValueError(f"Invalid area '{area}'. Must be 'A', 'B', or 'C'")
        
        if target_mw < 0:
            raise ValueError(f"Generation target must be non-negative, got {target_mw}")
        
        if target_mw > 5000:
            logger.warning(f"Very high generation target for {area}: {target_mw} MW")
        
        self.supergrid.set_area_generation(area_id, target_mw)
        
    @safe_grid_operation
    @log_operation
    def scale_area_load(self, area: str, scale_factor: float) -> None:
        """
        Scale all loads in an area
        
        Args:
            area: Area identifier ("A", "B", or "C")
            scale_factor: Multiplication factor (1.0 = no change)
            
        Raises:
            ValueError: If area is invalid or scale_factor is out of range
        """
        # Validate inputs
        try:
            area_id = AreaID(area)
        except ValueError:
            raise ValueError(f"Invalid area '{area}'. Must be 'A', 'B', or 'C'")
        
        if scale_factor < 0:
            raise ValueError(f"Scale factor must be non-negative, got {scale_factor}")
        
        if scale_factor > 3.0:
            logger.warning(f"Very high scale factor for {area}: {scale_factor}")
        
        self.supergrid.scale_loads(area_id, scale_factor)
        
    @validate_converged
    def get_tie_line_flows(self) -> List[Dict]:
        """
        Get current power flows on tie-lines
        
        Returns:
            List of tie-line flow dictionaries
        """
        return self.supergrid.get_tie_line_flows()
        
    def get_generators(self) -> Dict[str, List[Dict]]:
        """
        Get information about all controllable generators
        
        Returns:
            Dictionary mapping area IDs to generator lists
        """
        return self.supergrid.get_controllable_generators()
        
    @log_operation
    def reset(self) -> None:
        """Reset grid to base case"""
        self.supergrid.reset_to_base_case()
        self._last_result = None
        
    @property
    def is_converged(self) -> bool:
        """Check if last power flow converged"""
        return self._last_result.converged if self._last_result else False
        
    @property
    def is_secure(self) -> bool:
        """Check if system is in secure operating state"""
        return self._last_result.is_secure if self._last_result else False
        
    def __repr__(self) -> str:
        return f"DEMSGrid({self.supergrid})"


# Convenience function
def create_grid(config: Optional[SuperGridConfig] = None) -> DEMSGrid:
    """
    Create and initialize a DEMS Grid instance
    
    Args:
        config: Optional grid configuration
        
    Returns:
        Initialized DEMSGrid
    """
    return DEMSGrid(config)
