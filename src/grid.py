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

import warnings

logger = logging.getLogger(__name__)


class DEMSGrid:
    """
    High-level wrapper for the DEMS Tri-Area Super-Grid.
    
    .. deprecated::
        FIX BUG-19: This is a thin redundant wrapper over SuperGrid.
        Prefer using SuperGrid directly for new code. This wrapper
        is maintained for backward compatibility only.
    """
    
    def __init__(self, config: Optional[SuperGridConfig] = None):
        warnings.warn(
            "DEMSGrid is deprecated. Use SuperGrid directly.",
            DeprecationWarning, stacklevel=2
        )
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
        Return the global grid state.
        
        Includes per-area states, tie-line flows, and aggregate global metrics; reflects the current network model (run power flow to update results).
        
        Returns:
            Dict: Mapping containing area states, tie-line flow information, and global metrics.
        """
        return self.supergrid.get_global_state()
        
    @validate_converged
    def get_area_state(self, area: str) -> Dict:
        """
        Retrieve the state metrics for the specified area.
        
        Parameters:
            area (str): Area identifier — one of "A", "B", or "C".
        
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
        Set the total generation target for a specified area, distributing it across that area's controllable generators.
        
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
        Scale all loads in the specified area by a multiplicative factor.
        
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
        Retrieve the current power flows across tie-lines connecting different grid areas.
        
        Returns:
            List of dictionaries representing tie-line power flow data.
        """
        return self.supergrid.get_tie_line_flows()
        
    def get_generators(self) -> Dict[str, List[Dict]]:
        """
        Return information about controllable generators grouped by area.
        
        Returns:
            dict: Mapping from area identifier (e.g., "A", "B", "C") to a list of generator information dictionaries for each controllable generator in that area.
        """
        return self.supergrid.get_controllable_generators()
        
    @log_operation
    def reset(self) -> None:
        """
        Reset the underlying grid to its stored base case.
        
        Restores the SuperGrid to its base-case configuration and clears the cached last power flow result on this DEMSGrid instance.
        """
        self.supergrid.reset_to_base_case()
        self._last_result = None
        
    @property
    def is_converged(self) -> bool:
        """
        Indicates whether the most recent power flow run converged.
        
        Returns:
            True if the last power flow result exists and is converged, False otherwise.
        """
        return self._last_result.converged if self._last_result else False
        
    @property
    def is_secure(self) -> bool:
        """
        Report whether the most recent power flow result indicates the system is in a secure operating state.
        
        Returns:
            `true` if the last power flow result is secure, `false` otherwise.
        """
        return self._last_result.is_secure if self._last_result else False
        
    def __repr__(self) -> str:
        """
        Return a concise string representation of the DEMSGrid that references its underlying SuperGrid.
        
        Returns:
            repr_str (str): A string in the form "DEMSGrid(<SuperGrid_repr>)" where <SuperGrid_repr> is the underlying SuperGrid's representation.
        """
        return f"DEMSGrid({self.supergrid})"


# Convenience function
def create_grid(config: Optional[SuperGridConfig] = None) -> DEMSGrid:
    """
    Create a DEMSGrid wrapper configured with an optional SuperGridConfig.
    
    Parameters:
        config (Optional[SuperGridConfig]): Optional configuration used to construct the underlying SuperGrid; when omitted, a default configuration is used.
    
    Returns:
        DEMSGrid: An initialized DEMSGrid instance ready for running power flows and interacting with the tri-area super-grid.
    """
    return DEMSGrid(config)