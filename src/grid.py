"""
DEMS Grid Module
High-level interface to the Tri-Area Super-Grid simulation

This module provides a simplified API for interacting with the 
117-bus Pandapower simulation. For direct access to the simulation
engine, use the src.simulation module.
"""

import logging
import time
from functools import wraps
from typing import Dict, Optional, List, Callable, Any
from src.simulation import SuperGrid, PowerFlowRunner, PowerFlowResult
from src.simulation.supergrid import AreaID, SuperGridConfig

logger = logging.getLogger(__name__)


# ======================== DECORATORS ======================== #

def log_operation(func: Callable) -> Callable:
    """Decorator to log grid operations with execution details"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        logger.info(f"Starting {func_name}")
        try:
            result = func(*args, **kwargs)
            logger.info(f"Completed {func_name} successfully")
            return result
        except Exception as e:
            logger.error(f"Error in {func_name}: {str(e)}")
            raise
    return wrapper


def measure_performance(func: Callable) -> Callable:
    """Decorator to measure and log execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start_time
        logger.debug(f"{func.__name__} took {elapsed*1000:.2f}ms")
        return result
    return wrapper


def validate_converged(func: Callable) -> Callable:
    """Decorator to ensure power flow has converged before state operations"""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self._last_result is None:
            logger.warning(f"{func.__name__} called before running power flow")
        elif not self._last_result.converged:
            logger.warning(f"{func.__name__} called with non-converged power flow")
        return func(self, *args, **kwargs)
    return wrapper


def safe_grid_operation(func: Callable) -> Callable:
    """Decorator for safe grid operations with automatic error recovery"""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            logger.error(f"Grid operation {func.__name__} failed: {str(e)}")
            logger.info("Attempting automatic recovery...")
            try:
                self.reset()
                logger.info("Grid reset successful, retrying operation")
                return func(self, *args, **kwargs)
            except Exception as recovery_error:
                logger.critical(f"Recovery failed: {str(recovery_error)}")
                raise
    return wrapper


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
        Create a DEMSGrid instance that composes the simulation SuperGrid and a PowerFlowRunner.
        
        Parameters:
            config (Optional[SuperGridConfig]): Optional configuration used to construct the underlying SuperGrid. If omitted, a default configuration is used.
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
            area_state (Dict): Dictionary containing area-specific state metrics.
        """
        area_id = AreaID(area)
        return self.supergrid.get_area_state(area_id)
        
    @safe_grid_operation
    @log_operation
    def set_area_generation(self, area: str, target_mw: float) -> None:
        """
        Set the total generation target for a specified area, distributing it across that area's controllable generators.
        
        Parameters:
            area (str): Area identifier, one of "A", "B", or "C".
            target_mw (float): Total generation target in megawatts to be dispatched across the area's generators.
        """
        area_id = AreaID(area)
        self.supergrid.set_area_generation(area_id, target_mw)
        
    @safe_grid_operation
    @log_operation
    def scale_area_load(self, area: str, scale_factor: float) -> None:
        """
        Scale all loads in the specified area by a multiplicative factor.
        
        Parameters:
            area (str): Area identifier, one of "A", "B", or "C".
            scale_factor (float): Multiplicative factor to apply to each load in the area (1.0 = no change).
        """
        area_id = AreaID(area)
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