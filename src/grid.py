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
        """
        area_id = AreaID(area)
        return self.supergrid.get_area_state(area_id)
        
    @safe_grid_operation
    @log_operation
    def set_area_generation(self, area: str, target_mw: float) -> None:
        """
        Set target generation for an area (dispatched across generators)
        
        Args:
            area: Area identifier ("A", "B", or "C")
            target_mw: Total generation target in MW
        """
        area_id = AreaID(area)
        self.supergrid.set_area_generation(area_id, target_mw)
        
    @safe_grid_operation
    @log_operation
    def scale_area_load(self, area: str, scale_factor: float) -> None:
        """
        Scale all loads in an area
        
        Args:
            area: Area identifier ("A", "B", or "C")
            scale_factor: Multiplication factor (1.0 = no change)
        """
        area_id = AreaID(area)
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
