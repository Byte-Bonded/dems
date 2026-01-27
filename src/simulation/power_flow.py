"""
Power Flow Execution Engine
Runs AC/DC power flow calculations on the Super-Grid

Power flow (load flow) analysis determines:
1. Bus voltages (magnitude and angle)
2. Line power flows
3. Generator outputs
4. System losses
"""

import pandapower as pp
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import logging
import time

logger = logging.getLogger(__name__)


class PowerFlowAlgorithm(Enum):
    """Available power flow algorithms"""
    NEWTON_RAPHSON = "nr"       # Newton-Raphson (most common, fast convergence)
    FAST_DECOUPLED = "fdxb"     # Fast Decoupled XB (faster for large systems)
    GAUSS_SEIDEL = "gs"         # Gauss-Seidel (simple, slow convergence)
    DC = "dc"                   # DC approximation (linear, very fast)


@dataclass
class PowerFlowConfig:
    """Configuration for power flow calculation"""
    algorithm: PowerFlowAlgorithm = PowerFlowAlgorithm.NEWTON_RAPHSON
    max_iterations: int = 30
    tolerance_mva: float = 1e-8
    enforce_q_limits: bool = True
    calculate_voltage_angles: bool = True
    init_method: str = "auto"  # "auto", "flat", "dc", "results"
    
    # Convergence options
    check_connectivity: bool = True
    voltage_depend_loads: bool = False
    
    # DC power flow specific
    trafo_model: str = "t"  # "t" or "pi" for transformer model


@dataclass
class PowerFlowResult:
    """Results from a power flow calculation"""
    converged: bool
    iterations: int
    elapsed_time_ms: float
    
    # System-wide results
    total_generation_mw: float = 0.0
    total_generation_mvar: float = 0.0
    total_load_mw: float = 0.0
    total_load_mvar: float = 0.0
    total_losses_mw: float = 0.0
    total_losses_mvar: float = 0.0
    
    # Constraint violations
    num_voltage_violations: int = 0
    num_line_overloads: int = 0
    num_trafo_overloads: int = 0
    
    # Voltage statistics
    min_voltage_pu: float = 0.0
    max_voltage_pu: float = 0.0
    avg_voltage_pu: float = 0.0
    
    # Error info (if not converged)
    error_message: Optional[str] = None
    
    # Detailed results (optional)
    bus_results: Optional[Dict] = None
    line_results: Optional[Dict] = None
    gen_results: Optional[Dict] = None
    
    @property
    def is_secure(self) -> bool:
        """Check if system is in secure operating state"""
        return (self.converged and 
                self.num_voltage_violations == 0 and 
                self.num_line_overloads == 0)


class PowerFlowRunner:
    """
    Executes power flow calculations on pandapower networks
    
    Features:
    - Multiple algorithm support (NR, FDXB, GS, DC)
    - Automatic convergence handling
    - Result extraction and analysis
    - Constraint violation detection
    """
    
    def __init__(self, config: Optional[PowerFlowConfig] = None):
        """
        Initialize the power flow runner
        
        Args:
            config: Power flow configuration
        """
        self.config = config or PowerFlowConfig()
        self.last_result: Optional[PowerFlowResult] = None
        
    def run(
        self,
        net: pp.pandapowerNet,
        algorithm: Optional[PowerFlowAlgorithm] = None,
        verbose: bool = False,
    ) -> PowerFlowResult:
        """
        Execute power flow calculation
        
        Args:
            net: Pandapower network
            algorithm: Override default algorithm
            verbose: Print detailed output
            
        Returns:
            PowerFlowResult with convergence status and system metrics
        """
        algo = algorithm or self.config.algorithm
        
        start_time = time.time()
        converged = False
        iterations = 0
        error_msg = None
        
        try:
            if algo == PowerFlowAlgorithm.DC:
                # DC power flow (linear approximation)
                pp.rundcpp(net)
                converged = True
                iterations = 1
            else:
                # AC power flow
                pp.runpp(
                    net,
                    algorithm=algo.value,
                    max_iteration=self.config.max_iterations,
                    tolerance_mva=self.config.tolerance_mva,
                    enforce_q_lims=self.config.enforce_q_limits,
                    calculate_voltage_angles=self.config.calculate_voltage_angles,
                    init=self.config.init_method,
                    check_connectivity=self.config.check_connectivity,
                    voltage_depend_loads=self.config.voltage_depend_loads,
                )
                converged = net.converged
                iterations = net.get("_ppc", {}).get("iterations", 0)
                
        except pp.powerflow.LoadflowNotConverged as e:
            converged = False
            error_msg = f"Power flow did not converge: {str(e)}"
            logger.warning(error_msg)
            
        except Exception as e:
            converged = False
            error_msg = f"Power flow error: {str(e)}"
            logger.error(error_msg)
            
        elapsed_time = (time.time() - start_time) * 1000  # ms
        
        # Extract results if converged
        result = self._extract_results(
            net, converged, iterations, elapsed_time, error_msg
        )
        
        if verbose:
            self._print_summary(result)
            
        self.last_result = result
        return result
    
    def run_with_contingency(
        self,
        net: pp.pandapowerNet,
        contingency_elements: List[tuple],
    ) -> List[PowerFlowResult]:
        """
        Run power flow for N-1 contingency analysis
        
        Args:
            net: Pandapower network
            contingency_elements: List of (element_type, index) tuples
                                  e.g., [("line", 5), ("gen", 2)]
                                  
        Returns:
            List of PowerFlowResult for each contingency
        """
        results = []
        
        # Base case
        base_result = self.run(net)
        base_result.error_message = "Base case"
        results.append(base_result)
        
        # Contingency cases
        for elem_type, elem_idx in contingency_elements:
            # Store original state
            original_state = net[elem_type].at[elem_idx, "in_service"]
            
            # Apply contingency (take element out of service)
            net[elem_type].at[elem_idx, "in_service"] = False
            
            # Run power flow
            cont_result = self.run(net)
            cont_result.error_message = f"Contingency: {elem_type}[{elem_idx}] out"
            results.append(cont_result)
            
            # Restore original state
            net[elem_type].at[elem_idx, "in_service"] = original_state
            
        return results
    
    def _extract_results(
        self,
        net: pp.pandapowerNet,
        converged: bool,
        iterations: int,
        elapsed_time: float,
        error_msg: Optional[str],
    ) -> PowerFlowResult:
        """Extract comprehensive results from completed power flow"""
        
        result = PowerFlowResult(
            converged=converged,
            iterations=iterations,
            elapsed_time_ms=elapsed_time,
            error_message=error_msg,
        )
        
        if not converged or net.res_bus.empty:
            return result
            
        # Generation totals
        if not net.res_gen.empty:
            result.total_generation_mw = float(net.res_gen.p_mw.sum())
            result.total_generation_mvar = float(net.res_gen.q_mvar.sum())
            
        # Add external grid contribution
        if not net.res_ext_grid.empty:
            result.total_generation_mw += float(net.res_ext_grid.p_mw.sum())
            result.total_generation_mvar += float(net.res_ext_grid.q_mvar.sum())
            
        # Load totals
        if not net.res_load.empty:
            result.total_load_mw = float(net.res_load.p_mw.sum())
            result.total_load_mvar = float(net.res_load.q_mvar.sum())
            
        # Losses (generation - load)
        result.total_losses_mw = result.total_generation_mw - result.total_load_mw
        result.total_losses_mvar = result.total_generation_mvar - result.total_load_mvar
        
        # Voltage statistics
        result.min_voltage_pu = float(net.res_bus.vm_pu.min())
        result.max_voltage_pu = float(net.res_bus.vm_pu.max())
        result.avg_voltage_pu = float(net.res_bus.vm_pu.mean())
        
        # Constraint violations
        result.num_voltage_violations = int(
            ((net.res_bus.vm_pu < 0.95) | (net.res_bus.vm_pu > 1.05)).sum()
        )
        
        if not net.res_line.empty:
            result.num_line_overloads = int((net.res_line.loading_percent > 100).sum())
            
        if not net.res_trafo.empty:
            result.num_trafo_overloads = int((net.res_trafo.loading_percent > 100).sum())
            
        return result
    
    def _print_summary(self, result: PowerFlowResult) -> None:
        """Print a summary of power flow results"""
        status = "✓ Converged" if result.converged else "✗ Not Converged"
        
        print(f"\n{'='*50}")
        print(f"Power Flow Results: {status}")
        print(f"{'='*50}")
        print(f"Iterations: {result.iterations}")
        print(f"Time: {result.elapsed_time_ms:.2f} ms")
        print(f"\nGeneration: {result.total_generation_mw:.2f} MW / {result.total_generation_mvar:.2f} MVAr")
        print(f"Load:       {result.total_load_mw:.2f} MW / {result.total_load_mvar:.2f} MVAr")
        print(f"Losses:     {result.total_losses_mw:.2f} MW / {result.total_losses_mvar:.2f} MVAr")
        print(f"\nVoltage: {result.min_voltage_pu:.4f} - {result.max_voltage_pu:.4f} pu")
        print(f"Violations: {result.num_voltage_violations} voltage, "
              f"{result.num_line_overloads} line overloads")
        print(f"{'='*50}\n")
        
    def get_bus_results_dataframe(self, net: pp.pandapowerNet):
        """Get bus results as a pandas DataFrame"""
        return net.res_bus.copy() if not net.res_bus.empty else None
        
    def get_line_results_dataframe(self, net: pp.pandapowerNet):
        """Get line results as a pandas DataFrame"""
        return net.res_line.copy() if not net.res_line.empty else None
        
    def get_gen_results_dataframe(self, net: pp.pandapowerNet):
        """Get generator results as a pandas DataFrame"""
        return net.res_gen.copy() if not net.res_gen.empty else None


def run_time_series_power_flow(
    net: pp.pandapowerNet,
    load_profiles: np.ndarray,
    gen_profiles: np.ndarray,
    timesteps: int,
    runner: Optional[PowerFlowRunner] = None,
) -> List[PowerFlowResult]:
    """
    Run time-series power flow simulation
    
    Args:
        net: Pandapower network
        load_profiles: Array of shape (timesteps, num_loads) with load multipliers
        gen_profiles: Array of shape (timesteps, num_gens) with generation setpoints
        timesteps: Number of timesteps to simulate
        runner: PowerFlowRunner instance (creates default if None)
        
    Returns:
        List of PowerFlowResult for each timestep
    """
    if runner is None:
        runner = PowerFlowRunner()
        
    results = []
    
    # Store original values
    original_loads = net.load.p_mw.copy()
    original_gens = net.gen.p_mw.copy()
    
    for t in range(timesteps):
        # Apply load profile
        if load_profiles is not None:
            net.load.p_mw = original_loads * load_profiles[t]
            
        # Apply generation profile
        if gen_profiles is not None:
            for i, gen_idx in enumerate(net.gen.index):
                if i < gen_profiles.shape[1]:
                    net.gen.at[gen_idx, 'p_mw'] = gen_profiles[t, i]
                    
        # Run power flow
        result = runner.run(net)
        results.append(result)
        
    # Restore original values
    net.load.p_mw = original_loads
    net.gen.p_mw = original_gens
    
    return results


def quick_power_flow(net: pp.pandapowerNet) -> bool:
    """
    Quick power flow check - just returns convergence status
    
    Args:
        net: Pandapower network
        
    Returns:
        True if converged, False otherwise
    """
    try:
        pp.runpp(net, max_iteration=30)
        return net.converged
    except Exception:
        return False
