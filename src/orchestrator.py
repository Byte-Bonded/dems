"""
DEMS Grid Orchestrator
Coordinates all grid components and provides comprehensive logging

This orchestrator manages:
- Grid initialization and power flow
- DER (Distributed Energy Resources) operations
- Dynamic simulations
- State monitoring and logging
- Time-series simulations
"""

import logging
import sys
from datetime import datetime
from typing import Dict, List, Optional, Any
import json

from src.grid import DEMSGrid
from src.simulation.supergrid import SuperGridConfig
from src.simulation.power_flow import PowerFlowResult

# Configure logging
def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Setup comprehensive logging configuration for DEMS modules only.
    
    This function configures logging for the 'src' namespace without
    polluting the root logger, preventing log spam from third-party libraries.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for logging output
        
    Returns:
        Configured logger instance for the orchestrator
    """
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configure DEMS logger (not root logger)
    dems_logger = logging.getLogger('src')
    dems_logger.setLevel(getattr(logging, log_level.upper()))
    dems_logger.propagate = False  # Don't propagate to root
    
    # Clear any existing handlers to avoid duplicates
    dems_logger.handlers.clear()
    
    # Console handler for DEMS logs
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    dems_logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.DEBUG)  # Log everything to file
        file_handler.setFormatter(formatter)
        dems_logger.addHandler(file_handler)
    
    # Silence noisy third-party loggers
    logging.getLogger('pandapower').setLevel(logging.WARNING)
    logging.getLogger('numba').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
        
    return logging.getLogger(__name__)


class GridOrchestrator:
    """
    Main orchestrator for DEMS grid operations
    
    Manages the complete lifecycle of grid simulation including:
    - Initialization
    - Power flow execution
    - DER integration
    - State monitoring
    - Time-series simulation
    """
    
    def __init__(
        self,
        config: Optional[SuperGridConfig] = None,
        log_level: str = "INFO",
        log_file: Optional[str] = None
    ):
        """
        Initialize the Grid Orchestrator
        
        Args:
            config: SuperGrid configuration
            log_level: Logging level
            log_file: Optional log file path
        """
        # Setup logging
        self.logger = setup_logging(log_level, log_file)
        self.logger.info("=" * 80)
        self.logger.info("DEMS Grid Orchestrator Starting")
        self.logger.info("=" * 80)
        
        # Initialize grid
        self.logger.info("Initializing DEMS Grid...")
        self.grid = DEMSGrid(config)
        self.logger.info("[OK] Grid initialized successfully")
        
        # Initialize DER systems
        self.logger.info("Initializing Distributed Energy Resources...")
        self.grid.supergrid.initialize_der(add_default=True)
        self.logger.info("[OK] DER systems initialized")
        
        # Track simulation state
        self.simulation_step = 0
        self.history: List[Dict[str, Any]] = []
        self.start_time = datetime.now()
        
    def log_power_flow_result(self, result: PowerFlowResult, prefix: str = "") -> None:
        """
        Log detailed power flow results
        
        Args:
            result: PowerFlowResult to log
            prefix: Optional prefix for log messages
        """
        status = "[OK] CONVERGED" if result.converged else "[X] NOT CONVERGED"
        security = "SECURE" if result.is_secure else "INSECURE"
        
        self.logger.info(f"{prefix}Power Flow Status: {status} | {security}")
        self.logger.info(f"{prefix}Iterations: {result.iterations} | Time: {result.elapsed_time_ms:.2f} ms")
        self.logger.info(f"{prefix}Generation: {result.total_generation_mw:.2f} MW | {result.total_generation_mvar:.2f} MVAr")
        self.logger.info(f"{prefix}Load:       {result.total_load_mw:.2f} MW | {result.total_load_mvar:.2f} MVAr")
        self.logger.info(f"{prefix}Losses:     {result.total_losses_mw:.2f} MW ({result.total_losses_mw/result.total_generation_mw*100:.2f}%)")
        self.logger.info(f"{prefix}Voltage:    {result.min_voltage_pu:.4f} - {result.max_voltage_pu:.4f} pu (avg: {result.avg_voltage_pu:.4f})")
        
        # Log violations
        if result.num_voltage_violations > 0:
            self.logger.warning(f"{prefix}[!] Voltage violations: {result.num_voltage_violations} buses")
        if result.num_line_overloads > 0:
            self.logger.warning(f"{prefix}[!] Line overloads: {result.num_line_overloads} lines")
        if result.num_trafo_overloads > 0:
            self.logger.warning(f"{prefix}[!] Transformer overloads: {result.num_trafo_overloads} transformers")
            
        if result.error_message:
            self.logger.error(f"{prefix}Error: {result.error_message}")
    
    def log_grid_state(self, state: Dict, prefix: str = "") -> None:
        """
        Log comprehensive grid state
        
        Args:
            state: Grid state dictionary
            prefix: Optional prefix for log messages
        """
        self.logger.info(f"{prefix}{'=' * 60}")
        self.logger.info(f"{prefix}GRID STATE SUMMARY")
        self.logger.info(f"{prefix}{'=' * 60}")
        
        # Global metrics
        if 'global_metrics' in state:
            gm = state['global_metrics']
            self.logger.info(f"{prefix}Total Generation: {gm.get('total_generation_mw', 0):.2f} MW")
            self.logger.info(f"{prefix}Total Load:       {gm.get('total_load_mw', 0):.2f} MW")
            self.logger.info(f"{prefix}Total Losses:     {gm.get('total_losses_mw', 0):.2f} MW")
            self.logger.info(f"{prefix}Avg Frequency:    {gm.get('avg_frequency_hz', 50.0):.4f} Hz")
            
        # Area states
        for area in ['area_A', 'area_B', 'area_C']:
            if area in state:
                area_data = state[area]
                area_name = area.split('_')[1]
                self.logger.info(f"{prefix}--- Area {area_name} ---")
                self.logger.info(f"{prefix}  Generation: {area_data.get('total_generation_mw', 0):.2f} MW")
                self.logger.info(f"{prefix}  Load:       {area_data.get('total_load_mw', 0):.2f} MW")
                self.logger.info(f"{prefix}  Generators: {area_data.get('num_generators', 0)}")
                self.logger.info(f"{prefix}  Buses:      {area_data.get('num_buses', 0)}")
                
        # Tie-line flows
        if 'tie_lines' in state:
            self.logger.info(f"{prefix}--- Tie-Line Flows ---")
            for tie_line in state['tie_lines']:
                name = tie_line.get('name', 'Unknown')
                from_bus = tie_line.get('from_bus', '?')
                to_bus = tie_line.get('to_bus', '?')
                p_mw = tie_line.get('p_from_mw', 0)
                loading = tie_line.get('loading_percent', 0)
                overloaded = " [OVERLOADED]" if tie_line.get('is_overloaded', False) else ""
                self.logger.info(
                    f"{prefix}  {name}: Bus {from_bus} -> Bus {to_bus}: "
                    f"{p_mw:.2f} MW | Loading: {loading:.1f}%{overloaded}"
                )
    
    def run_single_power_flow(self, verbose: bool = True) -> PowerFlowResult:
        """
        Run a single power flow analysis
        
        Args:
            verbose: Whether to log detailed results
            
        Returns:
            PowerFlowResult
        """
        self.logger.info("Running power flow analysis...")
        result = self.grid.run_power_flow(verbose=False)
        
        if verbose:
            self.log_power_flow_result(result)
            
        return result
    
    def get_and_log_state(self) -> Dict:
        """
        Get current grid state and log it
        
        Returns:
            Grid state dictionary
        """
        self.logger.info("Retrieving grid state...")
        state = self.grid.get_state()
        self.log_grid_state(state)
        return state
    
    def run_area_analysis(self) -> None:
        """Run detailed analysis for each area"""
        self.logger.info("=" * 80)
        self.logger.info("AREA-BY-AREA ANALYSIS")
        self.logger.info("=" * 80)
        
        for area in ['A', 'B', 'C']:
            self.logger.info(f"\n--- Area {area} Details ---")
            area_state = self.grid.get_area_state(area)
            
            self.logger.info(f"Buses:      {area_state.get('num_buses', 0)}")
            self.logger.info(f"Generators: {area_state.get('num_generators', 0)}")
            self.logger.info(f"Generation: {area_state.get('total_generation_mw', 0):.2f} MW")
            self.logger.info(f"Load:       {area_state.get('total_load_mw', 0):.2f} MW")
            self.logger.info(f"Net Export: {area_state.get('net_export_mw', 0):.2f} MW")
    
    def log_der_summary(self) -> None:
        """Log DER (Distributed Energy Resources) summary"""
        self.logger.info("=" * 80)
        self.logger.info("DER SUMMARY")
        self.logger.info("=" * 80)
        
        if self.grid.supergrid.der_manager is None:
            self.logger.warning("DER Manager not initialized")
            return
        
        der_status = self.grid.supergrid.der_manager.get_status()
        
        solar = der_status.get("solar", {})
        wind = der_status.get("wind", {})
        battery = der_status.get("battery", {})
        ev = der_status.get("ev_charger", {})
        dr = der_status.get("demand_response", {})
        
        self.logger.info(f"Solar PV:     {solar.get('unit_count', 0)} units, {solar.get('total_capacity_mw', 0):.2f} MW capacity, {solar.get('current_output_mw', 0):.2f} MW output")
        self.logger.info(f"Wind:         {wind.get('unit_count', 0)} units, {wind.get('total_capacity_mw', 0):.2f} MW capacity, {wind.get('current_output_mw', 0):.2f} MW output")
        self.logger.info(f"Battery:      {battery.get('unit_count', 0)} units, {battery.get('capacity_mwh', 0):.2f} MWh, {battery.get('current_soc_pct', 0):.1f}% SOC")
        self.logger.info(f"EV Charging:  {ev.get('unit_count', 0)} units, {ev.get('max_power_mw', 0):.2f} MW max, {ev.get('current_power_mw', 0):.2f} MW current")
        self.logger.info(f"Demand Resp:  {dr.get('unit_count', 0)} units, {dr.get('available_mw', 0):.2f} MW available, {dr.get('curtailed_mw', 0):.2f} MW curtailed")
        
        total_capacity = (
            solar.get('total_capacity_mw', 0) +
            wind.get('total_capacity_mw', 0) +
            battery.get('power_mw', 0) +
            ev.get('max_power_mw', 0) +
            dr.get('available_mw', 0)
        )
        self.logger.info(f"Total DER:    {total_capacity:.2f} MW controllable capacity")
    
    def run_time_series_simulation(
        self,
        num_steps: int = 24,
        step_duration: str = "1 hour"
    ) -> List[Dict[str, Any]]:
        """
        Run time-series simulation
        
        Args:
            num_steps: Number of time steps
            step_duration: Description of step duration
            
        Returns:
            List of state snapshots
        """
        self.logger.info("=" * 80)
        self.logger.info(f"TIME-SERIES SIMULATION: {num_steps} steps ({step_duration} each)")
        self.logger.info("=" * 80)
        
        snapshots = []
        
        for step in range(num_steps):
            self.simulation_step += 1
            self.logger.info(f"\n--- Step {step + 1}/{num_steps} ---")
            
            # Run power flow
            result = self.run_single_power_flow(verbose=False)
            
            # Get state
            self.grid.get_state()
            
            # Create snapshot
            snapshot = {
                'step': step + 1,
                'timestamp': datetime.now().isoformat(),
                'converged': result.converged,
                'generation_mw': result.total_generation_mw,
                'load_mw': result.total_load_mw,
                'losses_mw': result.total_losses_mw,
                'min_voltage_pu': result.min_voltage_pu,
                'max_voltage_pu': result.max_voltage_pu,
            }
            snapshots.append(snapshot)
            self.history.append(snapshot)
            
            # Log summary
            self.logger.info(
                f"Gen: {result.total_generation_mw:.1f} MW | "
                f"Load: {result.total_load_mw:.1f} MW | "
                f"Voltage: {result.min_voltage_pu:.3f}-{result.max_voltage_pu:.3f} pu"
            )
            
            if not result.converged:
                self.logger.error(f"Power flow did not converge at step {step + 1}")
        
        self.logger.info("\n✓ Time-series simulation completed")
        return snapshots
    
    def run_contingency_analysis(self, contingencies: List[tuple]) -> None:
        """
        Run N-1 contingency analysis
        
        Args:
            contingencies: List of (element_type, index) tuples
        """
        self.logger.info("=" * 80)
        self.logger.info("CONTINGENCY ANALYSIS (N-1)")
        self.logger.info("=" * 80)
        
        results = self.grid.power_flow_runner.run_with_contingency(
            self.grid.supergrid.net,
            contingencies
        )
        
        for i, result in enumerate(results):
            prefix = "  "
            if i == 0:
                self.logger.info("\n[BASE CASE]")
            else:
                self.logger.info(f"\n[CONTINGENCY {i}]: {result.error_message}")
            
            self.log_power_flow_result(result, prefix)
    
    def save_history(self, filepath: str = "grid_history.json") -> None:
        """
        Save simulation history to JSON file
        
        Args:
            filepath: Output file path
        """
        self.logger.info(f"Saving simulation history to {filepath}...")
        
        output_data = {
            'metadata': {
                'start_time': self.start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'total_steps': len(self.history),
            },
            'history': self.history
        }
        
        with open(filepath, 'w') as f:
            json.dump(output_data, f, indent=2)
            
        self.logger.info(f"✓ History saved ({len(self.history)} steps)")
    
    def print_summary(self) -> None:
        """Print final summary"""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("SIMULATION SUMMARY")
        self.logger.info("=" * 80)
        self.logger.info(f"Total simulation time: {elapsed:.2f} seconds")
        self.logger.info(f"Total steps executed:  {len(self.history)}")
        self.logger.info(f"Grid buses:            {len(self.grid.supergrid.net.bus)}")
        self.logger.info(f"Generators:            {len(self.grid.supergrid.net.gen)}")
        self.logger.info(f"Lines:                 {len(self.grid.supergrid.net.line)}")
        self.logger.info("=" * 80)


def main():
    """Main orchestration function - demonstrates full grid operation"""
    
    # Create orchestrator with file logging
    log_file = f"dems_grid_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    orchestrator = GridOrchestrator(
        log_level="INFO",
        log_file=log_file
    )
    
    try:
        # 1. Run initial power flow
        orchestrator.logger.info("\n" + "=" * 80)
        orchestrator.logger.info("STEP 1: INITIAL POWER FLOW")
        orchestrator.logger.info("=" * 80)
        result = orchestrator.run_single_power_flow(verbose=True)
        
        if not result.converged:
            orchestrator.logger.error("Initial power flow did not converge!")
            return
        
        # 2. Get and log grid state
        orchestrator.logger.info("\n" + "=" * 80)
        orchestrator.logger.info("STEP 2: GRID STATE ANALYSIS")
        orchestrator.logger.info("=" * 80)
        orchestrator.get_and_log_state()
        
        # 3. Area-by-area analysis
        orchestrator.logger.info("\n" + "=" * 80)
        orchestrator.logger.info("STEP 3: AREA ANALYSIS")
        orchestrator.logger.info("=" * 80)
        orchestrator.run_area_analysis()
        
        # 4. DER summary
        orchestrator.logger.info("\n" + "=" * 80)
        orchestrator.logger.info("STEP 4: DER RESOURCES")
        orchestrator.logger.info("=" * 80)
        orchestrator.log_der_summary()
        
        # 5. Time-series simulation (short demo)
        orchestrator.logger.info("\n" + "=" * 80)
        orchestrator.logger.info("STEP 5: TIME-SERIES SIMULATION")
        orchestrator.logger.info("=" * 80)
        orchestrator.run_time_series_simulation(num_steps=5, step_duration="1 hour")
        
        # 6. Save history
        orchestrator.save_history()
        
        # 7. Print summary
        orchestrator.print_summary()
        
        orchestrator.logger.info("\n✓ Grid orchestration completed successfully")
        orchestrator.logger.info(f"✓ Full logs saved to: {log_file}")
        
    except Exception as e:
        orchestrator.logger.error(f"Orchestration failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
