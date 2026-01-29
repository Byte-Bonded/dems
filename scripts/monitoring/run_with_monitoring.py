#!/usr/bin/env python3
"""
Run DEMS Simulation with Prometheus Monitoring
Integrates GridOrchestrator with Prometheus metrics export
"""

import sys
import time
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.orchestrator import GridOrchestrator
from scripts.monitoring.prometheus_exporter import DEMSPrometheusExporter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def run_monitored_simulation(
    num_steps: int = 100,
    step_interval_seconds: float = 2.0,
    exporter_port: int = 9136
):
    """
    Run DEMS simulation with Prometheus monitoring
    
    Args:
        num_steps: Number of simulation steps
        step_interval_seconds: Time between steps (for pacing)
        exporter_port: Port for Prometheus exporter
    """
    
    logger.info("=" * 80)
    logger.info("DEMS Grid Simulation with Prometheus Monitoring")
    logger.info("=" * 80)
    
    # Initialize Prometheus exporter
    logger.info(f"Starting Prometheus exporter on port {exporter_port}...")
    exporter = DEMSPrometheusExporter(port=exporter_port)
    exporter.start()
    logger.info(f"✓ Metrics available at http://localhost:{exporter_port}/metrics")
    
    # Initialize Grid Orchestrator
    logger.info("\nInitializing DEMS Grid Orchestrator...")
    orchestrator = GridOrchestrator(log_level="INFO")
    logger.info("✓ Grid orchestrator ready")
    
    try:
        # Run initial power flow and get state
        logger.info("\n" + "=" * 80)
        logger.info("Initial Grid State")
        logger.info("=" * 80)
        
        result = orchestrator.run_single_power_flow(verbose=True)
        state = orchestrator.get_and_log_state()
        
        # Update metrics
        exporter.update_from_power_flow_result(result)
        exporter.update_from_grid_state(state)
        
        # Update DER metrics if available
        if orchestrator.grid.supergrid.der_manager:
            der_status = orchestrator.grid.supergrid.der_manager.get_status()
            exporter.update_from_der_status(der_status)
            orchestrator.log_der_summary()
        
        logger.info("\n✓ Initial metrics exported to Prometheus")
    logger.info(f"  View metrics: http://localhost:{exporter_port}/metrics")
        logger.info(f"  View in Prometheus: http://localhost:9090")
        
        # Run time-series simulation with monitoring
        logger.info("\n" + "=" * 80)
        logger.info(f"Starting Time-Series Simulation ({num_steps} steps)")
        logger.info("=" * 80)
        logger.info(f"Metrics update interval: {step_interval_seconds}s")
        logger.info("Press Ctrl+C to stop\n")
        
        for step in range(num_steps):
            step_num = step + 1
            logger.info(f"--- Step {step_num}/{num_steps} ---")
            
            # Run power flow
            result = orchestrator.run_single_power_flow(verbose=False)
            
            # Get state
            state = orchestrator.grid.get_state()
            
            # Update all metrics
            exporter.update_from_power_flow_result(result)
            exporter.update_from_grid_state(state)
            
            # Update DER metrics periodically
            if orchestrator.grid.supergrid.der_manager and step_num % 5 == 0:
                der_status = orchestrator.grid.supergrid.der_manager.get_status()
                exporter.update_from_der_status(der_status)
            
            # Log summary
            logger.info(
                f"  Gen: {result.total_generation_mw:.1f} MW | "
                f"Load: {result.total_load_mw:.1f} MW | "
                f"Loss: {result.total_losses_mw:.2f} MW | "
                f"Voltage: {result.min_voltage_pu:.3f}-{result.max_voltage_pu:.3f} pu | "
                f"Secure: {'YES' if result.is_secure else 'NO'}"
            )
            
            # Check for issues
            if not result.converged:
                logger.warning("  ⚠ Power flow did not converge!")
                exporter.simulation_errors.labels(error_type='convergence').inc()
            
            if result.num_voltage_violations > 0:
                logger.warning(f"  ⚠ {result.num_voltage_violations} voltage violations")
            
            if result.num_line_overloads > 0:
                logger.warning(f"  ⚠ {result.num_line_overloads} line overloads")
            
            # Wait before next step
            time.sleep(step_interval_seconds)
        
        logger.info("\n" + "=" * 80)
        logger.info("✓ Simulation Completed Successfully")
        logger.info("=" * 80)
        orchestrator.print_summary()
        
    except KeyboardInterrupt:
        logger.info("\n\n⚠ Simulation interrupted by user")
        logger.info("Shutting down gracefully...")
        
    except Exception as e:
        logger.error(f"\n✗ Simulation failed: {e}", exc_info=True)
        exporter.simulation_errors.labels(error_type='simulation_error').inc()
        raise
    
    finally:
        # Keep exporter running for a bit to allow final scrape
        logger.info("\nKeeping metrics endpoint alive for 10 seconds...")
        logger.info("(Press Ctrl+C to exit immediately)")
        try:
            time.sleep(10)
        except KeyboardInterrupt:
            pass
        
        logger.info("Shutting down exporter...")
        exporter.shutdown()
        logger.info("✓ Cleanup complete")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run DEMS Grid Simulation with Prometheus Monitoring"
    )
    parser.add_argument(
        '--steps', 
        type=int, 
        default=100,
        help='Number of simulation steps (default: 100)'
    )
    parser.add_argument(
        '--interval', 
        type=float, 
        default=2.0,
        help='Seconds between steps (default: 2.0)'
    )
    parser.add_argument(
        '--port', 
        type=int, 
        default=9136,
        help='Prometheus exporter port (default: 9136)'
    )
    
    args = parser.parse_args()
    
    run_monitored_simulation(
        num_steps=args.steps,
        step_interval_seconds=args.interval,
        exporter_port=args.port
    )
