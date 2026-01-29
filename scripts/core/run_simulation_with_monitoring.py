#!/usr/bin/env python3
"""
Run IEEE 39-Bus Simulation with Prometheus Monitoring

Complete integration of IEEE 39-bus power system simulation with monitoring:
- Real-time metrics export to Prometheus
- Continuous simulation with DER coordination
- Power flow analysis monitoring
- Load control demonstration with metrics
- Integration with Grafana dashboards
"""

import sys
import time
import logging
from pathlib import Path
from prometheus_client import start_http_server
import signal

# Add paths
sys.path.insert(0, str(Path(__file__).parent))

from ieee39_system_strict import StrictIEEE39BusSystem
from monitoring_integration import IEEE39BusMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
shutdown_flag = False


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    global shutdown_flag
    logger.info("\n🛑 Shutdown signal received. Stopping simulation...")
    shutdown_flag = True


def run_continuous_simulation(
    monitor: IEEE39BusMonitor,
    system: StrictIEEE39BusSystem,
    num_steps: int = 1000,
    step_interval: float = 5.0
):
    """
    Run continuous simulation with monitoring
    
    Args:
        monitor: Prometheus monitor instance
        system: IEEE 39-bus system instance
        num_steps: Number of simulation steps
        step_interval: Time between steps (seconds)
    """
    logger.info("\n" + "=" * 80)
    logger.info("CONTINUOUS SIMULATION MODE")
    logger.info("=" * 80)
    logger.info(f"Steps: {num_steps}")
    logger.info(f"Interval: {step_interval}s")
    logger.info("Press Ctrl+C to stop gracefully\n")
    
    for step in range(1, num_steps + 1):
        if shutdown_flag:
            logger.info("Stopping simulation due to shutdown signal")
            break
        
        logger.info(f"--- Step {step}/{num_steps} ---")
        
        try:
            # Measure power flow duration
            start_time = time.time()
            
            # Run power flow analysis
            system.run_pypower_powerflow()
            
            power_flow_duration = time.time() - start_time
            monitor.record_power_flow_duration(power_flow_duration)
            
            # Get system state
            state = system.get_system_state()
            
            # Update monitoring metrics
            monitor.update_system_state(state)
            monitor.update_generator_metrics(system.ieee_generators)
            monitor.update_der_metrics(system.der_systems)
            monitor.increment_update_count()
            
            # Log summary
            logger.info(f"  Generation: {state['total_generation_mw']:.1f} MW")
            logger.info(f"  Load:       {state['total_load_mw']:.1f} MW")
            logger.info(f"  Losses:     {state['total_generation_mw'] - state['total_load_mw']:.1f} MW")
            logger.info(f"  Frequency:  {state['frequency_hz']:.3f} Hz")
            logger.info(f"  Voltage:    {state['voltage_min']:.4f} - {state['voltage_max']:.4f} pu")
            logger.info(f"  PF Time:    {power_flow_duration:.3f}s")
            
            # Wait for next step
            if step < num_steps and not shutdown_flag:
                time.sleep(step_interval)
        
        except Exception as e:
            logger.error(f"Error in step {step}: {e}")
            time.sleep(step_interval)
    
    logger.info(f"\n✓ Simulation completed {step} steps")


def run_load_control_demo(
    monitor: IEEE39BusMonitor,
    system: StrictIEEE39BusSystem
):
    """
    Run load control demonstration with monitoring
    
    Tests system response to load changes with DER coordination
    """
    logger.info("\n" + "=" * 80)
    logger.info("LOAD CONTROL DEMONSTRATION")
    logger.info("=" * 80)
    
    # Get baseline
    logger.info("\n📊 Establishing baseline...")
    system.run_pypower_powerflow()
    baseline_state = system.get_system_state()
    
    monitor.update_system_state(baseline_state)
    monitor.update_generator_metrics(system.ieee_generators)
    monitor.update_der_metrics(system.der_systems)
    
    logger.info(f"  ✓ Baseline Load: {baseline_state['total_load_mw']:.1f} MW")
    logger.info(f"  ✓ Baseline Generation: {baseline_state['total_generation_mw']:.1f} MW")
    
    # Apply load increase
    logger.info("\n🚀 Applying load increase...")
    load_increase_mw = 100.0
    
    # Distribute load increase
    load_distribution = {
        19: 25.0,  # Bus 20
        20: 20.0,  # Bus 21
        22: 20.0,  # Bus 23
        23: 20.0,  # Bus 24
        26: 15.0   # Bus 27
    }
    
    for bus_idx, additional_load in load_distribution.items():
        system.ieee39_case['bus'][bus_idx][2] += additional_load
        logger.info(f"  • Bus {bus_idx + 1}: +{additional_load} MW")
    
    # Simulate DER response
    logger.info("\n🔋 DER Systems Responding...")
    der_response_total = 0.0
    
    # Solar response (reduce curtailment)
    for name, der in system.der_systems.items():
        if 'Solar' in name:
            power = der.update_power(irradiance=950, temperature=25)
            additional = power * 0.1
            der_response_total += additional
            logger.info(f"  • {name}: +{additional:.1f} MW")
    
    # Wind response (optimize dispatch)
    for name, der in system.der_systems.items():
        if 'Wind' in name:
            power = der.update_power(wind_speed=13)
            additional = power * 0.05
            der_response_total += additional
            logger.info(f"  • {name}: +{additional:.1f} MW")
    
    # BESS discharge
    for name, der in system.der_systems.items():
        if 'BESS' in name:
            discharge = der.update_power(0.1, power_command=15.0)
            der_response_total += discharge
            logger.info(f"  • {name}: +{discharge:.1f} MW (SOC: {der.soc:.1%})")
    
    # Demand Response
    dr_reduction = 0.0
    for name, der in system.der_systems.items():
        if 'DR_' in name:
            reduced_load = der.update_load(dr_signal=0.5)
            reduction = der.baseline_load - reduced_load
            dr_reduction += reduction
            logger.info(f"  • {name}: -{reduction:.1f} MW")
    
    # EV V2G
    v2g_total = 0.0
    for name, der in system.der_systems.items():
        if 'EV_' in name:
            v2g = der.v2g_capable * 0.5 * der.charger_capacity_kw / 1000.0
            v2g_total += v2g
            logger.info(f"  • {name}: +{v2g:.1f} MW (V2G)")
    
    total_der_contribution = der_response_total + dr_reduction + v2g_total
    
    # Run power flow with new conditions
    logger.info("\n⚡ Running power flow with load increase...")
    system.run_pypower_powerflow()
    new_state = system.get_system_state()
    
    # Update monitoring
    monitor.update_system_state(new_state)
    monitor.update_generator_metrics(system.ieee_generators)
    monitor.update_der_metrics(system.der_systems)
    
    # Update load control metrics
    load_control_results = {
        'load_increase_applied': load_increase_mw,
        'der_contribution': total_der_contribution,
        'load_control_success': (
            new_state['voltage_min'] >= 0.9 and 
            new_state['voltage_max'] <= 1.1 and
            new_state['power_flow_converged']
        )
    }
    monitor.update_load_control_metrics(load_control_results)
    
    # Print results
    logger.info("\n📈 LOAD CONTROL RESULTS:")
    logger.info("=" * 60)
    logger.info(f"  • Load Increase Applied:    {load_increase_mw:.1f} MW")
    logger.info(f"  • DER Generation Response:  {der_response_total:.1f} MW")
    logger.info(f"  • Demand Response:          {dr_reduction:.1f} MW")
    logger.info(f"  • V2G Contribution:         {v2g_total:.1f} MW")
    logger.info(f"  • Total DER Contribution:   {total_der_contribution:.1f} MW")
    logger.info(f"  • DER Response Ratio:       {total_der_contribution/load_increase_mw:.1%}")
    logger.info(f"\n  • New Load:                 {new_state['total_load_mw']:.1f} MW")
    logger.info(f"  • New Generation:           {new_state['total_generation_mw']:.1f} MW")
    logger.info(f"  • Voltage Range:            {new_state['voltage_min']:.4f} - {new_state['voltage_max']:.4f} pu")
    logger.info(f"  • Frequency:                {new_state['frequency_hz']:.3f} Hz")
    logger.info(f"  • System Status:            {'✓ STABLE' if load_control_results['load_control_success'] else '❌ STRESSED'}")
    
    return load_control_results


def main(
    mode: str = 'demo',
    steps: int = 100,
    interval: float = 5.0,
    metrics_port: int = 9136
):
    """
    Main execution function
    
    Args:
        mode: 'demo' for load control demo, 'continuous' for continuous simulation
        steps: Number of steps for continuous mode
        interval: Time between steps for continuous mode
        metrics_port: Port for Prometheus metrics endpoint
    """
    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("=" * 80)
    logger.info("IEEE 39-BUS SYSTEM WITH PROMETHEUS MONITORING")
    logger.info("=" * 80)
    logger.info(f"Mode: {mode.upper()}")
    logger.info(f"Metrics Port: {metrics_port}")
    
    try:
        # Start Prometheus HTTP server
        logger.info(f"\n🌐 Starting Prometheus metrics server on port {metrics_port}...")
        start_http_server(metrics_port)
        logger.info(f"✓ Metrics available at http://localhost:{metrics_port}/metrics")
        
        # Initialize monitor
        logger.info("\n📊 Initializing monitoring...")
        monitor = IEEE39BusMonitor()
        logger.info("✓ Monitoring system initialized")
        
        # Initialize IEEE 39-bus system
        logger.info("\n🔌 Initializing IEEE 39-Bus System...")
        system = StrictIEEE39BusSystem()
        logger.info("✓ IEEE 39-Bus system initialized")
        
        # Setup DER systems
        logger.info("\n🔋 Setting up DER systems...")
        der_count = system.setup_ieee_compliant_ders()
        logger.info(f"✓ {der_count} DER systems configured")
        
        # Setup PMU network
        logger.info("\n📡 Setting up PMU network...")
        pmu_count = system.setup_ieee_pmu_network()
        logger.info(f"✓ {pmu_count} PMU locations configured")
        
        # Run initial power flow
        logger.info("\n⚡ Running initial power flow analysis...")
        system.run_strict_ieee39_analysis()
        state = system.get_system_state()
        
        # Update initial metrics
        monitor.update_system_state(state)
        monitor.update_generator_metrics(system.ieee_generators)
        monitor.update_der_metrics(system.der_systems)
        monitor.increment_update_count()
        
        logger.info("✓ Initial metrics exported")
        logger.info(f"\n📍 Access Points:")
        logger.info(f"  • Metrics:     http://localhost:{metrics_port}/metrics")
        logger.info(f"  • Prometheus:  http://localhost:9090")
        logger.info(f"  • Grafana:     http://localhost:3000 (admin/dems2024)")
        
        # Run selected mode
        if mode == 'demo':
            # Run load control demonstration
            results = run_load_control_demo(monitor, system)
            
            logger.info("\n🎯 Load control demonstration completed")
            logger.info("  Metrics have been exported to Prometheus")
            logger.info("  View real-time data in Grafana dashboard")
        
        elif mode == 'continuous':
            # Run continuous simulation
            run_continuous_simulation(monitor, system, steps, interval)
            
            logger.info("\n🎯 Continuous simulation completed")
            logger.info(f"  Total steps executed: {steps}")
            logger.info("  All metrics exported to Prometheus")
        
        else:
            logger.error(f"Unknown mode: {mode}")
            return False
        
        logger.info("\n✅ SIMULATION COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info("Metrics server will continue running...")
        logger.info("Press Ctrl+C to stop")
        
        # Keep server running
        try:
            while not shutdown_flag:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        
        logger.info("\n👋 Shutting down gracefully...")
        return True
    
    except Exception as e:
        logger.error(f"❌ Error in main execution: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run IEEE 39-Bus simulation with Prometheus monitoring"
    )
    parser.add_argument(
        '--mode',
        choices=['demo', 'continuous'],
        default='demo',
        help='Simulation mode (default: demo)'
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=100,
        help='Number of steps for continuous mode (default: 100)'
    )
    parser.add_argument(
        '--interval',
        type=float,
        default=5.0,
        help='Interval between steps in seconds (default: 5.0)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=9136,
        help='Prometheus metrics port (default: 9136)'
    )
    
    args = parser.parse_args()
    
    success = main(
        mode=args.mode,
        steps=args.steps,
        interval=args.interval,
        metrics_port=args.port
    )
    
    sys.exit(0 if success else 1)
