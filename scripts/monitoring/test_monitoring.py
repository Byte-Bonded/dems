#!/usr/bin/env python3
"""
Standalone DEMS Monitoring Test
Tests the Prometheus exporter without requiring full DEMS imports
"""

import time
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Try to import prometheus_client
try:
    from prometheus_client import start_http_server, Gauge, Counter
    PROMETHEUS_AVAILABLE = True
except ImportError:
    logger.error("prometheus-client not installed. Install with: pip install prometheus-client")
    PROMETHEUS_AVAILABLE = False
    sys.exit(1)


class SimpleMetricsExporter:
    """Simplified metrics exporter for testing"""
    
    def __init__(self, port=9136):
        self.port = port
        
        # Create some test metrics
        self.generation_mw = Gauge('dems_total_generation_mw', 'Total generation in MW')
        self.load_mw = Gauge('dems_total_load_mw', 'Total load in MW')
        self.losses_mw = Gauge('dems_total_losses_mw', 'Total losses in MW')
        self.voltage_pu = Gauge('dems_avg_voltage_pu', 'Average voltage in per-unit')
        self.frequency_hz = Gauge('dems_system_frequency_hz', 'System frequency in Hz')
        self.grid_secure = Gauge('dems_grid_secure', 'Grid security status')
        self.step_counter = Counter('dems_simulation_step_total', 'Total simulation steps')
        
        logger.info(f"Metrics exporter initialized on port {port}")
    
    def start(self):
        """Start the HTTP server"""
        start_http_server(self.port)
        logger.info(f"✓ Prometheus metrics server started on http://localhost:{self.port}/metrics")
    
    def update_metrics(self, step):
        """Update metrics with simulated values"""
        import random
        
        # Simulate realistic grid values
        base_gen = 2500.0
        base_load = 2450.0
        
        generation = base_gen + random.uniform(-100, 100)
        load = base_load + random.uniform(-50, 50)
        losses = generation - load
        
        self.generation_mw.set(generation)
        self.load_mw.set(load)
        self.losses_mw.set(losses)
        self.voltage_pu.set(random.uniform(0.98, 1.02))
        self.frequency_hz.set(random.uniform(49.95, 50.05))
        self.grid_secure.set(1 if losses < 100 else 0)
        self.step_counter.inc()
        
        return generation, load, losses


def main():
    """Main test function"""
    logger.info("=" * 80)
    logger.info("DEMS Monitoring Test - Standalone Mode")
    logger.info("=" * 80)
    logger.info("")
    
    # Create and start exporter
    exporter = SimpleMetricsExporter(port=9136)
    exporter.start()
    
    logger.info("")
    logger.info("Test running. Access metrics at:")
    logger.info("  • Metrics: http://localhost:9136/metrics")
    logger.info("  • Prometheus: http://localhost:9090 (if running)")
    logger.info("  • Grafana: http://localhost:3000 (if running)")
    logger.info("")
    logger.info("Press Ctrl+C to stop")
    logger.info("")
    
    try:
        step = 0
        while True:
            step += 1
            generation, load, losses = exporter.update_metrics(step)
            
            logger.info(
                f"Step {step:3d} | "
                f"Gen: {generation:7.1f} MW | "
                f"Load: {load:7.1f} MW | "
                f"Loss: {losses:5.1f} MW ({losses/generation*100:.2f}%)"
            )
            
            time.sleep(2)  # Update every 2 seconds
            
    except KeyboardInterrupt:
        logger.info("")
        logger.info("⚠ Test stopped by user")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    finally:
        logger.info("Keeping metrics endpoint alive for 10 seconds...")
        time.sleep(10)
        logger.info("✓ Test complete")


if __name__ == "__main__":
    main()
