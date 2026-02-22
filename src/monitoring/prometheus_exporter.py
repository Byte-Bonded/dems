"""
Prometheus Exporter for DEMS Grid Simulation
Exposes comprehensive metrics from the 117-bus SuperGrid simulation

This exporter provides:
- Real-time power flow metrics (generation, load, losses)
- Voltage and frequency monitoring across all areas
- DER (Distributed Energy Resources) status
- Tie-line flow monitoring
- Grid stability indicators
- Performance metrics
"""

import logging
import time
from typing import Optional, Dict, Any
from datetime import datetime
from prometheus_client import (
    start_http_server, 
    Gauge, 
    Counter, 
    Histogram, 
    Info,
    CollectorRegistry,
    REGISTRY
)
import threading

logger = logging.getLogger(__name__)


class DEMSPrometheusExporter:
    """
    Prometheus exporter for DEMS Grid metrics
    
    Collects metrics from GridOrchestrator and exposes them on HTTP endpoint
    for Prometheus scraping.
    
    Usage:
        exporter = DEMSPrometheusExporter(port=9136)
        exporter.start()
        
        # Update metrics from orchestrator
        exporter.update_from_power_flow_result(result)
        exporter.update_from_grid_state(state)
    """
    
    def __init__(self, port: int = 9136, registry: Optional[CollectorRegistry] = None):
        """
        Initialize Prometheus exporter
        
        Args:
            port: HTTP port for metrics endpoint
            registry: Optional custom registry (uses default if None)
        """
        self.port = port
        self.registry = registry or REGISTRY
        self._server_thread = None
        self._lock = threading.Lock()
        
        # ==================== SYSTEM INFO ==================== #
        self.system_info = Info(
            'dems_system',
            'DEMS Grid System Information',
            registry=self.registry
        )
        self.system_info.info({
            'version': '1.0.0',
            'grid_type': '117-bus-supergrid',
            'base_frequency': '50Hz',
            'areas': '3'
        })
        
        # ==================== POWER FLOW METRICS ==================== #
        
        # Generation
        self.total_generation_mw = Gauge(
            'dems_total_generation_mw',
            'Total generation in MW',
            registry=self.registry
        )
        self.total_generation_mvar = Gauge(
            'dems_total_generation_mvar',
            'Total reactive generation in MVAr',
            registry=self.registry
        )
        
        # Load
        self.total_load_mw = Gauge(
            'dems_total_load_mw',
            'Total load in MW',
            registry=self.registry
        )
        self.total_load_mvar = Gauge(
            'dems_total_load_mvar',
            'Total reactive load in MVAr',
            registry=self.registry
        )
        
        # Losses
        self.total_losses_mw = Gauge(
            'dems_total_losses_mw',
            'Total system losses in MW',
            registry=self.registry
        )
        self.total_losses_mvar = Gauge(
            'dems_total_losses_mvar',
            'Total reactive losses in MVAr',
            registry=self.registry
        )
        self.loss_percentage = Gauge(
            'dems_loss_percentage',
            'Percentage of generation lost as losses',
            registry=self.registry
        )
        
        # ==================== VOLTAGE METRICS ==================== #
        
        self.min_voltage_pu = Gauge(
            'dems_min_voltage_pu',
            'Minimum bus voltage in per-unit',
            registry=self.registry
        )
        self.max_voltage_pu = Gauge(
            'dems_max_voltage_pu',
            'Maximum bus voltage in per-unit',
            registry=self.registry
        )
        self.avg_voltage_pu = Gauge(
            'dems_avg_voltage_pu',
            'Average bus voltage in per-unit',
            registry=self.registry
        )
        self.voltage_violations = Gauge(
            'dems_voltage_violations',
            'Number of buses with voltage violations',
            registry=self.registry
        )
        
        # ==================== FREQUENCY METRICS ==================== #
        
        self.system_frequency_hz = Gauge(
            'dems_system_frequency_hz',
            'System frequency in Hz',
            registry=self.registry
        )
        self.avg_frequency_hz = Gauge(
            'dems_avg_frequency_hz',
            'Average area frequency in Hz',
            registry=self.registry
        )
        
        # ==================== LOADING METRICS ==================== #
        
        self.line_overloads = Gauge(
            'dems_line_overloads',
            'Number of transmission lines overloaded',
            registry=self.registry
        )
        self.trafo_overloads = Gauge(
            'dems_trafo_overloads',
            'Number of transformers overloaded',
            registry=self.registry
        )
        self.max_line_loading_pct = Gauge(
            'dems_max_line_loading_percent',
            'Maximum line loading percentage',
            registry=self.registry
        )
        
        # ==================== AREA-SPECIFIC METRICS ==================== #
        
        self.area_generation_mw = Gauge(
            'dems_area_generation_mw',
            'Generation by area in MW',
            ['area'],
            registry=self.registry
        )
        self.area_load_mw = Gauge(
            'dems_area_load_mw',
            'Load by area in MW',
            ['area'],
            registry=self.registry
        )
        self.area_net_export_mw = Gauge(
            'dems_area_net_export_mw',
            'Net power export by area in MW (negative = import)',
            ['area'],
            registry=self.registry
        )
        
        # ==================== TIE-LINE METRICS ==================== #
        
        self.tie_line_flow_mw = Gauge(
            'dems_tie_line_flow_mw',
            'Power flow on tie-lines in MW',
            ['tie_line', 'from_area', 'to_area'],
            registry=self.registry
        )
        self.tie_line_loading_pct = Gauge(
            'dems_tie_line_loading_percent',
            'Tie-line loading percentage',
            ['tie_line'],
            registry=self.registry
        )
        
        # ==================== DER METRICS ==================== #
        
        # Solar
        self.solar_capacity_mw = Gauge(
            'dems_solar_capacity_mw',
            'Total solar PV capacity in MW',
            registry=self.registry
        )
        self.solar_output_mw = Gauge(
            'dems_solar_output_mw',
            'Current solar PV output in MW',
            registry=self.registry
        )
        self.solar_unit_count = Gauge(
            'dems_solar_unit_count',
            'Number of solar PV units',
            registry=self.registry
        )
        
        # Wind
        self.wind_capacity_mw = Gauge(
            'dems_wind_capacity_mw',
            'Total wind capacity in MW',
            registry=self.registry
        )
        self.wind_output_mw = Gauge(
            'dems_wind_output_mw',
            'Current wind output in MW',
            registry=self.registry
        )
        self.wind_unit_count = Gauge(
            'dems_wind_unit_count',
            'Number of wind turbines',
            registry=self.registry
        )
        
        # Battery Storage
        self.battery_capacity_mwh = Gauge(
            'dems_battery_capacity_mwh',
            'Total battery capacity in MWh',
            registry=self.registry
        )
        self.battery_soc_pct = Gauge(
            'dems_battery_soc_percent',
            'Battery state of charge percentage',
            registry=self.registry
        )
        self.battery_power_mw = Gauge(
            'dems_battery_power_mw',
            'Battery power (positive=discharge, negative=charge)',
            registry=self.registry
        )
        self.battery_unit_count = Gauge(
            'dems_battery_unit_count',
            'Number of battery units',
            registry=self.registry
        )
        
        # EV Charging
        self.ev_max_power_mw = Gauge(
            'dems_ev_max_power_mw',
            'Maximum EV charging power in MW',
            registry=self.registry
        )
        self.ev_current_power_mw = Gauge(
            'dems_ev_current_power_mw',
            'Current EV charging power in MW',
            registry=self.registry
        )
        self.ev_unit_count = Gauge(
            'dems_ev_unit_count',
            'Number of EV chargers',
            registry=self.registry
        )
        
        # Demand Response
        self.dr_available_mw = Gauge(
            'dems_dr_available_mw',
            'Available demand response capacity in MW',
            registry=self.registry
        )
        self.dr_curtailed_mw = Gauge(
            'dems_dr_curtailed_mw',
            'Currently curtailed load in MW',
            registry=self.registry
        )
        self.dr_unit_count = Gauge(
            'dems_dr_unit_count',
            'Number of demand response units',
            registry=self.registry
        )
        
        # ==================== CONVERGENCE METRICS ==================== #
        
        self.power_flow_converged = Gauge(
            'dems_power_flow_converged',
            'Power flow convergence status (1=converged, 0=not converged)',
            registry=self.registry
        )
        self.power_flow_iterations = Gauge(
            'dems_power_flow_iterations',
            'Number of iterations for power flow convergence',
            registry=self.registry
        )
        self.power_flow_duration_ms = Gauge(
            'dems_power_flow_duration_ms',
            'Power flow calculation duration in milliseconds',
            registry=self.registry
        )
        
        # ==================== GRID SECURITY ==================== #
        
        self.grid_secure = Gauge(
            'dems_grid_secure',
            'Grid security status (1=secure, 0=insecure)',
            registry=self.registry
        )
        
        # ==================== SIMULATION METRICS ==================== #
        
        self.simulation_step = Counter(
            'dems_simulation_step_total',
            'Total number of simulation steps executed',
            registry=self.registry
        )
        self.simulation_errors = Counter(
            'dems_simulation_errors_total',
            'Total number of simulation errors',
            ['error_type'],
            registry=self.registry
        )
        
        # ==================== PERFORMANCE METRICS ==================== #
        
        self.metrics_update_duration = Histogram(
            'dems_metrics_update_duration_seconds',
            'Time taken to update metrics',
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
            registry=self.registry
        )
        
        logger.info(f"DEMSPrometheusExporter initialized on port {port}")
    
    def start(self) -> None:
        """Start the Prometheus HTTP server"""
        try:
            start_http_server(self.port, registry=self.registry)
            logger.info(f"✓ Prometheus metrics server started on port {self.port}")
            logger.info(f"  Metrics endpoint: http://localhost:{self.port}/metrics")
        except Exception as e:
            logger.error(f"Failed to start Prometheus server: {e}")
            raise
    
    def update_from_power_flow_result(self, result: Any) -> None:
        """
        Update metrics from PowerFlowResult
        
        Args:
            result: PowerFlowResult from orchestrator
        """
        start_time = time.time()
        
        with self._lock:
            try:
                # Power flow metrics
                self.total_generation_mw.set(result.total_generation_mw)
                self.total_generation_mvar.set(result.total_generation_mvar)
                self.total_load_mw.set(result.total_load_mw)
                self.total_load_mvar.set(result.total_load_mvar)
                self.total_losses_mw.set(result.total_losses_mw)
                self.total_losses_mvar.set(result.total_losses_mvar)
                
                # Loss percentage
                if result.total_generation_mw > 0:
                    loss_pct = (result.total_losses_mw / result.total_generation_mw) * 100
                    self.loss_percentage.set(loss_pct)
                
                # Voltage metrics
                self.min_voltage_pu.set(result.min_voltage_pu)
                self.max_voltage_pu.set(result.max_voltage_pu)
                self.avg_voltage_pu.set(result.avg_voltage_pu)
                self.voltage_violations.set(result.num_voltage_violations)
                
                # Loading metrics
                self.line_overloads.set(result.num_line_overloads)
                self.trafo_overloads.set(result.num_trafo_overloads)
                
                # Convergence
                self.power_flow_converged.set(1 if result.converged else 0)
                self.power_flow_iterations.set(result.iterations)
                self.power_flow_duration_ms.set(result.elapsed_time_ms)
                
                # Grid security
                self.grid_secure.set(1 if result.is_secure else 0)
                
                # Increment simulation step
                self.simulation_step.inc()
                
                logger.debug("Updated power flow metrics")
                
            except Exception as e:
                logger.error(f"Error updating power flow metrics: {e}")
                self.simulation_errors.labels(error_type='power_flow_update').inc()
            finally:
                duration = time.time() - start_time
                self.metrics_update_duration.observe(duration)
    
    def update_from_grid_state(self, state: Dict[str, Any]) -> None:
        """
        Update metrics from grid state
        
        Args:
            state: Grid state dictionary from orchestrator
        """
        start_time = time.time()
        
        with self._lock:
            try:
                # Global metrics
                if 'global_metrics' in state:
                    gm = state['global_metrics']
                    self.avg_frequency_hz.set(gm.get('avg_frequency_hz', 50.0))
                    self.system_frequency_hz.set(gm.get('avg_frequency_hz', 50.0))
                
                # Area metrics
                for area_name in ['area_A', 'area_B', 'area_C']:
                    if area_name in state:
                        area_data = state[area_name]
                        area_label = area_name.split('_')[1]  # 'A', 'B', or 'C'
                        
                        self.area_generation_mw.labels(area=area_label).set(
                            area_data.get('total_generation_mw', 0)
                        )
                        self.area_load_mw.labels(area=area_label).set(
                            area_data.get('total_load_mw', 0)
                        )
                        self.area_net_export_mw.labels(area=area_label).set(
                            area_data.get('net_export_mw', 0)
                        )
                
                # Tie-line metrics
                if 'tie_lines' in state:
                    for tie_line in state['tie_lines']:
                        name = tie_line.get('name', 'unknown')
                        from_area = name.split('_')[0] if '_' in name else 'unknown'
                        to_area = name.split('_')[1] if '_' in name else 'unknown'
                        
                        self.tie_line_flow_mw.labels(
                            tie_line=name,
                            from_area=from_area,
                            to_area=to_area
                        ).set(tie_line.get('p_from_mw', 0))
                        
                        self.tie_line_loading_pct.labels(
                            tie_line=name
                        ).set(tie_line.get('loading_percent', 0))
                
                logger.debug("Updated grid state metrics")
                
            except Exception as e:
                logger.error(f"Error updating grid state metrics: {e}")
                self.simulation_errors.labels(error_type='grid_state_update').inc()
            finally:
                duration = time.time() - start_time
                self.metrics_update_duration.observe(duration)
    
    def update_from_der_status(self, der_status: Dict[str, Any]) -> None:
        """
        Update DER metrics from DER manager status
        
        Args:
            der_status: DER status dictionary from der_manager.get_status()
        """
        start_time = time.time()
        
        with self._lock:
            try:
                # Solar metrics
                if 'solar' in der_status:
                    solar = der_status['solar']
                    self.solar_capacity_mw.set(solar.get('total_capacity_mw', 0))
                    self.solar_output_mw.set(solar.get('current_output_mw', 0))
                    self.solar_unit_count.set(solar.get('unit_count', 0))
                
                # Wind metrics
                if 'wind' in der_status:
                    wind = der_status['wind']
                    self.wind_capacity_mw.set(wind.get('total_capacity_mw', 0))
                    self.wind_output_mw.set(wind.get('current_output_mw', 0))
                    self.wind_unit_count.set(wind.get('unit_count', 0))
                
                # Battery metrics
                if 'battery' in der_status:
                    battery = der_status['battery']
                    self.battery_capacity_mwh.set(battery.get('capacity_mwh', 0))
                    self.battery_soc_pct.set(battery.get('current_soc_pct', 0))
                    self.battery_power_mw.set(battery.get('current_power_mw', 0))
                    self.battery_unit_count.set(battery.get('unit_count', 0))
                
                # EV Charging metrics
                if 'ev_charger' in der_status:
                    ev = der_status['ev_charger']
                    self.ev_max_power_mw.set(ev.get('max_power_mw', 0))
                    self.ev_current_power_mw.set(ev.get('current_power_mw', 0))
                    self.ev_unit_count.set(ev.get('unit_count', 0))
                
                # Demand Response metrics
                if 'demand_response' in der_status:
                    dr = der_status['demand_response']
                    self.dr_available_mw.set(dr.get('available_mw', 0))
                    self.dr_curtailed_mw.set(dr.get('curtailed_mw', 0))
                    self.dr_unit_count.set(dr.get('unit_count', 0))
                
                logger.debug("Updated DER metrics")
                
            except Exception as e:
                logger.error(f"Error updating DER metrics: {e}")
                self.simulation_errors.labels(error_type='der_update').inc()
            finally:
                duration = time.time() - start_time
                self.metrics_update_duration.observe(duration)
    
    def shutdown(self) -> None:
        """Cleanup and shutdown"""
        logger.info("Shutting down Prometheus exporter")


if __name__ == "__main__":
    # Test the exporter
    logging.basicConfig(level=logging.INFO)
    
    exporter = DEMSPrometheusExporter(port=9136)
    exporter.start()
    
    print("Prometheus exporter running on http://localhost:9136/metrics")
    print("Press Ctrl+C to stop")
    
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        exporter.shutdown()
