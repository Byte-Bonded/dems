#!/usr/bin/env python3
"""
Prometheus Monitoring Integration for IEEE 39-Bus System

Provides comprehensive metrics export for the IEEE 39-bus power system simulation:
- Real-time power flow metrics
- Generator performance (10 generators)
- DER system status (Solar, Wind, BESS, EV, DR)
- Voltage and frequency monitoring
- System stability indicators
- Load control metrics
"""

import logging
from typing import Dict, Any, Optional
from prometheus_client import (
    Gauge, Counter, Histogram, Info, Summary,
    CollectorRegistry, REGISTRY
)
import threading
import time

logger = logging.getLogger(__name__)


class IEEE39BusMonitor:
    """
    Prometheus monitoring for IEEE 39-Bus System
    
    Exposes comprehensive metrics for:
    - Power flow analysis (PyPower/PandaPower)
    - Generator dynamics
    - DER coordination
    - System stability
    - Load control
    """
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """Initialize IEEE 39-bus monitoring"""
        self.registry = registry or REGISTRY
        self._lock = threading.Lock()
        
        # ==================== SYSTEM INFO ==================== #
        self.system_info = Info(
            'ieee39_system',
            'IEEE 39-Bus System Information',
            registry=self.registry
        )
        self.system_info.info({
            'version': '1.0.0',
            'system': 'IEEE 39-Bus New England',
            'buses': '39',
            'generators': '10',
            'branches': '46',
            'frequency': '50Hz',
            'der_count': '18'
        })
        
        # ==================== POWER FLOW METRICS ==================== #
        
        # Total Generation
        self.total_generation_mw = Gauge(
            'ieee39_total_generation_mw',
            'Total generation in MW',
            registry=self.registry
        )
        
        self.total_generation_mvar = Gauge(
            'ieee39_total_generation_mvar',
            'Total reactive generation in MVAr',
            registry=self.registry
        )
        
        # Total Load
        self.total_load_mw = Gauge(
            'ieee39_total_load_mw',
            'Total load in MW',
            registry=self.registry
        )
        
        self.total_load_mvar = Gauge(
            'ieee39_total_load_mvar',
            'Total reactive load in MVAr',
            registry=self.registry
        )
        
        # System Losses
        self.total_losses_mw = Gauge(
            'ieee39_total_losses_mw',
            'Total system losses in MW',
            registry=self.registry
        )
        
        # ==================== GENERATOR METRICS ==================== #
        
        self.generator_power_mw = Gauge(
            'ieee39_generator_power_mw',
            'Generator real power output in MW',
            ['bus', 'gen_type'],
            registry=self.registry
        )
        
        self.generator_reactive_mvar = Gauge(
            'ieee39_generator_reactive_mvar',
            'Generator reactive power output in MVAr',
            ['bus', 'gen_type'],
            registry=self.registry
        )
        
        self.generator_voltage_pu = Gauge(
            'ieee39_generator_voltage_pu',
            'Generator terminal voltage in per unit',
            ['bus', 'gen_type'],
            registry=self.registry
        )
        
        # ==================== VOLTAGE METRICS ==================== #
        
        self.bus_voltage_pu = Gauge(
            'ieee39_bus_voltage_pu',
            'Bus voltage in per unit',
            ['bus'],
            registry=self.registry
        )
        
        self.voltage_min_pu = Gauge(
            'ieee39_voltage_min_pu',
            'Minimum bus voltage in system',
            registry=self.registry
        )
        
        self.voltage_max_pu = Gauge(
            'ieee39_voltage_max_pu',
            'Maximum bus voltage in system',
            registry=self.registry
        )
        
        self.voltage_violations = Gauge(
            'ieee39_voltage_violations',
            'Number of buses with voltage violations',
            registry=self.registry
        )
        
        # ==================== FREQUENCY METRICS ==================== #
        
        self.system_frequency_hz = Gauge(
            'ieee39_system_frequency_hz',
            'System frequency in Hz',
            registry=self.registry
        )
        
        self.frequency_deviation_hz = Gauge(
            'ieee39_frequency_deviation_hz',
            'Frequency deviation from nominal (50 Hz)',
            registry=self.registry
        )
        
        # ==================== DER METRICS ==================== #
        
        # Solar PV
        self.solar_generation_mw = Gauge(
            'ieee39_solar_generation_mw',
            'Solar PV generation in MW',
            ['name', 'bus'],
            registry=self.registry
        )
        
        self.solar_irradiance = Gauge(
            'ieee39_solar_irradiance',
            'Solar irradiance in W/m²',
            ['name'],
            registry=self.registry
        )
        
        # Wind Turbine
        self.wind_generation_mw = Gauge(
            'ieee39_wind_generation_mw',
            'Wind turbine generation in MW',
            ['name', 'bus'],
            registry=self.registry
        )
        
        self.wind_speed_mps = Gauge(
            'ieee39_wind_speed_mps',
            'Wind speed in m/s',
            ['name'],
            registry=self.registry
        )
        
        # Battery Energy Storage
        self.bess_power_mw = Gauge(
            'ieee39_bess_power_mw',
            'Battery power (positive=discharge, negative=charge) in MW',
            ['name', 'bus'],
            registry=self.registry
        )
        
        self.bess_soc_percent = Gauge(
            'ieee39_bess_soc_percent',
            'Battery state of charge in percent',
            ['name', 'bus'],
            registry=self.registry
        )
        
        self.bess_energy_mwh = Gauge(
            'ieee39_bess_energy_mwh',
            'Battery energy stored in MWh',
            ['name', 'bus'],
            registry=self.registry
        )
        
        # Electric Vehicle
        self.ev_charging_power_mw = Gauge(
            'ieee39_ev_charging_power_mw',
            'EV aggregator charging power in MW',
            ['name', 'bus'],
            registry=self.registry
        )
        
        self.ev_connected_count = Gauge(
            'ieee39_ev_connected_count',
            'Number of EVs connected',
            ['name', 'bus'],
            registry=self.registry
        )
        
        self.ev_v2g_capable_count = Gauge(
            'ieee39_ev_v2g_capable_count',
            'Number of V2G capable EVs',
            ['name', 'bus'],
            registry=self.registry
        )
        
        # Demand Response
        self.dr_load_mw = Gauge(
            'ieee39_dr_load_mw',
            'Demand response controlled load in MW',
            ['name', 'bus'],
            registry=self.registry
        )
        
        self.dr_reduction_mw = Gauge(
            'ieee39_dr_reduction_mw',
            'Demand response reduction from baseline in MW',
            ['name', 'bus'],
            registry=self.registry
        )
        
        # Total DER Contribution
        self.total_der_generation_mw = Gauge(
            'ieee39_total_der_generation_mw',
            'Total DER generation in MW',
            registry=self.registry
        )
        
        self.total_der_storage_mw = Gauge(
            'ieee39_total_der_storage_mw',
            'Total DER storage dispatch in MW',
            registry=self.registry
        )
        
        # ==================== STABILITY METRICS ==================== #
        
        self.power_flow_converged = Gauge(
            'ieee39_power_flow_converged',
            'Power flow convergence status (1=converged, 0=failed)',
            registry=self.registry
        )
        
        self.power_flow_iterations = Gauge(
            'ieee39_power_flow_iterations',
            'Number of power flow iterations',
            registry=self.registry
        )
        
        self.system_secure = Gauge(
            'ieee39_system_secure',
            'System security status (1=secure, 0=insecure)',
            registry=self.registry
        )
        
        # ==================== LOAD CONTROL METRICS ==================== #
        
        self.load_increase_applied_mw = Gauge(
            'ieee39_load_increase_applied_mw',
            'Load increase applied in MW',
            registry=self.registry
        )
        
        self.der_response_total_mw = Gauge(
            'ieee39_der_response_total_mw',
            'Total DER response to load change in MW',
            registry=self.registry
        )
        
        self.load_control_success = Gauge(
            'ieee39_load_control_success',
            'Load control success indicator (1=success, 0=failure)',
            registry=self.registry
        )
        
        # ==================== PERFORMANCE METRICS ==================== #
        
        self.power_flow_duration_seconds = Histogram(
            'ieee39_power_flow_duration_seconds',
            'Power flow computation duration',
            registry=self.registry
        )
        
        self.update_cycle_duration_seconds = Histogram(
            'ieee39_update_cycle_duration_seconds',
            'Complete update cycle duration',
            registry=self.registry
        )
        
        self.metrics_update_count = Counter(
            'ieee39_metrics_update_count',
            'Total number of metrics updates',
            registry=self.registry
        )
        
        logger.info("✓ IEEE 39-Bus monitoring initialized")
    
    def update_power_flow_metrics(self, result: Dict[str, Any]):
        """Update metrics from power flow analysis result"""
        with self._lock:
            try:
                # Power flow convergence
                converged = result.get('success', False)
                self.power_flow_converged.set(1.0 if converged else 0.0)
                
                if 'iterations' in result:
                    self.power_flow_iterations.set(result['iterations'])
                
                # Bus results
                if 'bus' in result:
                    bus_data = result['bus']
                    voltages = bus_data[:, 7]  # VM column
                    
                    self.voltage_min_pu.set(voltages.min())
                    self.voltage_max_pu.set(voltages.max())
                    
                    # Count voltage violations
                    violations = ((voltages < 0.9) | (voltages > 1.1)).sum()
                    self.voltage_violations.set(float(violations))
                    
                    # Update individual bus voltages
                    for i, v in enumerate(voltages):
                        self.bus_voltage_pu.labels(bus=str(i+1)).set(v)
                
                # Generator results
                if 'gen' in result:
                    gen_data = result['gen']
                    total_pg = gen_data[:, 1].sum()  # PG column
                    total_qg = gen_data[:, 2].sum()  # QG column
                    
                    self.total_generation_mw.set(total_pg)
                    self.total_generation_mvar.set(total_qg)
                
                # Load
                if 'bus' in result:
                    bus_data = result['bus']
                    total_pd = bus_data[:, 2].sum()  # PD column
                    total_qd = bus_data[:, 3].sum()  # QD column
                    
                    self.total_load_mw.set(total_pd)
                    self.total_load_mvar.set(total_qd)
                
                # Losses
                if 'gen' in result and 'bus' in result:
                    losses = total_pg - total_pd
                    self.total_losses_mw.set(losses)
                
                logger.debug(f"Power flow metrics updated (converged={converged})")
                
            except Exception as e:
                logger.error(f"Error updating power flow metrics: {e}")
    
    def update_system_state(self, state: Dict[str, Any]):
        """Update metrics from system state"""
        with self._lock:
            try:
                # Frequency
                freq = state.get('frequency_hz', 50.0)
                self.system_frequency_hz.set(freq)
                self.frequency_deviation_hz.set(freq - 50.0)
                
                # Power flow status
                if 'power_flow_converged' in state:
                    self.power_flow_converged.set(
                        1.0 if state['power_flow_converged'] else 0.0
                    )
                
                # Voltages
                if 'voltage_min' in state:
                    self.voltage_min_pu.set(state['voltage_min'])
                if 'voltage_max' in state:
                    self.voltage_max_pu.set(state['voltage_max'])
                
                # Security
                voltage_ok = (
                    state.get('voltage_min', 1.0) >= 0.9 and
                    state.get('voltage_max', 1.0) <= 1.1
                )
                freq_ok = abs(freq - 50.0) < 1.0
                secure = voltage_ok and freq_ok and state.get('power_flow_converged', False)
                self.system_secure.set(1.0 if secure else 0.0)
                
                logger.debug("System state metrics updated")
                
            except Exception as e:
                logger.error(f"Error updating system state: {e}")
    
    def update_generator_metrics(self, generators: Dict[str, Any]):
        """Update individual generator metrics"""
        with self._lock:
            try:
                for bus_num, gen_data in generators.items():
                    gen_type = gen_data.get('Type', 'Unknown')
                    bus_label = f"Bus_{bus_num}"
                    
                    # Power output
                    if 'P_mech' in gen_data:
                        self.generator_power_mw.labels(
                            bus=bus_label, 
                            gen_type=gen_type
                        ).set(gen_data['P_mech'])
                    
                    # Voltage
                    if 'voltage_pu' in gen_data:
                        self.generator_voltage_pu.labels(
                            bus=bus_label,
                            gen_type=gen_type
                        ).set(gen_data['voltage_pu'])
                
                logger.debug(f"Updated metrics for {len(generators)} generators")
                
            except Exception as e:
                logger.error(f"Error updating generator metrics: {e}")
    
    def update_der_metrics(self, der_systems: Dict[str, Any]):
        """Update DER system metrics"""
        with self._lock:
            try:
                total_gen = 0.0
                total_storage = 0.0
                
                for name, der in der_systems.items():
                    der_type = type(der).__name__
                    location = getattr(der, 'location', 'Unknown')
                    
                    # Solar PV
                    if 'Solar' in der_type:
                        power = getattr(der, 'current_output_mw', 0.0)
                        irradiance = getattr(der, 'irradiance', 0.0)
                        
                        self.solar_generation_mw.labels(
                            name=name, bus=location
                        ).set(power)
                        
                        self.solar_irradiance.labels(name=name).set(irradiance)
                        total_gen += power
                    
                    # Wind Turbine
                    elif 'Wind' in der_type:
                        power = getattr(der, 'current_output_mw', 0.0)
                        wind_speed = getattr(der, 'wind_speed', 0.0)
                        
                        self.wind_generation_mw.labels(
                            name=name, bus=location
                        ).set(power)
                        
                        self.wind_speed_mps.labels(name=name).set(wind_speed)
                        total_gen += power
                    
                    # Battery Storage
                    elif 'Battery' in der_type:
                        power = getattr(der, 'current_power_mw', 0.0)
                        soc = getattr(der, 'soc', 0.5)
                        energy = getattr(der, 'current_energy_mwh', 0.0)
                        
                        self.bess_power_mw.labels(
                            name=name, bus=location
                        ).set(power)
                        
                        self.bess_soc_percent.labels(
                            name=name, bus=location
                        ).set(soc * 100)
                        
                        self.bess_energy_mwh.labels(
                            name=name, bus=location
                        ).set(energy)
                        
                        total_storage += power
                    
                    # Electric Vehicle
                    elif 'ElectricVehicle' in der_type:
                        power = getattr(der, 'total_charging_power_mw', 0.0)
                        connected = getattr(der, 'connected_evs', 0)
                        v2g = getattr(der, 'v2g_capable', 0)
                        
                        self.ev_charging_power_mw.labels(
                            name=name, bus=location
                        ).set(power)
                        
                        self.ev_connected_count.labels(
                            name=name, bus=location
                        ).set(connected)
                        
                        self.ev_v2g_capable_count.labels(
                            name=name, bus=location
                        ).set(v2g)
                    
                    # Demand Response
                    elif 'DemandResponse' in der_type:
                        load = getattr(der, 'current_load_mw', 0.0)
                        baseline = getattr(der, 'baseline_load', 0.0)
                        reduction = baseline - load
                        
                        self.dr_load_mw.labels(
                            name=name, bus=location
                        ).set(load)
                        
                        self.dr_reduction_mw.labels(
                            name=name, bus=location
                        ).set(reduction)
                
                # Update totals
                self.total_der_generation_mw.set(total_gen)
                self.total_der_storage_mw.set(total_storage)
                
                logger.debug(f"Updated metrics for {len(der_systems)} DER systems")
                
            except Exception as e:
                logger.error(f"Error updating DER metrics: {e}")
    
    def update_load_control_metrics(self, results: Dict[str, Any]):
        """Update load control test metrics"""
        with self._lock:
            try:
                if 'load_increase_applied' in results:
                    self.load_increase_applied_mw.set(results['load_increase_applied'])
                
                if 'der_contribution' in results:
                    self.der_response_total_mw.set(results['der_contribution'])
                
                if 'load_control_success' in results:
                    self.load_control_success.set(
                        1.0 if results['load_control_success'] else 0.0
                    )
                
                logger.debug("Load control metrics updated")
                
            except Exception as e:
                logger.error(f"Error updating load control metrics: {e}")
    
    def record_power_flow_duration(self, duration_seconds: float):
        """Record power flow computation time"""
        self.power_flow_duration_seconds.observe(duration_seconds)
    
    def increment_update_count(self):
        """Increment metrics update counter"""
        self.metrics_update_count.inc()


def create_monitor() -> IEEE39BusMonitor:
    """Create and return IEEE 39-Bus monitor instance"""
    return IEEE39BusMonitor()


if __name__ == "__main__":
    print("IEEE 39-Bus Prometheus Monitor")
    print("=" * 60)
    
    monitor = create_monitor()
    print("✓ Monitor initialized successfully")
    print(f"  • Total metrics: ~100+ time series")
    print(f"  • Generator monitoring: 10 units")
    print(f"  • DER monitoring: 18 systems")
    print(f"  • Bus monitoring: 39 buses")
