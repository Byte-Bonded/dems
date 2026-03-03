"""
DEMS Prometheus Metrics Exporter

Exposes metrics from the DEMS simulation and RL agent for Prometheus scraping.
Integrates with GridOrchestrator and training callbacks.

Metrics exposed:
- System frequency and voltage
- Power generation, load, losses
- DER output (solar, wind, battery)
- RL agent training progress
- Economic and carbon metrics
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Optional, Dict, Any

from prometheus_client import (
    Counter, Gauge, Histogram, Info, Summary,
    start_http_server, REGISTRY, generate_latest
)

if TYPE_CHECKING:
    from src.simulation.orchestrator import GridOrchestrator

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM METRICS
# ─────────────────────────────────────────────────────────────────────────────

# Frequency
dems_system_frequency = Gauge(
    'dems_system_frequency_hz',
    'System-wide frequency (Hz)',
    ['region']
)
dems_rocof = Gauge(
    'dems_rocof_hz_per_s',
    'Rate of Change of Frequency (Hz/s)'
)

# Voltage
dems_min_voltage = Gauge('dems_min_voltage_pu', 'Minimum bus voltage (p.u.)')
dems_max_voltage = Gauge('dems_max_voltage_pu', 'Maximum bus voltage (p.u.)')
dems_avg_voltage = Gauge('dems_avg_voltage_pu', 'Average bus voltage (p.u.)')
dems_voltage_violations = Gauge(
    'dems_voltage_violations_total',
    'Number of buses with voltage violations'
)

# Power balance
dems_generation = Gauge('dems_generation_mw', 'Total generation (MW)')
dems_load = Gauge('dems_load_mw', 'Total load (MW)')
dems_losses = Gauge('dems_losses_mw', 'Total system losses (MW)')
dems_loss_percentage = Gauge('dems_loss_percentage', 'System losses (%)')

# Area metrics
dems_area_frequency = Gauge(
    'dems_area_frequency_hz',
    'Area frequency (Hz)',
    ['area']
)
dems_area_load = Gauge(
    'dems_area_load_mw',
    'Area load (MW)',
    ['area']
)
dems_area_generation = Gauge(
    'dems_area_generation_mw',
    'Area generation (MW)',
    ['area']
)

# Tie-lines
dems_tie_ab_flow = Gauge('dems_tie_ab_flow_mw', 'Tie-line A→B flow (MW)')
dems_tie_bc_flow = Gauge('dems_tie_bc_flow_mw', 'Tie-line B→C flow (MW)')
dems_tie_ac_flow = Gauge('dems_tie_ac_flow_mw', 'Tie-line A→C flow (MW)')

# Protection
dems_ufls_activated = Gauge('dems_ufls_activated', 'UFLS protection active (1/0)')
dems_line_overloads = Gauge('dems_line_overloads', 'Number of line overloads')
dems_protection_trips = Gauge('dems_protection_trips', 'Protection trip count')

# ─────────────────────────────────────────────────────────────────────────────
# DER METRICS
# ─────────────────────────────────────────────────────────────────────────────

dems_solar_output = Gauge('dems_solar_output_mw', 'Solar PV output (MW)')
dems_wind_output = Gauge('dems_wind_output_mw', 'Wind output (MW)')
dems_battery_discharge = Gauge('dems_battery_discharge_mw', 'Battery discharge (MW)')
dems_battery_soc = Gauge('dems_battery_soc', 'Battery state of charge (0-1)')
dems_renewable_fraction = Gauge('dems_renewable_fraction', 'Renewable fraction (0-1)')
dems_solar_curtailment = Gauge('dems_solar_curtailment_frac', 'Solar curtailment (0-1)')
dems_wind_curtailment = Gauge('dems_wind_curtailment_frac', 'Wind curtailment (0-1)')

# ─────────────────────────────────────────────────────────────────────────────
# RL AGENT METRICS
# ─────────────────────────────────────────────────────────────────────────────

dems_rl_training_active = Gauge('dems_rl_training_active', 'Training active (1/0)')
dems_rl_training_steps = Counter('dems_rl_training_steps', 'Total training steps')
dems_rl_episodes = Counter('dems_rl_episodes_total', 'Total episodes completed')
dems_rl_episode_reward = Gauge('dems_rl_episode_reward', 'Last episode reward')
dems_rl_episode_violations = Gauge('dems_rl_episode_violations', 'Last episode violations')
dems_rl_mean_reward = Gauge('dems_rl_mean_reward', 'Mean reward (100 ep window)')
dems_rl_policy_loss = Gauge('dems_rl_policy_loss', 'Policy loss')
dems_rl_value_loss = Gauge('dems_rl_value_loss', 'Value loss')
dems_rl_entropy = Gauge('dems_rl_entropy', 'Policy entropy')
dems_rl_eval_reward = Gauge('dems_rl_eval_reward', 'Evaluation mean reward')

# ─────────────────────────────────────────────────────────────────────────────
# ECONOMIC METRICS
# ─────────────────────────────────────────────────────────────────────────────

dems_operating_cost = Gauge('dems_operating_cost_usd_h', 'Operating cost ($/h)')
dems_smc = Gauge('dems_system_marginal_cost_usd_mwh', 'System marginal cost ($/MWh)')
dems_tou_price = Gauge('dems_tou_price_usd_mwh', 'TOU price ($/MWh)')
dems_rtp_price = Gauge('dems_rtp_price_usd_mwh', 'RTP price ($/MWh)')

# Carbon
dems_carbon_emissions = Gauge('dems_carbon_emissions_tco2_h', 'Carbon emissions (tCO2/h)')
dems_carbon_intensity = Gauge('dems_carbon_intensity_kgco2_mwh', 'Carbon intensity (kgCO2/MWh)')

# ─────────────────────────────────────────────────────────────────────────────
# PERFORMANCE METRICS
# ─────────────────────────────────────────────────────────────────────────────

dems_simulation_step_time = Gauge(
    'dems_simulation_step_time_seconds',
    'Time to execute one simulation step'
)
dems_powerflow_convergence_failures = Counter(
    'dems_powerflow_convergence_failures',
    'Power flow convergence failures'
)

dems_api_request_duration = Histogram(
    'dems_api_request_duration_seconds',
    'API request duration',
    ['method', 'endpoint'],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)


class DEMSMetricsExporter:
    """
    Exports DEMS metrics to Prometheus.
    
    Usage:
        exporter = DEMSMetricsExporter(orchestrator)
        exporter.start(port=9091)
        
        # In simulation loop:
        exporter.update()
        
        # When done:
        exporter.stop()
    """
    
    def __init__(
        self,
        orchestrator: Optional['GridOrchestrator'] = None,
        port: int = 9091,
        update_interval: float = 1.0
    ):
        self.orchestrator = orchestrator
        self.port = port
        self.update_interval = update_interval
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_state: Optional[Dict[str, Any]] = None
        
    def start(self, port: Optional[int] = None):
        """Start the metrics HTTP server."""
        if self._running:
            logger.warning("Metrics exporter already running")
            return
            
        if port is not None:
            self.port = port
            
        try:
            start_http_server(self.port)
            logger.info(f"Prometheus metrics server started on port {self.port}")
            self._running = True
            
            # Start background update thread if orchestrator is set
            if self.orchestrator is not None:
                self._thread = threading.Thread(target=self._update_loop, daemon=True)
                self._thread.start()
                
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
            raise
            
    def stop(self):
        """Stop the metrics exporter."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
            
    def _update_loop(self):
        """Background loop to update metrics."""
        while self._running:
            try:
                self.update()
            except Exception as e:
                logger.error(f"Error updating metrics: {e}")
            time.sleep(self.update_interval)
            
    def update(self, state: Optional[Dict[str, Any]] = None):
        """
        Update all metrics from current state.
        
        Args:
            state: Optional state dict. If None, fetches from orchestrator.
        """
        if state is None and self.orchestrator is not None:
            state = self.orchestrator.get_full_state()
            
        if state is None:
            return
            
        self._last_state = state
        self._update_system_metrics(state)
        self._update_der_metrics(state)
        self._update_economic_metrics(state)
        
    def _update_system_metrics(self, state: Dict[str, Any]):
        """Update system-level metrics."""
        global_metrics = state.get('global_metrics', {})
        
        # Frequency
        freq = global_metrics.get('system_frequency_hz', 50.0)
        dems_system_frequency.labels(region='system').set(freq)
        
        rocof = global_metrics.get('rocof_hz_s', 0.0)
        dems_rocof.set(rocof)
        
        # Voltage
        dems_min_voltage.set(global_metrics.get('min_voltage_pu', 1.0))
        dems_max_voltage.set(global_metrics.get('max_voltage_pu', 1.0))
        dems_avg_voltage.set(global_metrics.get('avg_voltage_pu', 1.0))
        dems_voltage_violations.set(global_metrics.get('voltage_violations', 0))
        
        # Power balance
        gen = global_metrics.get('total_generation_mw', 0.0)
        load = global_metrics.get('total_load_mw', 0.0)
        losses = global_metrics.get('total_losses_mw', 0.0)
        dems_generation.set(gen)
        dems_load.set(load)
        dems_losses.set(losses)
        
        if load > 0:
            dems_loss_percentage.set(100 * losses / load)
        
        # Protection
        dems_ufls_activated.set(1 if global_metrics.get('ufls_active', False) else 0)
        dems_line_overloads.set(global_metrics.get('line_overloads', 0))
        dems_protection_trips.set(global_metrics.get('protection_trips', 0))
        
        # Area metrics
        areas = state.get('areas', {})
        for area_name, area_data in areas.items():
            area_label = area_name.replace('area_', '').upper()
            dems_area_frequency.labels(area=area_label).set(
                area_data.get('frequency_hz', 50.0)
            )
            dems_area_load.labels(area=area_label).set(
                area_data.get('load_mw', 0.0)
            )
            dems_area_generation.labels(area=area_label).set(
                area_data.get('generation_mw', 0.0)
            )
            
        # Tie-lines
        tie_lines = state.get('tie_lines', {})
        dems_tie_ab_flow.set(tie_lines.get('AB', {}).get('flow_mw', 0.0))
        dems_tie_bc_flow.set(tie_lines.get('BC', {}).get('flow_mw', 0.0))
        dems_tie_ac_flow.set(tie_lines.get('AC', {}).get('flow_mw', 0.0))
        
    def _update_der_metrics(self, state: Dict[str, Any]):
        """Update DER metrics."""
        der = state.get('der', {})
        
        # Solar
        solar = der.get('solar', {})
        dems_solar_output.set(solar.get('output_mw', 0.0))
        dems_solar_curtailment.set(solar.get('curtailment_frac', 0.0))
        
        # Wind
        wind = der.get('wind', {})
        dems_wind_output.set(wind.get('output_mw', 0.0))
        dems_wind_curtailment.set(wind.get('curtailment_frac', 0.0))
        
        # Battery
        battery = der.get('battery', {})
        dems_battery_discharge.set(battery.get('discharge_mw', 0.0))
        dems_battery_soc.set(battery.get('soc', 0.5))
        
        # Renewable fraction
        global_metrics = state.get('global_metrics', {})
        dems_renewable_fraction.set(global_metrics.get('renewable_fraction', 0.0))
        
    def _update_economic_metrics(self, state: Dict[str, Any]):
        """Update economic and carbon metrics."""
        economics = state.get('economics', {})
        
        dems_operating_cost.set(economics.get('operating_cost_usd_h', 0.0))
        dems_smc.set(economics.get('system_marginal_cost_usd_mwh', 0.0))
        dems_tou_price.set(economics.get('tou_price_usd_mwh', 0.0))
        dems_rtp_price.set(economics.get('rtp_price_usd_mwh', 0.0))
        
        # Carbon
        carbon = state.get('carbon', economics)  # Fallback to economics
        dems_carbon_emissions.set(carbon.get('carbon_emissions_tco2_h', 0.0))
        dems_carbon_intensity.set(carbon.get('carbon_intensity_kgco2_mwh', 0.0))
        
    def update_rl_metrics(
        self,
        training_active: bool = False,
        steps: int = 0,
        episodes: int = 0,
        episode_reward: float = 0.0,
        episode_violations: int = 0,
        mean_reward: float = 0.0,
        policy_loss: float = 0.0,
        value_loss: float = 0.0,
        entropy: float = 0.0,
        eval_reward: Optional[float] = None
    ):
        """Update RL agent training metrics."""
        dems_rl_training_active.set(1 if training_active else 0)
        dems_rl_episode_reward.set(episode_reward)
        dems_rl_episode_violations.set(episode_violations)
        dems_rl_mean_reward.set(mean_reward)
        dems_rl_policy_loss.set(policy_loss)
        dems_rl_value_loss.set(value_loss)
        dems_rl_entropy.set(entropy)
        
        if eval_reward is not None:
            dems_rl_eval_reward.set(eval_reward)
            
    def update_step_time(self, step_time: float):
        """Update simulation step time metric."""
        dems_simulation_step_time.set(step_time)
        
    def record_powerflow_failure(self):
        """Record a power flow convergence failure."""
        dems_powerflow_convergence_failures.inc()
        
    def get_metrics_text(self) -> str:
        """Get current metrics in Prometheus text format."""
        return generate_latest(REGISTRY).decode('utf-8')


# Convenience functions for standalone use
_global_exporter: Optional[DEMSMetricsExporter] = None


def init_metrics(orchestrator: Optional['GridOrchestrator'] = None, port: int = 9091):
    """Initialize and start the global metrics exporter."""
    global _global_exporter
    _global_exporter = DEMSMetricsExporter(orchestrator, port)
    _global_exporter.start()
    return _global_exporter


def get_exporter() -> Optional[DEMSMetricsExporter]:
    """Get the global metrics exporter instance."""
    return _global_exporter


def update_metrics(state: Optional[Dict[str, Any]] = None):
    """Update metrics using the global exporter."""
    if _global_exporter is not None:
        _global_exporter.update(state)


if __name__ == '__main__':
    # Test the exporter standalone
    import argparse
    
    parser = argparse.ArgumentParser(description='DEMS Prometheus Metrics Exporter')
    parser.add_argument('--port', type=int, default=9091, help='HTTP server port')
    args = parser.parse_args()
    
    exporter = DEMSMetricsExporter(port=args.port)
    exporter.start()
    
    print(f"DEMS Metrics Exporter running on http://localhost:{args.port}/metrics")
    print("Press Ctrl+C to stop")
    
    # Generate some test data
    try:
        import random
        while True:
            # Simulate state updates
            test_state = {
                'global_metrics': {
                    'system_frequency_hz': 50.0 + random.uniform(-0.1, 0.1),
                    'min_voltage_pu': 0.97 + random.uniform(-0.02, 0.02),
                    'max_voltage_pu': 1.03 + random.uniform(-0.02, 0.02),
                    'avg_voltage_pu': 1.0 + random.uniform(-0.01, 0.01),
                    'total_generation_mw': 850 + random.uniform(-50, 50),
                    'total_load_mw': 800 + random.uniform(-30, 30),
                    'total_losses_mw': 15 + random.uniform(-5, 5),
                    'renewable_fraction': 0.35 + random.uniform(-0.1, 0.1),
                },
                'der': {
                    'solar': {'output_mw': 150 + random.uniform(-20, 20)},
                    'wind': {'output_mw': 200 + random.uniform(-30, 30)},
                    'battery': {'soc': 0.6 + random.uniform(-0.1, 0.1)},
                },
                'economics': {
                    'operating_cost_usd_h': 5000 + random.uniform(-500, 500),
                    'carbon_intensity_kgco2_mwh': 350 + random.uniform(-50, 50),
                }
            }
            exporter.update(test_state)
            time.sleep(5)
    except KeyboardInterrupt:
        print("\nStopping...")
        exporter.stop()
