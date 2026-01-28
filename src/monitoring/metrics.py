"""
DEMS Monitoring Module
Prometheus metrics and system monitoring utilities

Provides:
- Prometheus metrics for energy, grid, and agent monitoring
- MetricsCollector for centralized metrics updates
- Performance tracking decorators
"""

from prometheus_client import Counter, Gauge, Histogram
import functools
from typing import Callable, Any


# ======================== ENERGY METRICS ======================== #

energy_generated = Gauge(
    "dems_energy_generated_kwh", 
    "Total energy generated in kWh"
)
energy_consumed = Gauge(
    "dems_energy_consumed_kwh", 
    "Total energy consumed in kWh"
)
storage_level = Gauge(
    "dems_storage_level_kwh", 
    "Current storage level in kWh"
)


# ======================== GRID METRICS ======================== #

grid_frequency = Gauge(
    "dems_grid_frequency_hz", 
    "Grid frequency in Hz"
)
grid_voltage = Gauge(
    "dems_grid_voltage_v", 
    "Grid voltage in V"
)
grid_generation_mw = Gauge(
    "dems_grid_generation_mw",
    "Total grid generation in MW"
)
grid_load_mw = Gauge(
    "dems_grid_load_mw",
    "Total grid load in MW"
)
grid_losses_mw = Gauge(
    "dems_grid_losses_mw",
    "Total grid losses in MW"
)
voltage_violations = Gauge(
    "dems_voltage_violations",
    "Number of voltage violations"
)
line_overloads = Gauge(
    "dems_line_overloads",
    "Number of line overloads"
)


# ======================== PERFORMANCE METRICS ======================== #

optimization_requests = Counter(
    "dems_optimization_requests_total", 
    "Total optimization requests"
)
optimization_errors = Counter(
    "dems_optimization_errors_total", 
    "Total optimization errors"
)
optimization_duration = Histogram(
    "dems_optimization_duration_seconds", 
    "Optimization duration in seconds",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
)
power_flow_duration = Histogram(
    "dems_power_flow_duration_seconds",
    "Power flow calculation duration in seconds",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5)
)


# ======================== AGENT METRICS ======================== #

agent_training_episodes = Counter(
    "dems_agent_training_episodes_total", 
    "Total training episodes"
)
agent_rewards = Gauge(
    "dems_agent_reward", 
    "Current agent reward"
)
agent_epsilon = Gauge(
    "dems_agent_epsilon",
    "Current exploration epsilon"
)


# ======================== DECORATORS ======================== #

def track_performance(func: Callable) -> Callable:
    """Decorator to track function performance with Prometheus metrics"""
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        optimization_requests.inc()
        with optimization_duration.time():
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                optimization_errors.inc()
                raise
    return wrapper


def track_power_flow(func: Callable) -> Callable:
    """Decorator to track power flow execution time"""
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with power_flow_duration.time():
            return func(*args, **kwargs)
    return wrapper


# ======================== METRICS COLLECTOR ======================== #

class MetricsCollector:
    """
    Centralized metrics collector for DEMS
    
    Provides methods to update all Prometheus metrics from various
    system components.
    """

    def __init__(self):
        self.metrics = {}

    def update_energy_metrics(
        self, 
        generated: float, 
        consumed: float, 
        storage: float
    ) -> None:
        """
        Update energy metrics
        
        Args:
            generated: Total energy generated (kWh)
            consumed: Total energy consumed (kWh)
            storage: Current storage level (kWh)
        """
        energy_generated.set(generated)
        energy_consumed.set(consumed)
        storage_level.set(storage)

    def update_grid_metrics(
        self, 
        frequency: float, 
        voltage: float,
        generation_mw: float = 0,
        load_mw: float = 0,
        losses_mw: float = 0
    ) -> None:
        """
        Update grid metrics
        
        Args:
            frequency: Grid frequency (Hz)
            voltage: Grid voltage (V)
            generation_mw: Total generation (MW)
            load_mw: Total load (MW)
            losses_mw: Total losses (MW)
        """
        grid_frequency.set(frequency)
        grid_voltage.set(voltage)
        if generation_mw:
            grid_generation_mw.set(generation_mw)
        if load_mw:
            grid_load_mw.set(load_mw)
        if losses_mw:
            grid_losses_mw.set(losses_mw)

    def update_violations(
        self,
        num_voltage_violations: int = 0,
        num_line_overloads: int = 0
    ) -> None:
        """
        Update violation counts
        
        Args:
            num_voltage_violations: Number of voltage violations
            num_line_overloads: Number of line overloads
        """
        voltage_violations.set(num_voltage_violations)
        line_overloads.set(num_line_overloads)

    def record_agent_reward(self, reward: float) -> None:
        """Record agent reward"""
        agent_rewards.set(reward)
        
    def record_training_episode(self) -> None:
        """Increment training episode counter"""
        agent_training_episodes.inc()


# Singleton instance
metrics_collector = MetricsCollector()


__all__ = [
    # Metrics
    "energy_generated",
    "energy_consumed", 
    "storage_level",
    "grid_frequency",
    "grid_voltage",
    "grid_generation_mw",
    "grid_load_mw",
    "grid_losses_mw",
    "voltage_violations",
    "line_overloads",
    "optimization_requests",
    "optimization_errors",
    "optimization_duration",
    "power_flow_duration",
    "agent_training_episodes",
    "agent_rewards",
    "agent_epsilon",
    # Decorators
    "track_performance",
    "track_power_flow",
    # Classes
    "MetricsCollector",
    "metrics_collector",
]
