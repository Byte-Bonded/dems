"""Monitoring utilities for Prometheus integration"""

from prometheus_client import Counter, Gauge, Histogram
import functools
from typing import Callable, Any


# Energy Metrics
energy_generated = Gauge("dems_energy_generated_kwh", "Total energy generated")
energy_consumed = Gauge("dems_energy_consumed_kwh", "Total energy consumed")
storage_level = Gauge("dems_storage_level_kwh", "Current storage level")
grid_frequency = Gauge("dems_grid_frequency_hz", "Grid frequency")
grid_voltage = Gauge("dems_grid_voltage_v", "Grid voltage")

# Performance Metrics
optimization_requests = Counter(
    "dems_optimization_requests_total", "Total optimization requests"
)
optimization_errors = Counter(
    "dems_optimization_errors_total", "Total optimization errors"
)
optimization_duration = Histogram(
    "dems_optimization_duration_seconds", "Optimization duration"
)

# Agent Metrics
agent_training_episodes = Counter(
    "dems_agent_training_episodes_total", "Total training episodes"
)
agent_rewards = Gauge("dems_agent_reward", "Current agent reward")


def track_performance(func: Callable) -> Callable:
    """Decorator to track function performance"""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with optimization_duration.time():
            try:
                result = func(*args, **kwargs)
                optimization_requests.inc()
                return result
            except Exception as e:
                optimization_errors.inc()
                raise

    return wrapper


class MetricsCollector:
    """Collects and publishes metrics"""

    def __init__(self):
        self.metrics = {}

    def update_energy_metrics(
        self, generated: float, consumed: float, storage: float
    ) -> None:
        """Update energy metrics"""
        energy_generated.set(generated)
        energy_consumed.set(consumed)
        storage_level.set(storage)

    def update_grid_metrics(self, frequency: float, voltage: float) -> None:
        """Update grid metrics"""
        grid_frequency.set(frequency)
        grid_voltage.set(voltage)

    def record_agent_reward(self, reward: float) -> None:
        """Record agent reward"""
        agent_rewards.set(reward)
