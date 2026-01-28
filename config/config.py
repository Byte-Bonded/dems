"""
Configuration management for DEMS
Centralized configuration with environment variable support
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional


def _get_env_bool(key: str, default: bool = False) -> bool:
    """Get boolean from environment variable"""
    return os.getenv(key, str(default)).lower() in ("true", "1", "yes")


def _get_env_int(key: str, default: int) -> int:
    """Get integer from environment variable"""
    return int(os.getenv(key, str(default)))


def _get_env_float(key: str, default: float) -> float:
    """Get float from environment variable"""
    return float(os.getenv(key, str(default)))


@dataclass
class APIConfig:
    """API server configuration"""
    host: str = field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _get_env_int("API_PORT", 8000))
    debug: bool = field(default_factory=lambda: _get_env_bool("DEBUG", False))
    workers: int = field(default_factory=lambda: _get_env_int("API_WORKERS", 4))


@dataclass
class GridConfig:
    """Power grid configuration"""
    num_areas: int = 3
    buses_per_area: int = 39
    nominal_frequency_hz: float = 50.0
    voltage_limits_pu: tuple = (0.95, 1.05)
    frequency_limits_hz: tuple = (49.5, 50.5)


@dataclass
class RLConfig:
    """Reinforcement learning configuration"""
    algorithm: str = field(default_factory=lambda: os.getenv("RL_ALGORITHM", "PPO"))
    learning_rate: float = field(default_factory=lambda: _get_env_float("LEARNING_RATE", 0.0003))
    batch_size: int = field(default_factory=lambda: _get_env_int("BATCH_SIZE", 64))
    n_steps: int = field(default_factory=lambda: _get_env_int("N_STEPS", 2048))
    gamma: float = field(default_factory=lambda: _get_env_float("GAMMA", 0.99))
    tensorboard_log: Optional[str] = field(default_factory=lambda: os.getenv("TENSORBOARD_LOG"))


@dataclass
class MonitoringConfig:
    """Monitoring and metrics configuration"""
    prometheus_port: int = field(default_factory=lambda: _get_env_int("PROMETHEUS_PORT", 9090))
    grafana_port: int = field(default_factory=lambda: _get_env_int("GRAFANA_PORT", 3000))
    metrics_interval_s: float = field(default_factory=lambda: _get_env_float("METRICS_INTERVAL", 1.0))


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    log_dir: Path = field(default_factory=lambda: Path(os.getenv("LOG_DIR", "logs")))
    log_to_file: bool = field(default_factory=lambda: _get_env_bool("LOG_TO_FILE", True))


@dataclass 
class Config:
    """
    Main DEMS configuration
    
    Aggregates all configuration sections and provides environment variable
    overrides for containerized deployments.
    """
    api: APIConfig = field(default_factory=APIConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Legacy flat attributes for backward compatibility
    @property
    def API_HOST(self) -> str:
        return self.api.host
    
    @property
    def API_PORT(self) -> int:
        return self.api.port
    
    @property
    def DEBUG(self) -> bool:
        return self.api.debug
    
    @property
    def LOG_LEVEL(self) -> str:
        return self.logging.level

    def __repr__(self) -> str:
        return (
            f"Config(api={self.api.host}:{self.api.port}, "
            f"grid={self.grid.num_areas}x{self.grid.buses_per_area} buses, "
            f"debug={self.api.debug})"
        )


# Global configuration instance
config = Config()


# Legacy exports for backward compatibility
API_HOST = config.api.host
API_PORT = config.api.port
DEBUG = config.api.debug
LOG_LEVEL = config.logging.level
