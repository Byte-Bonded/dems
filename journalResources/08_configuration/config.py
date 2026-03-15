"""
Configuration management for DEMS
Centralized configuration with environment variable support
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional
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
class MultiAgentRLConfig:
    """Hierarchical multi-agent PPO configuration"""
    algorithm: str = "PPO"
    # Central coordinating agent
    central_learning_rate: float = field(default_factory=lambda: _get_env_float("CENTRAL_LR", 3e-4))
    central_net_arch: List[int] = field(default_factory=lambda: [256, 256])
    central_n_steps: int = field(default_factory=lambda: _get_env_int("CENTRAL_N_STEPS", 2048))
    central_batch_size: int = field(default_factory=lambda: _get_env_int("CENTRAL_BATCH_SIZE", 64))
    # Microgrid-level agents (×3)
    mg_learning_rate: float = field(default_factory=lambda: _get_env_float("MG_LR", 3e-4))
    mg_net_arch: List[int] = field(default_factory=lambda: [256, 256])
    mg_n_steps: int = field(default_factory=lambda: _get_env_int("MG_N_STEPS", 2048))
    mg_batch_size: int = field(default_factory=lambda: _get_env_int("MG_BATCH_SIZE", 64))
    # Sub-agents (inverter, renewable, load × 3)
    sub_learning_rate: float = field(default_factory=lambda: _get_env_float("SUB_LR", 3e-4))
    sub_net_arch: List[int] = field(default_factory=lambda: [128, 128])
    sub_n_steps: int = field(default_factory=lambda: _get_env_int("SUB_N_STEPS", 1024))
    sub_batch_size: int = field(default_factory=lambda: _get_env_int("SUB_BATCH_SIZE", 32))
    # Common
    gamma: float = field(default_factory=lambda: _get_env_float("GAMMA", 0.99))
    gae_lambda: float = field(default_factory=lambda: _get_env_float("GAE_LAMBDA", 0.95))
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    # Training schedule
    total_timesteps: int = field(default_factory=lambda: _get_env_int("TOTAL_TIMESTEPS", 1_000_000))
    eval_freq: int = field(default_factory=lambda: _get_env_int("EVAL_FREQ", 10_000))
    n_eval_episodes: int = 5
    # CTDE
    ctde_enabled: bool = True
    shared_critic: bool = True
    tensorboard_log: Optional[str] = field(default_factory=lambda: os.getenv("TENSORBOARD_LOG", "logs/tb"))


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
    rl: MultiAgentRLConfig = field(default_factory=MultiAgentRLConfig)
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
