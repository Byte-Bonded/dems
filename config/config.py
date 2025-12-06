"""
Configuration management
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    """Application configuration"""

    # API Settings
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", 8000))
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"

    # Energy Grid Settings
    GRID_SIZE: int = int(os.getenv("GRID_SIZE", 10))
    STORAGE_CAPACITY: float = float(os.getenv("STORAGE_CAPACITY", 1000.0))
    NUM_NODES: int = int(os.getenv("NUM_NODES", 10))

    # RL Agent Settings
    OBSERVATION_SIZE: int = int(os.getenv("OBSERVATION_SIZE", 50))
    ACTION_SIZE: int = int(os.getenv("ACTION_SIZE", 10))
    LEARNING_RATE: float = float(os.getenv("LEARNING_RATE", 0.0003))

    # Monitoring Settings
    PROMETHEUS_PORT: int = int(os.getenv("PROMETHEUS_PORT", 9090))
    GRAFANA_PORT: int = int(os.getenv("GRAFANA_PORT", 3000))

    # Database Settings
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", "postgresql://user:password@localhost:5432/dems"
    )
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    def __repr__(self) -> str:
        return f"Config(API={self.API_HOST}:{self.API_PORT}, GRID={self.GRID_SIZE}, DEBUG={self.DEBUG})"


config = Config()
