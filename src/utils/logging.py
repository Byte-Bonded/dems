"""
DEMS Logging Utilities
Centralized logging configuration
"""

import logging
import sys
from enum import Enum
from typing import Optional
from pathlib import Path


class LogLevel(Enum):
    """Log level enumeration"""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


def setup_logger(
    name: str = "DEMS",
    level: LogLevel = LogLevel.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
    propagate: bool = False,
) -> logging.Logger:
    """
    Set up a logger with console and optional file output
    
    Args:
        name: Logger name (uses hierarchy, e.g., 'DEMS.grid')
        level: Logging level
        log_file: Optional path to log file
        format_string: Custom format string
        propagate: Whether to propagate to parent loggers
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level.value)
    logger.propagate = propagate
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Default format
    if format_string is None:
        format_string = "%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s"
    
    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level.value)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        # Create log directory if needed
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # Log everything to file
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a child logger under the DEMS namespace
    
    Args:
        name: Logger name (will be prefixed with 'src.')
        
    Returns:
        Logger instance
    """
    return logging.getLogger(f"src.{name}")


def silence_third_party_loggers() -> None:
    """Silence noisy third-party loggers"""
    noisy_loggers = [
        'pandapower',
        'numba',
        'matplotlib',
        'PIL',
        'urllib3',
        'asyncio',
    ]
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
