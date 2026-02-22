"""
Dynamic Energy Management System (DEMS)
A RL Agentic Framework for intelligent energy management

Modules:
- grid: High-level grid interface (DEMSGrid)
- simulation: Pandapower-based power system simulation
- agent: RL agents and Gymnasium environment
- core: Energy management and grid monitoring
- utils: Common utilities and decorators
- orchestrator: Grid orchestration and logging
"""

__version__ = "0.1.0"
__author__ = "DEMS Team"

# Core components
from .core import EnergyManager, GridMonitor, StabilityStatus
from .agent import RLAgent, DEMSEnvironment
from .grid import DEMSGrid
from .orchestrator import MonitoringOrchestrator
# FIX BUG-18: backward compat alias
GridOrchestrator = MonitoringOrchestrator

__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Grid
    "DEMSGrid",
    "GridOrchestrator",
    # Core
    "EnergyManager",
    "GridMonitor",
    "StabilityStatus",
    # Agent
    "RLAgent",
    "DEMSEnvironment",
]
