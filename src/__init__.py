"""
Dynamic Energy Management System (DEMS)
A Hierarchical Multi-Agent RL Framework for intelligent energy management

Modules:
- grid: High-level grid interface (DEMSGrid)
- simulation: Pandapower-based power system simulation
- agent: Hierarchical multi-agent PPO environments and training
- core: Grid monitoring and stability assessment
- utils: Common utilities and decorators
- orchestrator: Grid orchestration and logging
"""

__version__ = "0.2.0"
__author__ = "DEMS Team"

# Core components
from .core import GridMonitor, StabilityStatus
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
    "GridMonitor",
    "StabilityStatus",
]
