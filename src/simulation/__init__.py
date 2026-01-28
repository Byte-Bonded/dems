"""
DEMS Simulation Module
Pandapower-based Tri-Area Super-Grid simulation engine

Includes:
- 117-bus SuperGrid (3× IEEE 39-bus)
- DER management (Solar, Wind, Battery, EV, DR)
- Power flow analysis
- Dynamic models (generators, AVR, governor, AGC)
"""

from .supergrid import SuperGrid, AreaConfig
from .tie_lines import TieLineConfig, create_tie_lines
from .power_flow import PowerFlowRunner, PowerFlowResult
from .der import DERManager, DERType, DERSpec, DERState
from .dynamics import (
    DynamicsCoordinator,
    SynchronousGeneratorDynamic,
    ExcitationSystem,
    GovernorTurbine,
    PowerSystemStabilizer,
    AutomaticGenerationControl,
    DynamicLoadModel,
    ProtectionRelay,
    IEEE39_GENERATOR_DATA,
)

__all__ = [
    # Grid
    "SuperGrid",
    "AreaConfig", 
    "TieLineConfig",
    "create_tie_lines",
    # Power Flow
    "PowerFlowRunner",
    "PowerFlowResult",
    # DER
    "DERManager",
    "DERType",
    "DERSpec",
    "DERState",
    # Dynamics
    "DynamicsCoordinator",
    "SynchronousGeneratorDynamic",
    "ExcitationSystem",
    "GovernorTurbine",
    "PowerSystemStabilizer",
    "AutomaticGenerationControl",
    "DynamicLoadModel",
    "ProtectionRelay",
    "IEEE39_GENERATOR_DATA",
]
