"""
DEMS Simulation Module
Pandapower-based power system simulation engine

Current System: Kundur Two-Area System (optimized for PSS testing)
Legacy: IEEE 39-bus triple system (archived in legacy/ieee39bus/)

Includes:
- Kundur Two-Area System (4 generators, 11 buses)
- DER management (Solar, Wind, Battery, EV, DR)
- Power flow analysis
- Dynamic models (generators, AVR, governor, AGC, PSS)
"""

from .kundur import KundurTwoAreaSystem, KundurConfig, AreaID, GeneratorParams, KUNDUR_GENERATORS
from .power_flow import PowerFlowRunner, PowerFlowResult, PowerFlowConfig
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
    # Kundur System
    "KundurTwoAreaSystem",
    "KundurConfig",
    "AreaID",
    "GeneratorParams",
    "KUNDUR_GENERATORS",
    # Power Flow
    "PowerFlowRunner",
    "PowerFlowResult",
    "PowerFlowConfig",
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
