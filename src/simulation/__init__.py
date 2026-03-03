"""
DEMS Simulation Module
Pandapower-based Tri-Area Super-Grid simulation engine

Includes:
- 117-bus SuperGrid (3× IEEE 39-bus)
- DER management (Solar, Wind, EV, DR)
- Power flow analysis
- Dynamic models (generators, AVR, governor, AGC)
- Physics engine for multi-agent RL
- Microgrid simulation (standalone NR solver, DER components)
"""

from .supergrid import SuperGrid, AreaConfig, SuperGridConfig, AreaID
from .tie_lines import TieLineConfig, create_tie_lines
from .power_flow import PowerFlowRunner, PowerFlowResult, PowerFlowAlgorithm
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
    LoadFrequencyController,
)
from .orchestrator import (
    PhysicsEngine,
    GridOrchestrator,  # backward compat alias
    ScenarioConfig,
    StochasticProfileGenerator,
)
from .microgrid import (
    MicrogridCase,
    PowerFlowSolver,
    MicrogridController,
    SolarPV,
    WindTurbine,
    DieselGenerator,
    DERComponent as MicrogridDERComponent,
    create_example_microgrid,
    create_ieee14_case,
)

__all__ = [
    # Grid
    "SuperGrid",
    "AreaConfig",
    "SuperGridConfig",
    "AreaID",
    "TieLineConfig",
    "create_tie_lines",
    # Power Flow
    "PowerFlowRunner",
    "PowerFlowResult",
    "PowerFlowAlgorithm",
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
    "LoadFrequencyController",
    # Physics Engine (multi-agent RL entry point)
    "PhysicsEngine",
    "GridOrchestrator",  # backward compat alias
    "ScenarioConfig",
    "StochasticProfileGenerator",
    # Microgrid
    "MicrogridCase",
    "PowerFlowSolver",
    "MicrogridController",
    "SolarPV",
    "WindTurbine",
    "DieselGenerator",
    "MicrogridDERComponent",
    "create_example_microgrid",
    "create_ieee14_case",
]
