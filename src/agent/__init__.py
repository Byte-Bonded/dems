"""
DEMS Hierarchical Multi-Agent PPO System.

Architecture:
    Central PPO Agent (1×)
        ├── MG PPO Agent A (1×)
        │   ├── Inverter Sub-Agent A
        │   ├── Renewable Sub-Agent A
        │   └── Load Sub-Agent A
        ├── MG PPO Agent B (1×)
        │   ├── Inverter Sub-Agent B
        │   ├── Renewable Sub-Agent B
        │   └── Load Sub-Agent B
        └── MG PPO Agent C (1×)
            ├── Inverter Sub-Agent C
            ├── Renewable Sub-Agent C
            └── Load Sub-Agent C

Modules:
    constraints   - Grid constraint validation (IEGC, IEEE)
    observations  - Observation builders per hierarchy level
    rewards       - Reward functions per hierarchy level
    environments  - Gymnasium environments per hierarchy level
    coordinator   - MultiAgentStepCoordinator
    ppo_agents    - SB3 PPO wrappers (Central, MG, Sub)
    training      - CTDE trainer
    metrics       - Episode logging
"""

from .coordinator import MultiAgentStepCoordinator
from .ppo_agents import (
    PPOAgentWrapper,
    CentralPPOAgent,
    MicrogridPPOAgent,
    SubPPOAgent,
    detect_device,
)
from .training import HierarchicalTrainer, TrainingConfig
from .metrics import EpisodeLogger

from .constraints import (
    GridConstraintValidator,
    ConstraintViolation,
    ViolationType,
)
from .observations import (
    MicrogridObsBuilder,
    CentralObsBuilder,
    SubAgentObsBuilder,
    MG_OBS_DIM,
    CENTRAL_OBS_DIM,
)
from .rewards import (
    MicrogridReward,
    CentralReward,
    InverterSubReward,
    RenewableSubReward,
    LoadSubReward,
)
from .environments import (
    MicrogridEnv,
    CentralCoordEnv,
    InverterSubEnv,
    RenewableSubEnv,
    LoadSubEnv,
)

__all__ = [
    # Coordinator
    "MultiAgentStepCoordinator",
    # Agents
    "PPOAgentWrapper",
    "CentralPPOAgent",
    "MicrogridPPOAgent",
    "SubPPOAgent",
    # Training
    "HierarchicalTrainer",
    "TrainingConfig",
    # Metrics
    "EpisodeLogger",
    # Constraints
    "GridConstraintValidator",
    "ConstraintViolation",
    "ViolationType",
    # Observations
    "MicrogridObsBuilder",
    "CentralObsBuilder",
    "SubAgentObsBuilder",
    "MG_OBS_DIM",
    "CENTRAL_OBS_DIM",
    # Rewards
    "MicrogridReward",
    "CentralReward",
    "InverterSubReward",
    "RenewableSubReward",
    "LoadSubReward",
    # Environments
    "MicrogridEnv",
    "CentralCoordEnv",
    "InverterSubEnv",
    "RenewableSubEnv",
    "LoadSubEnv",
]
