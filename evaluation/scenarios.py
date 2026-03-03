"""
IEEE Evaluation Scenarios for DEMS SuperGrid

Defines test scenarios for comprehensive microgrid stability and economic
evaluation following IEEE standards and best practices for power system testing.

Scenarios:
    1. Base Case:     Normal 24h diurnal operation
    2. High RE:       80%+ renewable penetration (low inertia stress)
    3. N-1 Gen:       Loss of largest generator contingency
    4. N-1 Tie:       Loss of inter-area tie-line
    5. Load Ramp:     Sudden 20% load increase (morning ramp)
    6. Islanding:     Area disconnection event
    7. Price Spike:   3× wholesale price increase
    8. Low Inertia:   Multiple synchronous generators offline
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class ScenarioDefinition:
    """Definition of an evaluation scenario.

    A scenario modifies the PhysicsEngine configuration or applies
    perturbations at specific timesteps.
    """
    name: str
    description: str
    ieee_reference: str           # Relevant IEEE standard section

    # Episode parameters (overrides to ScenarioConfig)
    episode_length_steps: int = 288   # 24h default
    dt_control_s: float = 300.0

    # Stochastic profile overrides
    solar_peak_w_m2: Optional[float] = None
    wind_mean_m_s: Optional[float] = None
    wind_std_m_s: Optional[float] = None
    load_variation_pct: Optional[float] = None

    # Perturbation events: list of (step, action_fn_name, params)
    events: List[Dict] = field(default_factory=list)

    # Evaluation criteria
    frequency_band_hz: Tuple[float, float] = (49.5, 50.5)
    voltage_band_pu: Tuple[float, float] = (0.95, 1.05)
    max_settl_time_s: float = 60.0
    min_damping_ratio: float = 0.05

    # Category for grouping in reports
    category: str = "general"


def apply_scenario_event(
    engine,
    event: Dict,
    step: int,
) -> None:
    """Apply a scenario event to the PhysicsEngine at a given step.

    Event types:
        'trip_generator': Remove a generator (gen_idx)
        'trip_tieline':   Disconnect a tie-line (line_idx)
        'load_step':      Sudden load change (area, factor)
        'price_spike':    Change pricing signal (multiplier)
        'reconnect_generator': Bring generator back online
        'reconnect_tieline':   Reconnect tie-line
    """
    event_type = event.get('type', '')
    event_step = event.get('step', 0)

    if step != event_step:
        return

    logger.info(f"Scenario event at step {step}: {event_type}")

    if event_type == 'trip_generator':
        gen_idx = event.get('gen_idx', 0)
        try:
            engine.sg.net.gen.at[gen_idx, 'in_service'] = False
            logger.info(f"  Tripped generator {gen_idx}")
        except Exception as e:
            logger.warning(f"  Failed to trip generator {gen_idx}: {e}")

    elif event_type == 'trip_tieline':
        line_idx = event.get('line_idx', 0)
        try:
            engine.sg.net.line.at[line_idx, 'in_service'] = False
            logger.info(f"  Tripped tie-line {line_idx}")
        except Exception as e:
            logger.warning(f"  Failed to trip tie-line {line_idx}: {e}")

    elif event_type == 'load_step':
        area = event.get('area', None)
        factor = event.get('factor', 1.2)
        try:
            if area is not None:
                from src.simulation.supergrid import AreaID
                engine.sg.scale_loads(AreaID(area), factor)
            else:
                # Scale all loads
                for area_id in ['A', 'B', 'C']:
                    from src.simulation.supergrid import AreaID
                    engine.sg.scale_loads(AreaID(area_id), factor)
            logger.info(f"  Applied load step: area={area}, factor={factor}")
        except Exception as e:
            logger.warning(f"  Failed to apply load step: {e}")

    elif event_type == 'reconnect_generator':
        gen_idx = event.get('gen_idx', 0)
        try:
            engine.sg.net.gen.at[gen_idx, 'in_service'] = True
            logger.info(f"  Reconnected generator {gen_idx}")
        except Exception as e:
            logger.warning(f"  Failed to reconnect generator: {e}")

    elif event_type == 'reconnect_tieline':
        line_idx = event.get('line_idx', 0)
        try:
            engine.sg.net.line.at[line_idx, 'in_service'] = True
            logger.info(f"  Reconnected tie-line {line_idx}")
        except Exception as e:
            logger.warning(f"  Failed to reconnect tie-line: {e}")


# ======================== SCENARIO DEFINITIONS ======================== #

def base_case_scenario() -> ScenarioDefinition:
    """Normal 24h diurnal operation."""
    return ScenarioDefinition(
        name="base_case",
        description="Normal 24-hour diurnal operation with standard load/RE profiles",
        ieee_reference="IEEE Std 1547-2018 §11.2 (Normal operating conditions)",
        category="normal",
    )


def high_renewable_scenario() -> ScenarioDefinition:
    """High renewable penetration (>80%) — low inertia stress test."""
    return ScenarioDefinition(
        name="high_renewable",
        description="High renewable penetration (80%+) with reduced system inertia",
        ieee_reference="IEEE Std 1547-2018 §6.5 (Frequency response), IEEE 2800-2022",
        solar_peak_w_m2=1200.0,
        wind_mean_m_s=12.0,
        wind_std_m_s=2.0,
        category="stress",
    )


def n1_generator_scenario() -> ScenarioDefinition:
    """N-1 contingency: loss of largest generator."""
    return ScenarioDefinition(
        name="n1_generator",
        description="N-1 contingency: trip of largest generator at step 48 (4h)",
        ieee_reference="NERC TPL-001-4, IEEE Std 1547-2018 §6.5.1",
        events=[
            {
                'type': 'trip_generator',
                'step': 48,   # At hour 4 (48 × 5min)
                'gen_idx': 0,  # Largest generator (nuclear, 600 MW)
            },
        ],
        category="contingency",
    )


def n1_tieline_scenario() -> ScenarioDefinition:
    """N-1 contingency: loss of inter-area tie-line."""
    return ScenarioDefinition(
        name="n1_tieline",
        description="N-1 contingency: trip of tie-line AB at step 72 (6h)",
        ieee_reference="NERC TPL-001-4, IEEE Std 1547-2018 §6.5.2",
        events=[
            {
                'type': 'trip_tieline',
                'step': 72,
                'line_idx': 0,  # First AB tie-line
            },
        ],
        category="contingency",
    )


def load_ramp_scenario() -> ScenarioDefinition:
    """Sudden 20% load increase simulating morning ramp."""
    return ScenarioDefinition(
        name="load_ramp",
        description="Sudden 20% load increase across all areas at step 36 (3h)",
        ieee_reference="IEEE Std 1547-2018 §6.5 (Load rejection/acceptance)",
        events=[
            {
                'type': 'load_step',
                'step': 36,
                'area': None,  # All areas
                'factor': 1.20,
            },
        ],
        category="stress",
    )


def islanding_scenario() -> ScenarioDefinition:
    """Area C islanding via tie-line disconnection."""
    return ScenarioDefinition(
        name="islanding",
        description="Area C islanding: trip all BC and AC tie-lines at step 60, reconnect at step 84",
        ieee_reference="IEEE Std 1547-2018 §8.2 (Islanding), §6.4.1 (Unintentional islanding)",
        events=[
            # Disconnect Area C
            {'type': 'trip_tieline', 'step': 60, 'line_idx': 3},  # BC tie 1
            {'type': 'trip_tieline', 'step': 60, 'line_idx': 4},  # BC tie 2
            {'type': 'trip_tieline', 'step': 60, 'line_idx': 5},  # BC tie 3
            {'type': 'trip_tieline', 'step': 60, 'line_idx': 6},  # AC tie 1
            {'type': 'trip_tieline', 'step': 60, 'line_idx': 7},  # AC tie 2
            # Reconnect
            {'type': 'reconnect_tieline', 'step': 84, 'line_idx': 3},
            {'type': 'reconnect_tieline', 'step': 84, 'line_idx': 4},
            {'type': 'reconnect_tieline', 'step': 84, 'line_idx': 5},
            {'type': 'reconnect_tieline', 'step': 84, 'line_idx': 6},
            {'type': 'reconnect_tieline', 'step': 84, 'line_idx': 7},
        ],
        category="contingency",
    )


def price_spike_scenario() -> ScenarioDefinition:
    """Sudden 3× wholesale price increase at peak hours."""
    return ScenarioDefinition(
        name="price_spike",
        description="3× wholesale price increase from step 108 to 156 (peak hours 9-13)",
        ieee_reference="IEEE Std 2030-2011 §5.3 (Market integration)",
        events=[
            {'type': 'price_spike', 'step': 108, 'multiplier': 3.0},
            {'type': 'price_spike', 'step': 156, 'multiplier': 1.0},  # Reset
        ],
        category="economic",
    )


def low_inertia_scenario() -> ScenarioDefinition:
    """Low system inertia: multiple synchronous generators offline."""
    return ScenarioDefinition(
        name="low_inertia",
        description="Multiple generators tripped at step 24, reducing system inertia by 40%",
        ieee_reference="IEEE Std 1547-2018 §6.5 (Low inertia systems), ENTSO-E guidance",
        events=[
            {'type': 'trip_generator', 'step': 24, 'gen_idx': 2},
            {'type': 'trip_generator', 'step': 24, 'gen_idx': 5},
            {'type': 'trip_generator', 'step': 24, 'gen_idx': 8},
        ],
        category="stress",
    )


def get_all_scenarios() -> List[ScenarioDefinition]:
    """Return all defined evaluation scenarios."""
    return [
        base_case_scenario(),
        high_renewable_scenario(),
        n1_generator_scenario(),
        n1_tieline_scenario(),
        load_ramp_scenario(),
        islanding_scenario(),
        price_spike_scenario(),
        low_inertia_scenario(),
    ]


def get_scenario_by_name(name: str) -> Optional[ScenarioDefinition]:
    """Look up a scenario by name."""
    for s in get_all_scenarios():
        if s.name == name:
            return s
    return None


def get_scenarios_by_category(category: str) -> List[ScenarioDefinition]:
    """Get all scenarios in a category."""
    return [s for s in get_all_scenarios() if s.category == category]
