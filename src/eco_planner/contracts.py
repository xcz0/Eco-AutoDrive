"""Canonical shared planner and simulator timing/shape contracts."""

from __future__ import annotations

from typing import Final

import numpy as np

PLANNER_HORIZON: Final = 80
PLANNER_ACTOR_COUNT: Final = 11
PLANNER_STATE_DIM: Final = 4
SIMULATOR_STEP_S: Final = 0.1
METADRIVE_PHYSICS_STEP_S: Final = 0.02
METADRIVE_DECISION_REPEAT: Final = 5
CLOSED_LOOP_EXECUTION_STEPS: Final = 5
DECISION_INTERVAL_S: Final = CLOSED_LOOP_EXECUTION_STEPS * SIMULATOR_STEP_S
TRAFFIC_HISTORY_FRAMES: Final = 21
TRAFFIC_HISTORY_WARMUP_STEPS: Final = TRAFFIC_HISTORY_FRAMES - 1

AGENT_HISTORY_DIM: Final = 11
AGENT_COUNT: Final = 32
STATIC_OBJECT_DIM: Final = 10
STATIC_OBJECT_COUNT: Final = 5
LANE_POINTS: Final = 20
LANE_FEATURE_DIM: Final = 12
LANE_COUNT: Final = 70
ROUTE_LANE_COUNT: Final = 25


def evaluation_plan_cycles(evaluated_horizon_steps: int) -> int:
    """Return the number of evaluation prefixes needed for a horizon."""

    if type(evaluated_horizon_steps) is not int or evaluated_horizon_steps <= 0:
        raise ValueError("evaluated horizon steps must be a positive integer")
    cycles = evaluated_horizon_steps + CLOSED_LOOP_EXECUTION_STEPS - 1
    return cycles // CLOSED_LOOP_EXECUTION_STEPS


def validate_metadrive_timestep(physics_step_s: float | int, decision_repeat: int) -> float:
    """Validate that a MetaDrive decision advances exactly one planner timestep."""

    if type(physics_step_s) not in {int, float} or physics_step_s <= 0.0:
        raise ValueError("physics_world_step_size must be positive")
    if type(decision_repeat) is not int or decision_repeat <= 0:
        raise ValueError("decision_repeat must be a positive integer")
    timestep_s = float(physics_step_s) * decision_repeat
    if not np.isclose(timestep_s, SIMULATOR_STEP_S, rtol=0.0, atol=1e-12):
        raise ValueError(
            f"physics_world_step_size * decision_repeat must equal {SIMULATOR_STEP_S} seconds"
        )
    return timestep_s
