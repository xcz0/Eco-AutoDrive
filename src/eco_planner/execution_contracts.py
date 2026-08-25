"""Shared planner and simulator timing/shape contracts."""

from __future__ import annotations

from typing import Final

PLANNER_FUTURE_STEPS: Final = 80
EVALUATION_EXECUTION_STEPS: Final = 5
ROLLOUT_EXECUTION_STEPS: Final = 1
SIMULATOR_STEP_S: Final = 0.1
TRAFFIC_HISTORY_STEPS: Final = 20


def evaluation_plan_cycles(evaluated_horizon_steps: int) -> int:
    """Return the number of five-step evaluation prefixes needed for a horizon."""

    if type(evaluated_horizon_steps) is not int or evaluated_horizon_steps <= 0:
        raise ValueError("evaluated horizon steps must be a positive integer")
    return (evaluated_horizon_steps + EVALUATION_EXECUTION_STEPS - 1) // EVALUATION_EXECUTION_STEPS
