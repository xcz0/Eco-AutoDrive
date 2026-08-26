"""Fixed planner and simulator contracts owned by the environment boundary."""

from __future__ import annotations

from enum import Enum
from typing import Final

PLANNER_HORIZON: Final = 80
SIMULATOR_STEP_S: Final = 0.1
METADRIVE_PHYSICS_STEP_S: Final = 0.02
METADRIVE_DECISION_REPEAT: Final = 5
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


class ExecutionMode(str, Enum):
    """The only trajectory-prefix modes supported by the project."""

    ROLLOUT = "rollout"
    EVALUATION = "evaluation"

    @property
    def steps(self) -> int:
        """Return the fixed number of 10 Hz points executed for this mode."""

        return 1 if self is ExecutionMode.ROLLOUT else 5
