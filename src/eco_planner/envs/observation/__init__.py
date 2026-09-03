"""Pure observation state, encoding components, and fixed planner ABI."""

from __future__ import annotations

from eco_planner.contracts import (
    AGENT_COUNT,
    AGENT_HISTORY_DIM,
    LANE_COUNT,
    LANE_FEATURE_DIM,
    LANE_POINTS,
    ROUTE_LANE_COUNT,
    STATIC_OBJECT_COUNT,
    STATIC_OBJECT_DIM,
    TRAFFIC_HISTORY_FRAMES,
)

from .builder import ObservationBuilder
from .history import TrafficHistory
from .map import MapSnapshot
from .scene import TrafficObservationAudit, TrafficSceneEncoder


class PlannerObservationSpec:
    """Compatibility view of the fixed official planner observation ABI."""

    __slots__ = ()
    _VALUES = (
        TRAFFIC_HISTORY_FRAMES,
        AGENT_HISTORY_DIM,
        AGENT_COUNT,
        STATIC_OBJECT_DIM,
        STATIC_OBJECT_COUNT,
        LANE_POINTS,
        LANE_FEATURE_DIM,
        LANE_COUNT,
        LANE_POINTS,
        LANE_FEATURE_DIM,
        ROUTE_LANE_COUNT,
    )

    def __init__(self, *values: int) -> None:
        if values and values != self._VALUES:
            raise ValueError("planner observation dimensions are fixed by the project ABI")

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PlannerObservationSpec)

    @property
    def time_len(self) -> int:
        return TRAFFIC_HISTORY_FRAMES

    @property
    def agent_state_dim(self) -> int:
        return AGENT_HISTORY_DIM

    @property
    def agent_num(self) -> int:
        return AGENT_COUNT

    @property
    def static_objects_state_dim(self) -> int:
        return STATIC_OBJECT_DIM

    @property
    def static_objects_num(self) -> int:
        return STATIC_OBJECT_COUNT

    @property
    def lane_len(self) -> int:
        return LANE_POINTS

    @property
    def lane_state_dim(self) -> int:
        return LANE_FEATURE_DIM

    @property
    def lane_num(self) -> int:
        return LANE_COUNT

    @property
    def route_len(self) -> int:
        return LANE_POINTS

    @property
    def route_state_dim(self) -> int:
        return LANE_FEATURE_DIM

    @property
    def route_num(self) -> int:
        return ROUTE_LANE_COUNT

    @classmethod
    def from_planner_config(cls, config: object) -> PlannerObservationSpec:
        values = tuple(
            getattr(config, name)
            for name in (
                "time_len",
                "agent_state_dim",
                "agent_num",
                "static_objects_state_dim",
                "static_objects_num",
                "lane_len",
                "lane_state_dim",
                "lane_num",
                "route_len",
                "route_state_dim",
                "route_num",
            )
        )
        return cls(*values)


__all__ = [
    "ObservationBuilder",
    "MapSnapshot",
    "PlannerObservationSpec",
    "TrafficHistory",
    "TrafficObservationAudit",
    "TrafficSceneEncoder",
]
