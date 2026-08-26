"""Pure observation state, encoding components, and fixed planner ABI."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from eco_planner.envs.array_types import BatchObservation, SingleObservation
from eco_planner.envs.contracts import (
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
from eco_planner.envs.observation.builder import ObservationBuilder
from eco_planner.envs.observation.history import TrafficHistory
from eco_planner.envs.observation.map import MapSnapshot
from eco_planner.envs.observation.scene import TrafficObservationAudit, TrafficSceneEncoder


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


def collate_observations(observations: Sequence[SingleObservation]) -> BatchObservation:
    """Stack same-schema single-environment observations into a planner batch."""

    if not observations:
        raise ValueError("cannot collate an empty observation sequence")
    return {
        "ego_current_state": torch.stack([item["ego_current_state"] for item in observations]),
        "neighbor_agents_past": torch.stack(
            [item["neighbor_agents_past"] for item in observations]
        ),
        "static_objects": torch.stack([item["static_objects"] for item in observations]),
        "lanes": torch.stack([item["lanes"] for item in observations]),
        "lanes_speed_limit": torch.stack([item["lanes_speed_limit"] for item in observations]),
        "lanes_has_speed_limit": torch.stack(
            [item["lanes_has_speed_limit"] for item in observations]
        ),
        "route_lanes": torch.stack([item["route_lanes"] for item in observations]),
        "route_lanes_speed_limit": torch.stack(
            [item["route_lanes_speed_limit"] for item in observations]
        ),
        "route_lanes_has_speed_limit": torch.stack(
            [item["route_lanes_has_speed_limit"] for item in observations]
        ),
    }


__all__ = [
    "ObservationBuilder",
    "MapSnapshot",
    "PlannerObservationSpec",
    "TrafficHistory",
    "TrafficObservationAudit",
    "TrafficSceneEncoder",
    "collate_observations",
]
