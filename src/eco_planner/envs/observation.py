"""Single-environment observation collation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import torch

from eco_planner.envs.array_types import BatchObservation, SingleObservation


class _PlannerObservationConfig(Protocol):
    time_len: int
    agent_state_dim: int
    agent_num: int
    static_objects_state_dim: int
    static_objects_num: int
    lane_len: int
    lane_state_dim: int
    lane_num: int
    route_len: int
    route_state_dim: int
    route_num: int


@dataclass(frozen=True, slots=True)
class PlannerObservationSpec:
    """Planner observation dimensions consumed by the MetaDrive adapters."""

    time_len: int
    agent_state_dim: int
    agent_num: int
    static_objects_state_dim: int
    static_objects_num: int
    lane_len: int
    lane_state_dim: int
    lane_num: int
    route_len: int
    route_state_dim: int
    route_num: int

    @classmethod
    def from_planner_config(
        cls, config: _PlannerObservationConfig
    ) -> PlannerObservationSpec:
        """Copy only the observation dimensions from a planner configuration."""

        return cls(
            time_len=config.time_len,
            agent_state_dim=config.agent_state_dim,
            agent_num=config.agent_num,
            static_objects_state_dim=config.static_objects_state_dim,
            static_objects_num=config.static_objects_num,
            lane_len=config.lane_len,
            lane_state_dim=config.lane_state_dim,
            lane_num=config.lane_num,
            route_len=config.route_len,
            route_state_dim=config.route_state_dim,
            route_num=config.route_num,
        )


def collate_observations(
    observations: Sequence[SingleObservation],
) -> BatchObservation:
    """Stack same-schema single-environment observations into a planner batch."""

    if not observations:
        raise ValueError("cannot collate an empty observation sequence")
    return {
        "ego_current_state": torch.stack(
            [observation["ego_current_state"] for observation in observations]
        ),
        "neighbor_agents_past": torch.stack(
            [observation["neighbor_agents_past"] for observation in observations]
        ),
        "static_objects": torch.stack(
            [observation["static_objects"] for observation in observations]
        ),
        "lanes": torch.stack([observation["lanes"] for observation in observations]),
        "lanes_speed_limit": torch.stack(
            [observation["lanes_speed_limit"] for observation in observations]
        ),
        "lanes_has_speed_limit": torch.stack(
            [observation["lanes_has_speed_limit"] for observation in observations]
        ),
        "route_lanes": torch.stack([observation["route_lanes"] for observation in observations]),
        "route_lanes_speed_limit": torch.stack(
            [observation["route_lanes_speed_limit"] for observation in observations]
        ),
        "route_lanes_has_speed_limit": torch.stack(
            [observation["route_lanes_has_speed_limit"] for observation in observations]
        ),
    }
