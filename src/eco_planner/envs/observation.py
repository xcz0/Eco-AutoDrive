"""Single-environment observation collation."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from eco_planner.envs.array_types import BatchObservation, SingleObservation


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
