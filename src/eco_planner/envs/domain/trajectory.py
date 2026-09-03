"""Pure planner-trajectory conversion utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..array_types import (
    TrajectoryArray,
    WorldAngularVelocityArray,
    WorldHeadingArray,
    WorldPointArray,
    WorldVectorArray,
    WorldVelocityArray,
)
from ..geometry import (
    local_points_to_world,
    rear_axle_position,
    shortest_angle_delta,
)


@dataclass(frozen=True, slots=True)
class WorldTrajectory:
    centers: WorldPointArray
    headings: WorldHeadingArray
    velocities: WorldVelocityArray
    angular_velocities: WorldAngularVelocityArray


def to_world_trajectory(
    trajectory: TrajectoryArray,
    *,
    center_position: WorldVectorArray,
    center_heading: float,
    rear_wheelbase: float,
    timestep_s: float,
) -> WorldTrajectory:
    """Convert the fixed ego-local rear-axle ABI to world-frame vehicle targets."""

    anchor_rear_axle = rear_axle_position(
        center_position.astype(np.float64), center_heading, rear_wheelbase
    )
    future_rear_axles = local_points_to_world(
        trajectory[:, :2].astype(np.float64), anchor_rear_axle, center_heading
    )
    relative_headings = np.arctan2(trajectory[:, 3], trajectory[:, 2]).astype(np.float64)
    future_headings = center_heading + relative_headings
    future_directions = np.column_stack((np.cos(future_headings), np.sin(future_headings)))
    future_centers = future_rear_axles + rear_wheelbase * future_directions
    centers = np.vstack((center_position.astype(np.float64), future_centers))
    headings = np.concatenate(([center_heading], future_headings))
    return WorldTrajectory(
        centers=centers,
        headings=headings,
        velocities=np.diff(centers, axis=0) / timestep_s,
        angular_velocities=shortest_angle_delta(np.diff(headings)) / timestep_s,
    )
