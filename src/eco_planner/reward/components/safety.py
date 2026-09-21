"""Safety and legality gate for RL reward objectives."""

from __future__ import annotations

import math

from eco_planner.envs.domain import TransitionMetrics

from ..config import RewardGatesConfig


def safety_gate(
    config: RewardGatesConfig, metrics: TransitionMetrics
) -> tuple[float, float, float, float]:
    step = metrics.input
    collision = any(
        (
            config.collision_vehicle and step.crash_vehicle,
            config.collision_object and step.crash_object,
            config.collision_building and step.crash_building,
            config.collision_human and step.crash_human,
            config.collision_sidewalk and step.crash_sidewalk,
        )
    )
    collision_score = float(not collision)
    drivable_score = float(not step.out_of_road)
    heading_error = abs(
        math.atan2(
            math.sin(step.heading_rad - step.route_heading_rad),
            math.cos(step.heading_rad - step.route_heading_rad),
        )
    )
    wrong_direction_score = float(heading_error <= config.wrong_direction_max_heading_error_rad)
    return (
        collision_score * drivable_score * wrong_direction_score,
        collision_score,
        drivable_score,
        wrong_direction_score,
    )


__all__ = ["safety_gate"]
