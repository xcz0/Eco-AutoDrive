"""Safety and legality gate for RL reward objectives."""

from __future__ import annotations

from pydantic import StrictBool

from eco_planner.envs.domain import TransitionMetrics

from .strict import StrictRewardModel


class RewardGatesConfig(StrictRewardModel):
    collision_vehicle: StrictBool
    collision_object: StrictBool
    collision_building: StrictBool
    collision_human: StrictBool
    collision_sidewalk: StrictBool


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
    wrong_direction_score = float(not metrics.wrong_direction)
    return (
        collision_score * drivable_score * wrong_direction_score,
        collision_score,
        drivable_score,
        wrong_direction_score,
    )


__all__ = ["RewardGatesConfig", "safety_gate"]
