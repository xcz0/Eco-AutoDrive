"""Reward-specific time-to-collision component."""

from __future__ import annotations

import math

import numpy as np

from eco_planner.envs.domain import TransitionMetrics

from ..config import TTCRewardConfig


def ttc_score(config: TTCRewardConfig, metrics: TransitionMetrics) -> tuple[float, float, bool]:
    min_ttc_s, has_candidate = _minimum_time_to_collision_s(config, metrics)
    score = float(
        np.clip(
            (min_ttc_s - config.critical_ttc_s) / (config.safe_ttc_s - config.critical_ttc_s),
            0.0,
            1.0,
        )
    )
    return score, min_ttc_s, has_candidate


def _minimum_time_to_collision_s(
    config: TTCRewardConfig, metrics: TransitionMetrics
) -> tuple[float, bool]:
    step = metrics.input
    forward = np.asarray([math.cos(step.heading_rad), math.sin(step.heading_rad)])
    ego_position = np.asarray(step.position_xy_m, dtype=np.float64)
    ego_velocity = np.asarray(step.velocity_xy_mps, dtype=np.float64)
    objects = [
        (
            np.asarray(item.position_xy_m, dtype=np.float64),
            np.asarray(item.velocity_xy_mps, dtype=np.float64),
            item.heading_rad,
            item.width_m,
            item.length_m,
        )
        for item in step.traffic_frame.participants
    ]
    objects.extend(
        (
            np.asarray(item.position_xy_m, dtype=np.float64),
            np.zeros(2, dtype=np.float64),
            item.heading_rad,
            item.width_m,
            item.length_m,
        )
        for item in step.traffic_frame.static_objects
    )
    left = np.asarray([-forward[1], forward[0]])
    candidates: list[float] = []
    for position, velocity, heading, width_m, length_m in objects:
        relative = position - ego_position
        longitudinal = float(np.dot(relative, forward))
        if longitudinal <= 0.0:
            continue
        heading_delta = heading - step.heading_rad
        half_length = 0.5 * (
            length_m * abs(math.cos(heading_delta)) + width_m * abs(math.sin(heading_delta))
        )
        half_width = 0.5 * (
            length_m * abs(math.sin(heading_delta)) + width_m * abs(math.cos(heading_delta))
        )
        lateral_bound = step.ego_width_m / 2 + half_width + config.lateral_margin_m
        if abs(float(np.dot(relative, left))) > lateral_bound:
            continue
        closing_speed = float(np.dot(ego_velocity - velocity, forward))
        if closing_speed < config.minimum_closing_speed_mps:
            continue
        clearance = longitudinal - (
            step.ego_length_m / 2 + half_length + config.longitudinal_margin_m
        )
        candidates.append(max(0.0, clearance) / closing_speed)
    if not candidates:
        return config.maximum_ttc_s, False
    return min(min(candidates), config.maximum_ttc_s), True


__all__ = ["ttc_score"]
