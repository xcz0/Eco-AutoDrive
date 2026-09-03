"""PlannerRFT-style reward objective over objective-neutral transition metrics."""

from __future__ import annotations

import math
from dataclasses import fields

import numpy as np

from eco_planner.envs.domain.metrics import (
    TransitionMetrics,
    minimum_time_to_collision_s,
)
from eco_planner.rl.reward.audit import PlannerRFTEnergyRewardAudit
from eco_planner.rl.reward.config import PlannerRFTEnergyRewardConfig


def score_plannerrft_energy_step(
    config: PlannerRFTEnergyRewardConfig,
    metrics: TransitionMetrics,
) -> PlannerRFTEnergyRewardAudit:
    """Score one executed transition without accessing simulator objects."""

    step = metrics.input
    collision = any(
        (
            config.gates.collision_vehicle and step.crash_vehicle,
            config.gates.collision_object and step.crash_object,
            config.gates.collision_building and step.crash_building,
            config.gates.collision_human and step.crash_human,
            config.gates.collision_sidewalk and step.crash_sidewalk,
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
    wrong_direction_score = float(
        heading_error <= config.gates.wrong_direction_max_heading_error_rad
    )
    reward_gate = collision_score * drivable_score * wrong_direction_score
    min_ttc_s, has_ttc_candidate = minimum_time_to_collision_s(
        metrics,
        maximum_ttc_s=config.ttc.maximum_ttc_s,
        minimum_closing_speed_mps=config.ttc.minimum_closing_speed_mps,
        lateral_margin_m=config.ttc.lateral_margin_m,
        longitudinal_margin_m=config.ttc.longitudinal_margin_m,
    )
    ttc_score = _linear_score(min_ttc_s, config.ttc.critical_ttc_s, config.ttc.safe_ttc_s)
    progress_score = float(
        np.clip(
            max(0.0, step.route_progress_delta_m) / config.progress.full_score_delta_m,
            0.0,
            1.0,
        )
    )
    comfort_score = min(
        _comfort_component(
            abs(metrics.longitudinal_acceleration_mps2),
            config.comfort.longitudinal_acceleration_limit_mps2,
        ),
        _comfort_component(
            abs(metrics.lateral_acceleration_mps2), config.comfort.lateral_acceleration_limit_mps2
        ),
        _comfort_component(metrics.jerk_mps3, config.comfort.jerk_limit_mps3),
        _comfort_component(abs(step.yaw_rate_radps), config.comfort.yaw_rate_limit_radps),
    )
    overspeed_mps = max(0.0, metrics.speed_mps - step.speed_limit_mps)
    speed_score = float(
        np.clip(
            1.0
            - max(0.0, overspeed_mps - config.speed.overspeed_margin_mps)
            / (config.speed.zero_score_overspeed_mps - config.speed.overspeed_margin_mps),
            0.0,
            1.0,
        )
    )
    energy_distance_valid = metrics.step_distance_m >= config.energy.minimum_step_distance_m
    measured_fuel_ml_per_km = metrics.energy.fuel_ml_per_km
    if metrics.energy.fuel_ml is None:
        raise ValueError("PlannerRFT energy reward requires a fuel-volume metric")
    fuel_proxy_ml_per_km = (
        measured_fuel_ml_per_km
        if energy_distance_valid and measured_fuel_ml_per_km is not None
        else 0.0
    )
    energy_score = (
        math.exp(-fuel_proxy_ml_per_km / config.energy.reference_ml_per_km)
        if energy_distance_valid
        else 0.0
    )
    weights = config.weights
    reward_ungated = (
        weights.ttc * ttc_score
        + weights.progress * progress_score
        + weights.comfort * comfort_score
        + weights.speed * speed_score
        + weights.energy * energy_score
    ) / weights.total
    output = PlannerRFTEnergyRewardAudit(
        profile_name="plannerrft_energy_v1",
        reward_total=reward_gate * reward_ungated,
        reward_ungated=reward_ungated,
        reward_gate=reward_gate,
        collision_score=collision_score,
        drivable_score=drivable_score,
        wrong_direction_score=wrong_direction_score,
        ttc_score=ttc_score,
        progress_score=progress_score,
        comfort_score=comfort_score,
        speed_score=speed_score,
        energy_score=energy_score,
        has_ttc_candidate=has_ttc_candidate,
        min_ttc_s=min_ttc_s,
        route_progress_delta_m=step.route_progress_delta_m,
        speed_mps=metrics.speed_mps,
        speed_limit_mps=step.speed_limit_mps,
        overspeed_mps=overspeed_mps,
        longitudinal_acceleration_mps2=metrics.longitudinal_acceleration_mps2,
        lateral_acceleration_mps2=metrics.lateral_acceleration_mps2,
        jerk_mps3=metrics.jerk_mps3,
        yaw_rate_radps=step.yaw_rate_radps,
        step_distance_m=metrics.step_distance_m,
        native_step_energy_ml=step.native_step_energy_ml,
        native_episode_energy_ml=step.native_episode_energy_ml,
        executed_fuel_proxy_step_energy_ml=metrics.energy.fuel_ml,
        executed_fuel_proxy_ml_per_km=fuel_proxy_ml_per_km,
        energy_distance_valid=energy_distance_valid,
    )
    values = [
        value for item in fields(output) if isinstance((value := getattr(output, item.name)), float)
    ]
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("PlannerRFT energy reward produced a non-finite audit value")
    return output


def _linear_score(value: float, lower: float, upper: float) -> float:
    return float(np.clip((value - lower) / (upper - lower), 0.0, 1.0))


def _comfort_component(value: float, limit: float) -> float:
    return float(np.clip(1.0 - max(0.0, value - limit) / limit, 0.0, 1.0))


__all__ = ["score_plannerrft_energy_step"]
