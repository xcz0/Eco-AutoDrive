"""PlannerRFT-style gated objective over reusable reward components."""

from __future__ import annotations

import math
from dataclasses import fields

from eco_planner.envs.domain import TransitionMetrics

from ..components import (
    comfort_score,
    energy_score,
    progress_score,
    safety_gate,
    speed_score,
    ttc_score,
)
from ..config import PlannerRFTEnergyRewardConfig
from ..result import RewardComponents, RewardDiagnostics, RewardResult


def evaluate_plannerrft_energy_step(
    config: PlannerRFTEnergyRewardConfig,
    metrics: TransitionMetrics,
) -> RewardResult:
    """Evaluate one transition without accessing simulator or runtime objects."""

    gate, collision, drivable, wrong_direction = safety_gate(config.gates, metrics)
    ttc, min_ttc_s, has_ttc_candidate = ttc_score(config.ttc, metrics)
    progress = progress_score(config.progress, metrics)
    comfort = comfort_score(config.comfort, metrics)
    speed, overspeed_mps = speed_score(config.speed, metrics)
    energy, fuel_ml_per_km, energy_distance_valid = energy_score(config.energy, metrics)
    components = RewardComponents(ttc, progress, comfort, speed, energy)
    weights = config.weights
    base_total = (
        weights.ttc * components.ttc
        + weights.progress * components.progress
        + weights.comfort * components.comfort
        + weights.speed * components.speed
        + weights.energy * components.energy
    ) / weights.total
    step = metrics.input
    fuel_ml = metrics.energy.fuel_ml
    if fuel_ml is None:
        raise RuntimeError("energy component accepted a missing fuel-volume metric")
    result = RewardResult(
        profile_name="plannerrft_energy_v1",
        total=gate * base_total,
        base_total=base_total,
        safety_gate=gate,
        components=components,
        diagnostics=RewardDiagnostics(
            collision_score=collision,
            drivable_score=drivable,
            wrong_direction_score=wrong_direction,
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
            executed_fuel_proxy_step_energy_ml=fuel_ml,
            executed_fuel_proxy_ml_per_km=fuel_ml_per_km,
            energy_distance_valid=energy_distance_valid,
        ),
    )
    values = [result.total, result.base_total, result.safety_gate]
    values.extend(
        float(getattr(result.components, item.name)) for item in fields(result.components)
    )
    values.extend(
        float(value)
        for item in fields(result.diagnostics)
        if not isinstance((value := getattr(result.diagnostics, item.name)), bool)
    )
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("PlannerRFT energy reward produced a non-finite result")
    return result


__all__ = ["evaluate_plannerrft_energy_step"]
