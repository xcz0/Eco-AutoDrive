"""Energy-quality reward component."""

from __future__ import annotations

import math

from eco_planner.envs.domain import TransitionMetrics

from ..config import EnergyRewardConfig


def energy_score(
    config: EnergyRewardConfig, metrics: TransitionMetrics
) -> tuple[float, float, bool]:
    if metrics.energy.fuel_ml is None:
        raise ValueError("PlannerRFT energy reward requires a fuel-volume metric")
    distance_valid = metrics.step_distance_m >= config.minimum_step_distance_m
    measured_ml_per_km = metrics.energy.fuel_ml_per_km
    fuel_ml_per_km = (
        measured_ml_per_km if distance_valid and measured_ml_per_km is not None else 0.0
    )
    score = (
        math.exp(-fuel_ml_per_km / config.reference_ml_per_km) if distance_valid else 0.0
    )
    return score, fuel_ml_per_km, distance_valid


__all__ = ["energy_score"]
