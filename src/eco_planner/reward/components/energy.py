"""Energy-quality reward component."""

from __future__ import annotations

import math

from eco_planner.envs.domain import TransitionMetrics

from ..config import EnergyRewardConfig


def calibrated_band_score(
    intensity_ml_per_km: float,
    full_score_ml_per_km: float,
    zero_score_ml_per_km: float,
) -> float:
    """Two-sided saturating efficiency-band score in [0, 1]; lower intensity is better."""
    score = (zero_score_ml_per_km - intensity_ml_per_km) / (
        zero_score_ml_per_km - full_score_ml_per_km
    )
    return max(0.0, min(1.0, score))


def energy_score_from_fuel(
    config: EnergyRewardConfig, fuel_ml: float, step_distance_m: float
) -> tuple[float, float, bool]:
    """Score one executed step from its fuel volume and distance.

    Pure so that online per-substep evaluation and offline sub-step rescoring
    share one definition of validity, intensity, and band/exponential score.
    """

    distance_valid = step_distance_m >= config.minimum_step_distance_m
    fuel_ml_per_km = fuel_ml * 1_000.0 / step_distance_m if distance_valid else 0.0
    if not distance_valid:
        score = 0.0
    elif config.mode == "calibrated_band":
        full = config.band_full_score_ml_per_km
        zero = config.band_zero_score_ml_per_km
        assert full is not None and zero is not None
        score = calibrated_band_score(fuel_ml_per_km, full, zero)
    else:
        score = math.exp(-fuel_ml_per_km / config.reference_ml_per_km)
    return score, fuel_ml_per_km, distance_valid


def energy_score(
    config: EnergyRewardConfig, metrics: TransitionMetrics
) -> tuple[float, float, bool]:
    if metrics.energy.fuel_ml is None:
        raise ValueError("PlannerRFT energy reward requires a fuel-volume metric")
    return energy_score_from_fuel(config, metrics.energy.fuel_ml, metrics.step_distance_m)


__all__ = ["calibrated_band_score", "energy_score", "energy_score_from_fuel"]
