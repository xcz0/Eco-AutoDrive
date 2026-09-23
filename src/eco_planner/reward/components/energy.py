"""Energy-quality reward component."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import Field, StrictFloat, model_validator

from eco_planner.envs.domain import TransitionMetrics

from .strict import StrictRewardModel


class EnergyRewardConfig(StrictRewardModel):
    mode: Literal["reference_exponential", "calibrated_band"] = "reference_exponential"
    reference_ml_per_km: StrictFloat = Field(gt=0.0)
    minimum_step_distance_m: StrictFloat = Field(gt=0.0)
    band_full_score_ml_per_km: StrictFloat | None = None
    band_zero_score_ml_per_km: StrictFloat | None = None

    @model_validator(mode="after")
    def validate_energy_band(self) -> EnergyRewardConfig:
        if self.mode == "calibrated_band":
            if self.band_full_score_ml_per_km is None or self.band_zero_score_ml_per_km is None:
                raise ValueError(
                    "calibrated_band energy mode requires band_full_score_ml_per_km "
                    "and band_zero_score_ml_per_km"
                )
            if self.band_zero_score_ml_per_km <= self.band_full_score_ml_per_km:
                raise ValueError("energy band requires zero_score above full_score")
        elif (
            self.band_full_score_ml_per_km is not None or self.band_zero_score_ml_per_km is not None
        ):
            raise ValueError("energy band thresholds are only allowed in calibrated_band mode")
        return self


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


__all__ = [
    "EnergyRewardConfig",
    "calibrated_band_score",
    "energy_score",
    "energy_score_from_fuel",
]
