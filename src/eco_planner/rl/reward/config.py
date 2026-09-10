"""Strict configuration for RL reward objectives."""

from __future__ import annotations

import math
from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, model_validator


class _StrictRewardModel(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)


class RewardWeightsConfig(_StrictRewardModel):
    ttc: StrictFloat = Field(gt=0.0)
    progress: StrictFloat = Field(gt=0.0)
    comfort: StrictFloat = Field(gt=0.0)
    speed: StrictFloat = Field(gt=0.0)
    energy: StrictFloat = Field(gt=0.0)

    @property
    def total(self) -> float:
        return self.ttc + self.progress + self.comfort + self.speed + self.energy


class NoEnergyRewardWeightsConfig(_StrictRewardModel):
    ttc: StrictFloat = Field(gt=0.0)
    progress: StrictFloat = Field(gt=0.0)
    comfort: StrictFloat = Field(gt=0.0)
    speed: StrictFloat = Field(gt=0.0)

    @property
    def total(self) -> float:
        return self.ttc + self.progress + self.comfort + self.speed


class RewardGatesConfig(_StrictRewardModel):
    collision_vehicle: StrictBool
    collision_object: StrictBool
    collision_building: StrictBool
    collision_human: StrictBool
    collision_sidewalk: StrictBool
    wrong_direction_max_heading_error_rad: StrictFloat = Field(gt=0.0, le=math.pi)


class TTCRewardConfig(_StrictRewardModel):
    critical_ttc_s: StrictFloat = Field(ge=0.0)
    safe_ttc_s: StrictFloat = Field(gt=0.0)
    maximum_ttc_s: StrictFloat = Field(gt=0.0)
    minimum_closing_speed_mps: StrictFloat = Field(gt=0.0)
    lateral_margin_m: StrictFloat = Field(ge=0.0)
    longitudinal_margin_m: StrictFloat = Field(ge=0.0)

    @model_validator(mode="after")
    def validate_ttc_bounds(self) -> TTCRewardConfig:
        if self.safe_ttc_s <= self.critical_ttc_s:
            raise ValueError("ttc.safe_ttc_s must exceed critical_ttc_s")
        if self.maximum_ttc_s < self.safe_ttc_s:
            raise ValueError("ttc.maximum_ttc_s must cover safe_ttc_s")
        return self


class ProgressRewardConfig(_StrictRewardModel):
    full_score_delta_m: StrictFloat = Field(gt=0.0)


class ComfortRewardConfig(_StrictRewardModel):
    longitudinal_acceleration_limit_mps2: StrictFloat = Field(gt=0.0)
    lateral_acceleration_limit_mps2: StrictFloat = Field(gt=0.0)
    jerk_limit_mps3: StrictFloat = Field(gt=0.0)
    yaw_rate_limit_radps: StrictFloat = Field(gt=0.0)


class SpeedRewardConfig(_StrictRewardModel):
    overspeed_margin_mps: StrictFloat = Field(ge=0.0)
    zero_score_overspeed_mps: StrictFloat = Field(gt=0.0)

    @model_validator(mode="after")
    def validate_speed_bounds(self) -> SpeedRewardConfig:
        if self.zero_score_overspeed_mps <= self.overspeed_margin_mps:
            raise ValueError("speed.zero_score_overspeed_mps must exceed overspeed_margin_mps")
        return self


class EnergyRewardConfig(_StrictRewardModel):
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


class PlannerRFTEnergyRewardConfig(_StrictRewardModel):
    name: Literal["plannerrft_energy_v1"]
    weights: RewardWeightsConfig
    gates: RewardGatesConfig
    ttc: TTCRewardConfig
    progress: ProgressRewardConfig
    comfort: ComfortRewardConfig
    speed: SpeedRewardConfig
    energy: EnergyRewardConfig


class PlannerRFTNoEnergyRewardConfig(_StrictRewardModel):
    """No-energy R0 objective; `energy` only normalizes the audited diagnostic score."""

    name: Literal["plannerrft_no_energy_v1"]
    weights: NoEnergyRewardWeightsConfig
    gates: RewardGatesConfig
    ttc: TTCRewardConfig
    progress: ProgressRewardConfig
    comfort: ComfortRewardConfig
    speed: SpeedRewardConfig
    energy: EnergyRewardConfig


RewardProfileConfig: TypeAlias = PlannerRFTEnergyRewardConfig | PlannerRFTNoEnergyRewardConfig


__all__ = [
    "ComfortRewardConfig",
    "EnergyRewardConfig",
    "NoEnergyRewardWeightsConfig",
    "PlannerRFTEnergyRewardConfig",
    "PlannerRFTNoEnergyRewardConfig",
    "ProgressRewardConfig",
    "RewardGatesConfig",
    "RewardProfileConfig",
    "RewardWeightsConfig",
    "SpeedRewardConfig",
    "TTCRewardConfig",
]
