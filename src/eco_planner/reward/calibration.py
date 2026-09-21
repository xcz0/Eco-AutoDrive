"""Pure reward calibration over objective-neutral measurement arrays.

Calibration consumes measured quantities and a resolved profile and returns a
transformed profile or scalar parameters. It never reads rollout episodes or RL
TensorDicts; the RL adapter owns measurement extraction and audit writeback.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictFloat

from .components.comfort import component_score
from .components.progress import score_delta
from .config import PlannerRFTNoEnergyRewardConfig

MOTION_LIMITS = {
    "longitudinal_acceleration_mps2": "longitudinal_acceleration_limit_mps2",
    "lateral_acceleration_mps2": "lateral_acceleration_limit_mps2",
    "jerk_mps3": "jerk_limit_mps3",
    "yaw_rate_radps": "yaw_rate_limit_radps",
}


class CalibrationTargets(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)
    progress_target_score: StrictFloat = Field(gt=0.0, lt=1.0)
    comfort_target_score: StrictFloat = Field(gt=0.0, lt=1.0)


class EnergyBandConfig(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)
    full_score_intensity_quantile: StrictFloat = Field(gt=0.0, lt=0.5)
    zero_score_intensity_quantile: StrictFloat = Field(gt=0.5, lt=1.0)


def scored_arrays(
    raw: Mapping[str, np.ndarray], profile: PlannerRFTNoEnergyRewardConfig
) -> dict[str, np.ndarray]:
    """Score raw motion/progress measurements with the profile's component mappings."""

    result = {
        "progress": np.asarray(
            [
                score_delta(x, profile.progress.full_score_delta_m)
                for x in raw["route_progress_delta_m"]
            ]
        )
    }
    for key, field in MOTION_LIMITS.items():
        result[key] = np.asarray(
            [component_score(abs(x), getattr(profile.comfort, field)) for x in raw[key]]
        )
    result["comfort"] = np.min([result[key] for key in MOTION_LIMITS], axis=0)
    return result


def calibrate(
    raw: Mapping[str, np.ndarray],
    base: PlannerRFTNoEnergyRewardConfig,
    study: CalibrationTargets,
) -> PlannerRFTNoEnergyRewardConfig:
    """Derive Progress/Comfort scales from this batch's measurement distribution."""

    for value in raw.values():
        if value.size == 0 or not np.isfinite(value).all():
            raise ValueError("calibration requires nonempty finite motion arrays")
    positive = raw["route_progress_delta_m"][raw["route_progress_delta_m"] > 0]
    if not positive.size:
        raise ValueError("progress calibration requires positive route progress")
    payload = base.model_dump()
    payload["progress"]["full_score_delta_m"] = float(
        np.median(positive) / study.progress_target_score
    )
    old_scores = scored_arrays(raw, base)
    for key, field in MOTION_LIMITS.items():
        if np.any(old_scores[key] == 0):
            payload["comfort"][field] = max(
                getattr(base.comfort, field),
                float(np.median(np.abs(raw[key])) / (2 - study.comfort_target_score)),
            )
    return PlannerRFTNoEnergyRewardConfig.model_validate(payload)


def energy_band_thresholds(intensity: np.ndarray, band: EnergyBandConfig) -> tuple[float, float]:
    """Derive the calibrated-band full/zero intensity thresholds from this batch."""

    full = float(np.quantile(intensity, band.full_score_intensity_quantile))
    zero = float(np.quantile(intensity, band.zero_score_intensity_quantile))
    return full, zero


__all__ = [
    "MOTION_LIMITS",
    "CalibrationTargets",
    "EnergyBandConfig",
    "calibrate",
    "energy_band_thresholds",
    "scored_arrays",
]
