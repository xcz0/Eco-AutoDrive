from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, StrictFloat


class ExpectedCalibration(BaseModel):
    """E-034 frozen calibration values used to verify the restored source batch."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)
    full_score_delta_m: StrictFloat = Field(gt=0.0)
    longitudinal_acceleration_limit_mps2: StrictFloat = Field(gt=0.0)
    lateral_acceleration_limit_mps2: StrictFloat = Field(gt=0.0)
    jerk_limit_mps3: StrictFloat = Field(gt=0.0)
    yaw_rate_limit_radps: StrictFloat = Field(gt=0.0)


class CalibrationMatchTolerance(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)
    rtol: StrictFloat = Field(ge=0.0)
    atol: StrictFloat = Field(ge=0.0)


class CalibrationTargets(Protocol):
    """Distribution target scores consumed by the fixed calibration rule."""

    progress_target_score: float
    comfort_target_score: float


class CalibrationGuardSource(Protocol):
    """Study configs carrying the E-034 frozen calibration provenance fields."""

    expected_calibration: ExpectedCalibration
    calibration_match_tolerance: CalibrationMatchTolerance
