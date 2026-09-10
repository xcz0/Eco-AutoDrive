from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, model_validator

from eco_planner.experiments.fixed_batch.config import (
    CalibrationMatchTolerance,
    ExpectedCalibration,
)


class AttributionThresholds(BaseModel):
    """Issue #94 Gate C endpoint thresholds reused by the C4 attribution rules."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)
    endpoint_max_actor_head_cosine: StrictFloat = Field(gt=-1.0, lt=1.0)
    min_normalized_advantage_rmse: StrictFloat = Field(ge=0.0)
    min_sign_flip_fraction: StrictFloat = Field(ge=0.0, le=1.0)


class AblationConfig(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)
    quantiles: list[StrictFloat] = Field(min_length=2)
    progress_target_score: StrictFloat = Field(gt=0.0, lt=1.0)
    comfort_target_score: StrictFloat = Field(gt=0.0, lt=1.0)
    calibration_match_tolerance: CalibrationMatchTolerance
    expected_calibration: ExpectedCalibration
    reference_match_tolerance: CalibrationMatchTolerance
    gate: AttributionThresholds

    @model_validator(mode="after")
    def validate_axes(self) -> AblationConfig:
        if (
            self.quantiles[0] != 0
            or self.quantiles[-1] != 1
            or sorted(set(self.quantiles)) != self.quantiles
        ):
            raise ValueError("quantiles must increase from zero to one")
        return self
