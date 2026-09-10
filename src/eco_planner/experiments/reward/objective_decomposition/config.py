from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, model_validator

from ..fixed_batch.config import (
    CalibrationMatchTolerance,
    ExpectedCalibration,
)


class GateThresholds(BaseModel):
    """Issue #94 Gate C engineering thresholds; not claimed as general theory."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)
    endpoint_max_actor_head_cosine: StrictFloat = Field(gt=-1.0, lt=1.0)
    min_normalized_advantage_rmse: StrictFloat = Field(ge=0.0)
    min_sign_flip_fraction: StrictFloat = Field(ge=0.0, le=1.0)
    min_stress_fraction_of_endpoint_separation: StrictFloat = Field(gt=0.0, le=1.0)


class EnergyBandConfig(BaseModel):
    """Issue #94 Task E: calibrated efficiency-band energy representation.

    Band thresholds are intensity quantiles of the fixed source batch; the expected
    values are the frozen pre-fixed thresholds guarded against batch drift. The
    quantile bounds keep the full-score threshold below the median and the zero-score
    threshold above it.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)
    full_score_intensity_quantile: StrictFloat = Field(gt=0.0, lt=0.5)
    zero_score_intensity_quantile: StrictFloat = Field(gt=0.5, lt=1.0)
    expected_full_score_ml_per_km: StrictFloat = Field(gt=0.0)
    expected_zero_score_ml_per_km: StrictFloat = Field(gt=0.0)
    match_tolerance: CalibrationMatchTolerance


class DecompositionConfig(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)
    lambdas: list[StrictFloat] = Field(min_length=1)
    quantiles: list[StrictFloat] = Field(min_length=2)
    progress_target_score: StrictFloat = Field(gt=0.0, lt=1.0)
    comfort_target_score: StrictFloat = Field(gt=0.0, lt=1.0)
    calibration_match_tolerance: CalibrationMatchTolerance
    expected_calibration: ExpectedCalibration
    energy_band: EnergyBandConfig | None = None
    gate: GateThresholds

    @model_validator(mode="after")
    def validate_axes(self) -> DecompositionConfig:
        if any(weight <= 0 for weight in self.lambdas) or sorted(set(self.lambdas)) != self.lambdas:
            raise ValueError("stress lambdas must be positive and strictly increasing")
        if (
            self.quantiles[0] != 0
            or self.quantiles[-1] != 1
            or sorted(set(self.quantiles)) != self.quantiles
        ):
            raise ValueError("quantiles must increase from zero to one")
        return self
