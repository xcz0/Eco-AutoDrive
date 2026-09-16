from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, StrictFloat


class CalibrationTargets(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)
    progress_target_score: StrictFloat = Field(gt=0.0, lt=1.0)
    comfort_target_score: StrictFloat = Field(gt=0.0, lt=1.0)


class EnergyBandConfig(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)
    full_score_intensity_quantile: StrictFloat = Field(gt=0.0, lt=0.5)
    zero_score_intensity_quantile: StrictFloat = Field(gt=0.5, lt=1.0)
