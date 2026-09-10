from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, model_validator


class CalibrationConfig(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)
    progress_target_score: StrictFloat = Field(gt=0.0, lt=1.0)
    comfort_target_score: StrictFloat = Field(gt=0.0, lt=1.0)
    lambdas: list[StrictFloat] = Field(min_length=2)
    quantiles: list[StrictFloat] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_axes(self) -> CalibrationConfig:
        if self.lambdas[0] != 0 or sorted(set(self.lambdas)) != self.lambdas:
            raise ValueError("lambdas must start at zero and strictly increase")
        if (
            self.quantiles[0] != 0
            or self.quantiles[-1] != 1
            or sorted(set(self.quantiles)) != self.quantiles
        ):
            raise ValueError("quantiles must increase from zero to one")
        return self
