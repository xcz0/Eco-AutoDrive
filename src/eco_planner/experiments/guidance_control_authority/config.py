from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, model_validator


class InterventionConfig(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)
    protocol: str
    runtime_seed: StrictInt = Field(ge=0)
    noise_seeds: list[StrictInt]
    longitudinal_actions: list[StrictFloat]
    lateral_action: StrictFloat
    window_steps: StrictInt = Field(ge=10, le=30)
    spearman_threshold: StrictFloat = Field(gt=0, le=1)
    noise_multiplier: StrictFloat = Field(gt=0)
    required_scenarios: StrictInt = Field(gt=0)
    overrides: list[str]

    @model_validator(mode="after")
    def validate_protocol(self) -> InterventionConfig:
        if len(self.noise_seeds) < 2 or len(set(self.noise_seeds)) != len(self.noise_seeds):
            raise ValueError("at least two distinct noise seeds are required")
        if min(self.noise_seeds) < 0:
            raise ValueError("noise seeds must be non-negative")
        if self.longitudinal_actions != [-1.0, -0.5, 0.0, 0.5, 1.0]:
            raise ValueError("Task D requires the five fixed longitudinal endpoints/interior arms")
        if self.lateral_action != 0.0:
            raise ValueError("Task D fixes lateral guidance to zero")
        return self
