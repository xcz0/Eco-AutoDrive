"""Reward definitions and comparisons on one persisted rollout batch."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, model_validator

from eco_planner.reward import CalibrationTargets, EnergyBandConfig


class RewardStudyConfig(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)
    lambdas: list[StrictFloat] = Field(min_length=1)
    quantiles: list[StrictFloat] = Field(min_length=2)
    calibration: CalibrationTargets
    energy_band: EnergyBandConfig | None
    representations: list[Literal["original", "calibrated", "band"]] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_axes(self) -> RewardStudyConfig:
        if self.lambdas[0] < 0 or self.lambdas != sorted(set(self.lambdas)):
            raise ValueError("lambdas must be nonnegative, sorted and unique")
        if self.quantiles[0] != 0 or self.quantiles[-1] != 1:
            raise ValueError("quantiles must span zero to one")
        if self.quantiles != sorted(set(self.quantiles)):
            raise ValueError("quantiles must be sorted and unique")
        if len(set(self.representations)) != len(self.representations):
            raise ValueError("representations must be unique")
        if "band" in self.representations and self.energy_band is None:
            raise ValueError("band representation requires explicit quantiles")
        return self
