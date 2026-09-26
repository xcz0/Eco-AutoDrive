"""Strict configuration for RL reward objectives."""

from __future__ import annotations

import re
from typing import Literal, TypeAlias

from pydantic import Field, StrictFloat, model_validator

from .components import (
    ComfortRewardConfig,
    EnergyRewardConfig,
    ProgressRewardConfig,
    RewardGatesConfig,
    SpeedRewardConfig,
    StrictRewardModel,
    TTCRewardConfig,
)


class RewardWeightsConfig(StrictRewardModel):
    ttc: StrictFloat = Field(gt=0.0)
    progress: StrictFloat = Field(gt=0.0)
    comfort: StrictFloat = Field(gt=0.0)
    speed: StrictFloat = Field(gt=0.0)
    energy: StrictFloat = Field(gt=0.0)

    @property
    def total(self) -> float:
        return self.ttc + self.progress + self.comfort + self.speed + self.energy


class NoEnergyRewardWeightsConfig(StrictRewardModel):
    ttc: StrictFloat = Field(gt=0.0)
    progress: StrictFloat = Field(gt=0.0)
    comfort: StrictFloat = Field(gt=0.0)
    speed: StrictFloat = Field(gt=0.0)

    @property
    def total(self) -> float:
        return self.ttc + self.progress + self.comfort + self.speed


class PlannerRFTEnergyRewardConfig(StrictRewardModel):
    name: Literal[
        "plannerrft_energy_v1",
        "plannerrft_energy_band_lam1_v1",
        "plannerrft_energy_band_lam2_v1",
        "plannerrft_energy_band_lam4_v1",
        "plannerrft_energy_band_lam8_v1",
        "plannerrft_energy_band_lam64_v1",
    ]
    weights: RewardWeightsConfig
    gates: RewardGatesConfig
    ttc: TTCRewardConfig
    progress: ProgressRewardConfig
    comfort: ComfortRewardConfig
    speed: SpeedRewardConfig
    energy: EnergyRewardConfig

    @model_validator(mode="after")
    def validate_band_lambda_name(self) -> PlannerRFTEnergyRewardConfig:
        """A named band profile must carry exactly the lambda encoded in its name."""

        match = re.fullmatch(r"plannerrft_energy_band_lam(\d+)_v1", self.name)
        if match is not None and self.weights.energy != float(match.group(1)):
            raise ValueError(
                f"reward profile {self.name} requires weights.energy={match.group(1)}, "
                f"got {self.weights.energy}"
            )
        return self


class PlannerRFTNoEnergyRewardConfig(StrictRewardModel):
    """No-energy R0 objective; `energy` only normalizes the audited diagnostic score."""

    name: Literal["plannerrft_no_energy_v1", "plannerrft_no_energy_calibrated_v1"]
    weights: NoEnergyRewardWeightsConfig
    gates: RewardGatesConfig
    ttc: TTCRewardConfig
    progress: ProgressRewardConfig
    comfort: ComfortRewardConfig
    speed: SpeedRewardConfig
    energy: EnergyRewardConfig


RewardProfileConfig: TypeAlias = PlannerRFTEnergyRewardConfig | PlannerRFTNoEnergyRewardConfig


__all__ = [
    "PlannerRFTEnergyRewardConfig",
    "PlannerRFTNoEnergyRewardConfig",
    "RewardProfileConfig",
]
