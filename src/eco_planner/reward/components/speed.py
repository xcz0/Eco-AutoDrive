"""Speed-limit reward component."""

from __future__ import annotations

import numpy as np
from pydantic import Field, StrictFloat, model_validator

from eco_planner.envs.domain import TransitionMetrics

from .strict import StrictRewardModel


class SpeedRewardConfig(StrictRewardModel):
    overspeed_margin_mps: StrictFloat = Field(ge=0.0)
    zero_score_overspeed_mps: StrictFloat = Field(gt=0.0)

    @model_validator(mode="after")
    def validate_speed_bounds(self) -> SpeedRewardConfig:
        if self.zero_score_overspeed_mps <= self.overspeed_margin_mps:
            raise ValueError("speed.zero_score_overspeed_mps must exceed overspeed_margin_mps")
        return self


def speed_score(config: SpeedRewardConfig, metrics: TransitionMetrics) -> tuple[float, float]:
    overspeed_mps = max(0.0, metrics.speed_mps - metrics.input.speed_limit_mps)
    score = float(
        np.clip(
            1.0
            - max(0.0, overspeed_mps - config.overspeed_margin_mps)
            / (config.zero_score_overspeed_mps - config.overspeed_margin_mps),
            0.0,
            1.0,
        )
    )
    return score, overspeed_mps


__all__ = ["SpeedRewardConfig", "speed_score"]
