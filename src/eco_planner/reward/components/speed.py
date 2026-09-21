"""Speed-limit reward component."""

from __future__ import annotations

import numpy as np

from eco_planner.envs.domain import TransitionMetrics

from ..config import SpeedRewardConfig


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


__all__ = ["speed_score"]
