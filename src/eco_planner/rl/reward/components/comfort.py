"""Motion-comfort reward component."""

from __future__ import annotations

import numpy as np

from eco_planner.envs.domain import TransitionMetrics

from ..config import ComfortRewardConfig


def comfort_score(config: ComfortRewardConfig, metrics: TransitionMetrics) -> float:
    return min(
        _component(
            abs(metrics.longitudinal_acceleration_mps2),
            config.longitudinal_acceleration_limit_mps2,
        ),
        _component(
            abs(metrics.lateral_acceleration_mps2), config.lateral_acceleration_limit_mps2
        ),
        _component(metrics.jerk_mps3, config.jerk_limit_mps3),
        _component(abs(metrics.input.yaw_rate_radps), config.yaw_rate_limit_radps),
    )


def _component(value: float, limit: float) -> float:
    return float(np.clip(1.0 - max(0.0, value - limit) / limit, 0.0, 1.0))


__all__ = ["comfort_score"]
