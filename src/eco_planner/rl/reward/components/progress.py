"""Route-progress reward component."""

from __future__ import annotations

import numpy as np

from eco_planner.envs.domain import TransitionMetrics

from ..config import ProgressRewardConfig


def progress_score(config: ProgressRewardConfig, metrics: TransitionMetrics) -> float:
    return score_delta(metrics.input.route_progress_delta_m, config.full_score_delta_m)


def score_delta(delta_m: float, full_score_delta_m: float) -> float:
    return float(
        np.clip(
            max(0.0, delta_m) / full_score_delta_m,
            0.0,
            1.0,
        )
    )


__all__ = ["progress_score"]
