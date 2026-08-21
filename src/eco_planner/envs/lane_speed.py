"""Programmatic lane speed-limit contracts."""

from __future__ import annotations

from typing import Any

import numpy as np

from eco_planner.envs.validation import is_real_scalar

PROGRAMMATIC_SPEED_LIMIT_SENTINEL_KMH = 1000.0
MAX_LANE_SPEED_LIMIT_KMH = 130.0


def validated_programmatic_speed_limit_kmh(value: object) -> float:
    if not is_real_scalar(value):
        raise TypeError("programmatic_lane_speed_limit_kmh must be a numeric km/h value")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0 or result > MAX_LANE_SPEED_LIMIT_KMH:
        raise ValueError(
            "programmatic_lane_speed_limit_kmh must be finite, positive, and no greater than "
            "130 km/h"
        )
    return result


def model_lane_speed_limit_mps(lane: Any) -> tuple[float, bool]:
    """Return the raw model value and mask while rejecting unconfigured sentinels."""

    speed_limit = getattr(lane, "speed_limit", None)
    if speed_limit is None:
        return 0.0, False
    if not is_real_scalar(speed_limit):
        raise TypeError(f"lane {lane.index!r} speed limit must be numeric or None")
    speed_limit_kmh = float(speed_limit)
    if not np.isfinite(speed_limit_kmh) or speed_limit_kmh < 0.0:
        raise ValueError(f"lane {lane.index!r} speed limit must be finite and non-negative")
    if speed_limit_kmh == 0.0:
        return 0.0, False
    if speed_limit_kmh == PROGRAMMATIC_SPEED_LIMIT_SENTINEL_KMH:
        raise RuntimeError(
            f"lane {lane.index!r} has raw speed limit {speed_limit!r} km/h: "
            "programmatic lane speed limit was not configured"
        )
    if speed_limit_kmh > MAX_LANE_SPEED_LIMIT_KMH:
        raise ValueError(
            f"lane {lane.index!r} speed limit {speed_limit!r} km/h exceeds "
            "the 130 km/h domain bound"
        )
    return speed_limit_kmh / 3.6, True
