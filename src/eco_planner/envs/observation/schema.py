"""Canonical field schema for the fixed planner observation ABI."""

from __future__ import annotations

import numpy as np

from eco_planner.contracts import (
    AGENT_COUNT,
    AGENT_HISTORY_DIM,
    LANE_COUNT,
    LANE_FEATURE_DIM,
    LANE_POINTS,
    ROUTE_LANE_COUNT,
    STATIC_OBJECT_COUNT,
    STATIC_OBJECT_DIM,
    TRAFFIC_HISTORY_FRAMES,
)

PLANNER_OBSERVATION_FIELDS: dict[str, tuple[tuple[int, ...], np.dtype]] = {
    "ego_current_state": ((10,), np.dtype(np.float32)),
    "neighbor_agents_past": (
        (AGENT_COUNT, TRAFFIC_HISTORY_FRAMES, AGENT_HISTORY_DIM),
        np.dtype(np.float32),
    ),
    "static_objects": ((STATIC_OBJECT_COUNT, STATIC_OBJECT_DIM), np.dtype(np.float32)),
    "lanes": ((LANE_COUNT, LANE_POINTS, LANE_FEATURE_DIM), np.dtype(np.float32)),
    "lanes_speed_limit": ((LANE_COUNT, 1), np.dtype(np.float32)),
    "lanes_has_speed_limit": ((LANE_COUNT, 1), np.dtype(np.bool_)),
    "route_lanes": (
        (ROUTE_LANE_COUNT, LANE_POINTS, LANE_FEATURE_DIM),
        np.dtype(np.float32),
    ),
    "route_lanes_speed_limit": ((ROUTE_LANE_COUNT, 1), np.dtype(np.float32)),
    "route_lanes_has_speed_limit": ((ROUTE_LANE_COUNT, 1), np.dtype(np.bool_)),
}
