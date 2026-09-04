"""Array contracts for the fixed planner observation ABI."""

# jaxtyping shape strings are runtime metadata, not Python forward annotations.
# ruff: noqa: F722, F821, UP037

from __future__ import annotations

from typing import TypeAlias, TypedDict

import numpy as np
from jaxtyping import Bool, Float32, Float64

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

LaneGeometryArray: TypeAlias = Float64[np.ndarray, f"{LANE_POINTS} 2"]
EncodedLaneArray: TypeAlias = Float32[np.ndarray, f"{LANE_POINTS} {LANE_FEATURE_DIM}"]
NeighborAgentsArray: TypeAlias = Float32[
    np.ndarray, f"{AGENT_COUNT} {TRAFFIC_HISTORY_FRAMES} {AGENT_HISTORY_DIM}"
]
StaticObjectsArray: TypeAlias = Float32[np.ndarray, f"{STATIC_OBJECT_COUNT} {STATIC_OBJECT_DIM}"]
PlannerOnlyObservationArray: TypeAlias = Float32[np.ndarray, "1"]


class MapObservationArrays(TypedDict):
    """Fixed official map fields before conversion to CPU tensors."""

    lanes: Float32[np.ndarray, f"{LANE_COUNT} {LANE_POINTS} {LANE_FEATURE_DIM}"]
    lanes_speed_limit: Float32[np.ndarray, f"{LANE_COUNT} 1"]
    lanes_has_speed_limit: Bool[np.ndarray, f"{LANE_COUNT} 1"]
    route_lanes: Float32[np.ndarray, f"{ROUTE_LANE_COUNT} {LANE_POINTS} {LANE_FEATURE_DIM}"]
    route_lanes_speed_limit: Float32[np.ndarray, f"{ROUTE_LANE_COUNT} 1"]
    route_lanes_has_speed_limit: Bool[np.ndarray, f"{ROUTE_LANE_COUNT} 1"]
