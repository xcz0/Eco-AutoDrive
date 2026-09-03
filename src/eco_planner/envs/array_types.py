"""Array shape and dtype contracts for planner-facing environment data."""

# jaxtyping shape strings are runtime metadata, not Python forward annotations.
# ruff: noqa: F722, F821, UP037

from __future__ import annotations

from typing import TypeAlias, TypedDict

import numpy as np
from jaxtyping import Bool, Float32, Float64

from eco_planner.contracts import PLANNER_HORIZON, TRAFFIC_HISTORY_FRAMES

TrajectoryArray: TypeAlias = Float32[np.ndarray, f"{PLANNER_HORIZON} 4"]
WorldVectorArray: TypeAlias = Float64[np.ndarray, "2"]
WorldPointArray: TypeAlias = Float64[np.ndarray, "points 2"]
WorldHeadingArray: TypeAlias = Float64[np.ndarray, "points"]
WorldVelocityArray: TypeAlias = Float64[np.ndarray, f"{PLANNER_HORIZON} 2"]
WorldAngularVelocityArray: TypeAlias = Float64[np.ndarray, f"{PLANNER_HORIZON}"]
ExecutionStateArray: TypeAlias = Float64[np.ndarray, "execution_steps 7"]
ExecutionPointArray: TypeAlias = Float64[np.ndarray, "execution_steps 2"]
ExecutionScalarArray: TypeAlias = Float64[np.ndarray, "execution_steps"]
ExecutionBooleanArray: TypeAlias = Bool[np.ndarray, "execution_steps"]
Float64Array: TypeAlias = Float64[np.ndarray, "*shape"]
LaneGeometryArray: TypeAlias = Float64[np.ndarray, "20 2"]
EncodedLaneArray: TypeAlias = Float32[np.ndarray, "20 12"]
NeighborAgentsArray: TypeAlias = Float32[np.ndarray, f"32 {TRAFFIC_HISTORY_FRAMES} 11"]
StaticObjectsArray: TypeAlias = Float32[np.ndarray, "5 10"]
PlannerOnlyObservationArray: TypeAlias = Float32[np.ndarray, "1"]


class NumpyMapObservation(TypedDict):
    """Fixed official map fields before conversion to CPU tensors."""

    lanes: Float32[np.ndarray, "70 20 12"]
    lanes_speed_limit: Float32[np.ndarray, "70 1"]
    lanes_has_speed_limit: Bool[np.ndarray, "70 1"]
    route_lanes: Float32[np.ndarray, "25 20 12"]
    route_lanes_speed_limit: Float32[np.ndarray, "25 1"]
    route_lanes_has_speed_limit: Bool[np.ndarray, "25 1"]
