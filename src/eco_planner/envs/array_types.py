"""Array shape and dtype contracts for planner-facing environment data."""

# jaxtyping shape strings are runtime metadata, not Python forward annotations.
# ruff: noqa: F722, F821, UP037

from __future__ import annotations

from typing import TypeAlias, TypedDict

import numpy as np
import torch
from jaxtyping import Bool, Float32, Float64

TrajectoryArray: TypeAlias = Float32[np.ndarray, "80 4"]
WorldVectorArray: TypeAlias = Float64[np.ndarray, "2"]
WorldPointArray: TypeAlias = Float64[np.ndarray, "points 2"]
WorldHeadingArray: TypeAlias = Float64[np.ndarray, "points"]
WorldVelocityArray: TypeAlias = Float64[np.ndarray, "80 2"]
WorldAngularVelocityArray: TypeAlias = Float64[np.ndarray, "80"]
ExecutionStateArray: TypeAlias = Float64[np.ndarray, "execution_steps 7"]
ExecutionPointArray: TypeAlias = Float64[np.ndarray, "execution_steps 2"]
ExecutionScalarArray: TypeAlias = Float64[np.ndarray, "execution_steps"]
ExecutionBooleanArray: TypeAlias = Bool[np.ndarray, "execution_steps"]
Float64Array: TypeAlias = Float64[np.ndarray, "*shape"]
LaneGeometryArray: TypeAlias = Float64[np.ndarray, "20 2"]
EncodedLaneArray: TypeAlias = Float32[np.ndarray, "20 12"]
ParticipantRowsArray: TypeAlias = Float64[np.ndarray, "participants 8"]
ParticipantHistoryArray: TypeAlias = Float64[np.ndarray, "selected 21 8"]
EncodedParticipantHistoryArray: TypeAlias = Float32[np.ndarray, "selected 21 11"]
NeighborAgentsArray: TypeAlias = Float32[np.ndarray, "32 21 11"]
StaticObjectsArray: TypeAlias = Float32[np.ndarray, "5 10"]
StaticObjectArray: TypeAlias = Float32[np.ndarray, "10"]
EgoStateArray: TypeAlias = Float32[np.ndarray, "10"]
PlannerOnlyObservationArray: TypeAlias = Float32[np.ndarray, "1"]


class NumpyMapObservation(TypedDict):
    """Fixed official map fields before conversion to CPU tensors."""

    lanes: Float32[np.ndarray, "70 20 12"]
    lanes_speed_limit: Float32[np.ndarray, "70 1"]
    lanes_has_speed_limit: Bool[np.ndarray, "70 1"]
    route_lanes: Float32[np.ndarray, "25 20 12"]
    route_lanes_speed_limit: Float32[np.ndarray, "25 1"]
    route_lanes_has_speed_limit: Bool[np.ndarray, "25 1"]


class NumpyObservation(NumpyMapObservation):
    """One unbatched official observation represented as NumPy arrays."""

    ego_current_state: Float32[np.ndarray, "10"]
    neighbor_agents_past: Float32[np.ndarray, "32 21 11"]
    static_objects: Float32[np.ndarray, "5 10"]


class SingleObservation(TypedDict):
    """One unbatched official observation represented as CPU tensors."""

    ego_current_state: Float32[torch.Tensor, "10"]
    neighbor_agents_past: Float32[torch.Tensor, "32 21 11"]
    static_objects: Float32[torch.Tensor, "5 10"]
    lanes: Float32[torch.Tensor, "70 20 12"]
    lanes_speed_limit: Float32[torch.Tensor, "70 1"]
    lanes_has_speed_limit: Bool[torch.Tensor, "70 1"]
    route_lanes: Float32[torch.Tensor, "25 20 12"]
    route_lanes_speed_limit: Float32[torch.Tensor, "25 1"]
    route_lanes_has_speed_limit: Bool[torch.Tensor, "25 1"]


class BatchObservation(TypedDict):
    """A planner batch formed only by stacking single observations."""

    ego_current_state: Float32[torch.Tensor, "batch 10"]
    neighbor_agents_past: Float32[torch.Tensor, "batch 32 21 11"]
    static_objects: Float32[torch.Tensor, "batch 5 10"]
    lanes: Float32[torch.Tensor, "batch 70 20 12"]
    lanes_speed_limit: Float32[torch.Tensor, "batch 70 1"]
    lanes_has_speed_limit: Bool[torch.Tensor, "batch 70 1"]
    route_lanes: Float32[torch.Tensor, "batch 25 20 12"]
    route_lanes_speed_limit: Float32[torch.Tensor, "batch 25 1"]
    route_lanes_has_speed_limit: Bool[torch.Tensor, "batch 25 1"]
