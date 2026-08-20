"""Validated MetaDrive traffic snapshots sampled at simulator observation time."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from metadrive.component.static_object.traffic_object import (
    TrafficBarrier,
    TrafficCone,
    TrafficObject,
    TrafficWarning,
)
from metadrive.component.traffic_participants.base_traffic_participant import (
    BaseTrafficParticipant,
)
from metadrive.component.traffic_participants.cyclist import Cyclist, CyclistBoundingBox
from metadrive.component.traffic_participants.pedestrian import (
    Pedestrian,
    PedestrianBoundingBox,
)
from metadrive.component.vehicle.base_vehicle import BaseVehicle

from eco_planner.envs.lane_speed import is_real_scalar

ParticipantKind = Literal["vehicle", "pedestrian", "bicycle"]
StaticObjectKind = Literal["barrier", "traffic_cone", "generic"]


@dataclass(frozen=True)
class TrafficParticipantState:
    """One dynamic traffic participant in world coordinates."""

    object_id: str
    kind: ParticipantKind
    position_xy_m: tuple[float, float]
    heading_rad: float
    velocity_xy_mps: tuple[float, float]
    width_m: float
    length_m: float


@dataclass(frozen=True)
class StaticTrafficObjectState:
    """One supported static traffic object in world coordinates."""

    object_id: str
    kind: StaticObjectKind
    position_xy_m: tuple[float, float]
    heading_rad: float
    width_m: float
    length_m: float


@dataclass(frozen=True)
class TrafficFrame:
    """Ego anchor and traffic objects captured after one 0.1 s simulator step."""

    simulator_step: int
    ego_center_xy_m: tuple[float, float]
    ego_heading_rad: float
    ego_rear_wheelbase_m: float
    participants: tuple[TrafficParticipantState, ...]
    static_objects: tuple[StaticTrafficObjectState, ...]


def capture_traffic_frame(env: Any) -> TrafficFrame:
    """Capture a validated, immutable traffic frame from a reset MetaDrive environment."""

    ego = getattr(env, "agent", None)
    engine = getattr(env, "engine", None)
    if ego is None or engine is None:
        raise RuntimeError("MetaDrive environment must be reset before sampling traffic")
    objects = engine.get_objects()
    if not isinstance(objects, dict):
        raise RuntimeError("MetaDrive engine objects must be exposed as a dictionary")

    rear_wheelbase = getattr(ego, "REAR_WHEELBASE", None)
    if not is_real_scalar(rear_wheelbase) or not np.isfinite(rear_wheelbase):
        raise RuntimeError("controlled vehicle must expose a finite rear wheelbase")
    if float(rear_wheelbase) <= 0.0:
        raise RuntimeError("controlled vehicle rear wheelbase must be positive")

    participants: list[TrafficParticipantState] = []
    static_objects: list[StaticTrafficObjectState] = []
    for obj in objects.values():
        participant_kind = _participant_kind(obj)
        if obj is ego:
            continue
        if participant_kind is not None:
            object_id = obj.id
            participants.append(
                TrafficParticipantState(
                    object_id=object_id,
                    kind=participant_kind,
                    position_xy_m=_finite_vector(obj.position, "position", object_id),
                    heading_rad=_finite_scalar(obj.heading_theta, "heading", object_id),
                    velocity_xy_mps=_finite_vector(obj.velocity, "velocity", object_id),
                    width_m=_positive_dimension(obj.WIDTH, "width", object_id),
                    length_m=_positive_dimension(obj.LENGTH, "length", object_id),
                )
            )
            continue

        static_kind = _static_object_kind(obj)
        if static_kind is None:
            continue
        object_id = obj.id
        static_objects.append(
            StaticTrafficObjectState(
                object_id=object_id,
                kind=static_kind,
                position_xy_m=_finite_vector(obj.position, "position", object_id),
                heading_rad=_finite_scalar(obj.heading_theta, "heading", object_id),
                width_m=_positive_dimension(obj.WIDTH, "width", object_id),
                length_m=_positive_dimension(obj.LENGTH, "length", object_id),
            )
        )

    simulator_step = getattr(engine, "episode_step", None)
    if type(simulator_step) is not int or simulator_step < 0:
        raise RuntimeError("MetaDrive engine episode_step must be a non-negative integer")
    return TrafficFrame(
        simulator_step=simulator_step,
        ego_center_xy_m=_finite_vector(ego.position, "ego position", "ego"),
        ego_heading_rad=_finite_scalar(ego.heading_theta, "ego heading", "ego"),
        ego_rear_wheelbase_m=float(rear_wheelbase),
        participants=tuple(sorted(participants, key=lambda state: state.object_id)),
        static_objects=tuple(sorted(static_objects, key=lambda state: state.object_id)),
    )


def _participant_kind(obj: object) -> ParticipantKind | None:
    if isinstance(obj, BaseVehicle):
        return "vehicle"
    if isinstance(obj, (Pedestrian, PedestrianBoundingBox)):
        return "pedestrian"
    if isinstance(obj, (Cyclist, CyclistBoundingBox)):
        return "bicycle"
    if isinstance(obj, BaseTrafficParticipant):
        raise TypeError(f"unsupported dynamic traffic participant: {type(obj).__name__}")
    return None


def _static_object_kind(obj: object) -> StaticObjectKind | None:
    if isinstance(obj, TrafficBarrier):
        return "barrier"
    if isinstance(obj, TrafficCone):
        return "traffic_cone"
    if isinstance(obj, TrafficWarning):
        return "generic"
    if isinstance(obj, TrafficObject):
        raise TypeError(f"unsupported static traffic object: {type(obj).__name__}")
    return None


def _finite_vector(value: object, field: str, object_id: str) -> tuple[float, float]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (2,) or not np.isfinite(array).all():
        raise ValueError(f"traffic object {object_id!r} {field} must be a finite 2D vector")
    return float(array[0]), float(array[1])


def _finite_scalar(value: object, field: str, object_id: str) -> float:
    if not is_real_scalar(value) or not np.isfinite(value):
        raise ValueError(f"traffic object {object_id!r} {field} must be finite and numeric")
    return float(value)


def _positive_dimension(value: object, field: str, object_id: str) -> float:
    result = _finite_scalar(value, field, object_id)
    if result <= 0.0:
        raise ValueError(f"traffic object {object_id!r} {field} must be positive")
    return result
