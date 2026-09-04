"""MetaDrive object traversal at the traffic-snapshot trust boundary."""

from __future__ import annotations

from typing import Protocol, cast

import numpy as np
from metadrive.component.static_object.traffic_object import (
    TrafficBarrier,
    TrafficCone,
    TrafficObject,
    TrafficWarning,
)
from metadrive.component.traffic_participants.base_traffic_participant import BaseTrafficParticipant
from metadrive.component.traffic_participants.cyclist import Cyclist, CyclistBoundingBox
from metadrive.component.traffic_participants.pedestrian import Pedestrian, PedestrianBoundingBox
from metadrive.component.vehicle.base_vehicle import BaseVehicle

from ..domain.traffic import (
    ParticipantKind,
    StaticObjectKind,
    StaticTrafficObjectState,
    TrafficFrame,
    TrafficParticipantState,
)


class _TrafficEngine(Protocol):
    episode_step: int

    def get_objects(self) -> dict[str, object]: ...


class _TrafficEgo(Protocol):
    REAR_WHEELBASE: float
    position: object
    heading_theta: float


class _TrafficEnvironment(Protocol):
    @property
    def agent(self) -> _TrafficEgo: ...

    @property
    def engine(self) -> _TrafficEngine: ...


def capture_traffic_frame(env: _TrafficEnvironment) -> TrafficFrame:
    """Capture and validate one immutable domain frame from reset MetaDrive state."""

    ego = env.agent
    engine = env.engine
    rear_wheelbase = _positive_scalar(ego.REAR_WHEELBASE, "rear wheelbase", "ego")
    participants: list[TrafficParticipantState] = []
    static_objects: list[StaticTrafficObjectState] = []
    for obj in engine.get_objects().values():
        if obj is ego:
            continue
        participant_kind = _participant_kind(obj)
        if participant_kind is not None:
            value = cast(BaseTrafficParticipant, obj)
            object_id = _object_id(value.id)
            participants.append(
                TrafficParticipantState(
                    object_id=object_id,
                    kind=participant_kind,
                    position_xy_m=_finite_vector(value.position, "position", object_id),
                    heading_rad=_finite_scalar(value.heading_theta, "heading", object_id),
                    velocity_xy_mps=_finite_vector(value.velocity, "velocity", object_id),
                    width_m=_positive_scalar(value.WIDTH, "width", object_id),
                    length_m=_positive_scalar(value.LENGTH, "length", object_id),
                )
            )
            continue
        static_kind = _static_object_kind(obj)
        if static_kind is not None:
            value = cast(TrafficObject, obj)
            object_id = _object_id(value.id)
            static_objects.append(
                StaticTrafficObjectState(
                    object_id=object_id,
                    kind=static_kind,
                    position_xy_m=_finite_vector(value.position, "position", object_id),
                    heading_rad=_finite_scalar(value.heading_theta, "heading", object_id),
                    width_m=_positive_scalar(value.WIDTH, "width", object_id),
                    length_m=_positive_scalar(value.LENGTH, "length", object_id),
                )
            )
    if type(engine.episode_step) is not int or engine.episode_step < 0:
        raise RuntimeError("MetaDrive engine episode_step must be a non-negative integer")
    return TrafficFrame(
        simulator_step=engine.episode_step,
        ego_center_xy_m=_finite_vector(ego.position, "ego position", "ego"),
        ego_heading_rad=_finite_scalar(ego.heading_theta, "ego heading", "ego"),
        ego_rear_wheelbase_m=rear_wheelbase,
        participants=tuple(sorted(participants, key=lambda item: item.object_id)),
        static_objects=tuple(sorted(static_objects, key=lambda item: item.object_id)),
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


def _object_id(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError("MetaDrive traffic object id must be a non-empty string")
    return value


def _finite_scalar(value: object, field: str, object_id: str) -> float:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != ():
        raise ValueError(f"traffic object {object_id!r} {field} must be a scalar")
    result = float(array.item())
    if not np.isfinite(result):
        raise ValueError(f"traffic object {object_id!r} {field} must be finite")
    return result


def _positive_scalar(value: object, field: str, object_id: str) -> float:
    result = _finite_scalar(value, field, object_id)
    if result <= 0.0:
        raise ValueError(f"traffic object {object_id!r} {field} must be positive")
    return result
