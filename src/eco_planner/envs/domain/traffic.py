"""Immutable traffic DTOs independent of simulator implementation details."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ParticipantKind = Literal["vehicle", "pedestrian", "bicycle"]
StaticObjectKind = Literal["barrier", "traffic_cone", "generic"]


@dataclass(frozen=True, slots=True)
class TrafficParticipantState:
    object_id: str
    kind: ParticipantKind
    position_xy_m: tuple[float, float]
    heading_rad: float
    velocity_xy_mps: tuple[float, float]
    width_m: float
    length_m: float


@dataclass(frozen=True, slots=True)
class StaticTrafficObjectState:
    object_id: str
    kind: StaticObjectKind
    position_xy_m: tuple[float, float]
    heading_rad: float
    width_m: float
    length_m: float


@dataclass(frozen=True, slots=True)
class TrafficFrame:
    """One immutable 10 Hz scene snapshot in world coordinates."""

    simulator_step: int
    ego_center_xy_m: tuple[float, float]
    ego_heading_rad: float
    ego_rear_wheelbase_m: float
    participants: tuple[TrafficParticipantState, ...]
    static_objects: tuple[StaticTrafficObjectState, ...]
