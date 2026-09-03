"""Planner-facing MetaDrive interfaces."""

# ruff: noqa: I001

# Torch must load before MetaDrive/Panda3D on Windows to avoid DLL initialization failures.
from .domain import (
    TrajectoryExecutionRecord,
    WorldTrajectory,
    MetaDriveFuelProxyProvider,
    StaticTrafficObjectState,
    TrafficFrame,
    TrafficParticipantState,
    TransitionMetricInput,
    derive_transition_metrics,
    TransitionMetrics,
)
from .observation import (
    PlannerObservationSpec,
    TrafficObservationAudit,
)
from .metadrive import (
    EnvSlotObservation,
    EnvSlotReset,
    EnvSlotStep,
    MetaDriveEnvSlot,
    ObservationMode,
    MetaDriveBackend,
)

__all__ = [
    "PlannerObservationSpec",
    "TrafficObservationAudit",
    "TrajectoryExecutionRecord",
    "WorldTrajectory",
    "EnvSlotObservation",
    "EnvSlotReset",
    "EnvSlotStep",
    "MetaDriveEnvSlot",
    "ObservationMode",
    "MetaDriveBackend",
    "MetaDriveFuelProxyProvider",
    "StaticTrafficObjectState",
    "TrafficFrame",
    "TrafficParticipantState",
    "TransitionMetricInput",
    "derive_transition_metrics",
    "TransitionMetrics",
]
