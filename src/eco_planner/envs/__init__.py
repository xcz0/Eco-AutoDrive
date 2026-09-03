"""Planner-facing MetaDrive interfaces."""

# ruff: noqa: I001

# Torch must load before MetaDrive/Panda3D on Windows to avoid DLL initialization failures.
from .domain import (
    TrajectoryExecutionRecord,
    TrajectoryExecutionResult,
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
    MetaDriveEnvSlot,
    ObservationMode,
    MetaDriveBackend,
)

__all__ = [
    "PlannerObservationSpec",
    "TrafficObservationAudit",
    "TrajectoryExecutionRecord",
    "TrajectoryExecutionResult",
    "WorldTrajectory",
    "EnvSlotObservation",
    "EnvSlotReset",
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
