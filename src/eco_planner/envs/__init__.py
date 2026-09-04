"""Lazy planner-facing environment façade."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .domain import (
        MetaDriveFuelProxyProvider,
        StaticTrafficObjectState,
        TrafficFrame,
        TrafficParticipantState,
        TrajectoryExecutionRecord,
        TrajectoryExecutionResult,
        TransitionMetricInput,
        TransitionMetrics,
        WorldTrajectory,
        derive_transition_metrics,
    )
    from .metadrive import (
        EnvSlotReset,
        EnvSlotState,
        EnvSlotStep,
        EnvSlotTiming,
        MetaDriveBackend,
        MetaDriveEnvSlot,
        ObservationMode,
    )
    from .observation import TrafficObservationAudit

_EXPORTS = {
    "MetaDriveFuelProxyProvider": (".domain", "MetaDriveFuelProxyProvider"),
    "StaticTrafficObjectState": (".domain", "StaticTrafficObjectState"),
    "TrafficFrame": (".domain", "TrafficFrame"),
    "TrafficParticipantState": (".domain", "TrafficParticipantState"),
    "TrajectoryExecutionRecord": (".domain", "TrajectoryExecutionRecord"),
    "TrajectoryExecutionResult": (".domain", "TrajectoryExecutionResult"),
    "TransitionMetricInput": (".domain", "TransitionMetricInput"),
    "TransitionMetrics": (".domain", "TransitionMetrics"),
    "WorldTrajectory": (".domain", "WorldTrajectory"),
    "derive_transition_metrics": (".domain", "derive_transition_metrics"),
    "EnvSlotReset": (".metadrive", "EnvSlotReset"),
    "EnvSlotState": (".metadrive", "EnvSlotState"),
    "EnvSlotStep": (".metadrive", "EnvSlotStep"),
    "EnvSlotTiming": (".metadrive", "EnvSlotTiming"),
    "MetaDriveBackend": (".metadrive", "MetaDriveBackend"),
    "MetaDriveEnvSlot": (".metadrive", "MetaDriveEnvSlot"),
    "ObservationMode": (".metadrive", "ObservationMode"),
    "TrafficObservationAudit": (".observation", "TrafficObservationAudit"),
}

__all__ = [
    "EnvSlotReset",
    "EnvSlotState",
    "EnvSlotStep",
    "EnvSlotTiming",
    "MetaDriveBackend",
    "MetaDriveEnvSlot",
    "MetaDriveFuelProxyProvider",
    "ObservationMode",
    "StaticTrafficObjectState",
    "TrafficFrame",
    "TrafficObservationAudit",
    "TrafficParticipantState",
    "TrajectoryExecutionRecord",
    "TrajectoryExecutionResult",
    "TransitionMetricInput",
    "TransitionMetrics",
    "WorldTrajectory",
    "derive_transition_metrics",
]


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
