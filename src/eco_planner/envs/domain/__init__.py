"""Lazy façade for framework-independent environment domain contracts."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .energy import (
        EnergyMetricName,
        EnergyMetricProvider,
        EnergyMetrics,
        EnergyTrace,
        MetaDriveFuelProxyProvider,
    )
    from .execution import TrajectoryExecutionRecord, TrajectoryExecutionResult
    from .fastsim import FASTSimEnergyConfig, FASTSimEnergyProvider
    from .traffic import (
        ParticipantKind,
        StaticObjectKind,
        StaticTrafficObjectState,
        TrafficFrame,
        TrafficParticipantState,
    )
    from .trajectory import WorldTrajectory, stationary_trajectory, to_world_trajectory
    from .transition import (
        STOPPED_SPEED_THRESHOLD_MPS,
        WRONG_DIRECTION_MAX_HEADING_ERROR_RAD,
        TransitionMetricInput,
        TransitionMetrics,
        any_collision,
        derive_transition_metrics,
    )

_EXPORTS = {
    "any_collision": (".transition", "any_collision"),
    "STOPPED_SPEED_THRESHOLD_MPS": (".transition", "STOPPED_SPEED_THRESHOLD_MPS"),
    "WRONG_DIRECTION_MAX_HEADING_ERROR_RAD": (
        ".transition",
        "WRONG_DIRECTION_MAX_HEADING_ERROR_RAD",
    ),
    "EnergyMetricName": (".energy", "EnergyMetricName"),
    "EnergyMetricProvider": (".energy", "EnergyMetricProvider"),
    "EnergyMetrics": (".energy", "EnergyMetrics"),
    "EnergyTrace": (".energy", "EnergyTrace"),
    "MetaDriveFuelProxyProvider": (".energy", "MetaDriveFuelProxyProvider"),
    "FASTSimEnergyConfig": (".fastsim", "FASTSimEnergyConfig"),
    "FASTSimEnergyProvider": (".fastsim", "FASTSimEnergyProvider"),
    "TrajectoryExecutionRecord": (".execution", "TrajectoryExecutionRecord"),
    "TrajectoryExecutionResult": (".execution", "TrajectoryExecutionResult"),
    "ParticipantKind": (".traffic", "ParticipantKind"),
    "StaticObjectKind": (".traffic", "StaticObjectKind"),
    "StaticTrafficObjectState": (".traffic", "StaticTrafficObjectState"),
    "TrafficFrame": (".traffic", "TrafficFrame"),
    "TrafficParticipantState": (".traffic", "TrafficParticipantState"),
    "WorldTrajectory": (".trajectory", "WorldTrajectory"),
    "stationary_trajectory": (".trajectory", "stationary_trajectory"),
    "to_world_trajectory": (".trajectory", "to_world_trajectory"),
    "TransitionMetricInput": (".transition", "TransitionMetricInput"),
    "TransitionMetrics": (".transition", "TransitionMetrics"),
    "derive_transition_metrics": (".transition", "derive_transition_metrics"),
}

__all__ = [
    "EnergyMetricName",
    "EnergyMetricProvider",
    "EnergyMetrics",
    "EnergyTrace",
    "FASTSimEnergyConfig",
    "FASTSimEnergyProvider",
    "MetaDriveFuelProxyProvider",
    "ParticipantKind",
    "StaticObjectKind",
    "StaticTrafficObjectState",
    "TrafficFrame",
    "TrafficParticipantState",
    "TrajectoryExecutionRecord",
    "TrajectoryExecutionResult",
    "TransitionMetricInput",
    "TransitionMetrics",
    "WorldTrajectory",
    "derive_transition_metrics",
    "stationary_trajectory",
    "to_world_trajectory",
    "any_collision",
    "STOPPED_SPEED_THRESHOLD_MPS",
    "WRONG_DIRECTION_MAX_HEADING_ERROR_RAD",
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
