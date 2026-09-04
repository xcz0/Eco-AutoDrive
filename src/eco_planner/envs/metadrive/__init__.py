"""Lazy façade for MetaDrive environment integration."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

# Torch must load before MetaDrive/Panda3D on Windows to avoid DLL initialization failures.
import torch as _torch

del _torch

if TYPE_CHECKING:
    from .map import LocalRouteUnavailableError
    from .policy import KinematicTrajectoryPolicy
    from .simulator import MetaDriveBackend
    from .slot import (
        EnvSlotReset,
        EnvSlotState,
        EnvSlotStep,
        EnvSlotTiming,
        MetaDriveEnvSlot,
        ObservationMode,
    )
    from .snapshot import capture_traffic_frame

_EXPORTS = {
    "EnvSlotReset": (".slot", "EnvSlotReset"),
    "EnvSlotState": (".slot", "EnvSlotState"),
    "EnvSlotStep": (".slot", "EnvSlotStep"),
    "EnvSlotTiming": (".slot", "EnvSlotTiming"),
    "KinematicTrajectoryPolicy": (".policy", "KinematicTrajectoryPolicy"),
    "LocalRouteUnavailableError": (".map", "LocalRouteUnavailableError"),
    "MetaDriveBackend": (".simulator", "MetaDriveBackend"),
    "MetaDriveEnvSlot": (".slot", "MetaDriveEnvSlot"),
    "ObservationMode": (".slot", "ObservationMode"),
    "capture_traffic_frame": (".snapshot", "capture_traffic_frame"),
}

__all__ = [
    "EnvSlotReset",
    "EnvSlotState",
    "EnvSlotStep",
    "EnvSlotTiming",
    "KinematicTrajectoryPolicy",
    "LocalRouteUnavailableError",
    "MetaDriveBackend",
    "MetaDriveEnvSlot",
    "ObservationMode",
    "capture_traffic_frame",
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
