"""Lazy façade for planner observation contracts and builders."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .arrays import MapObservationArrays
    from .builder import ObservationBuilder
    from .history import TrafficHistory
    from .scene import TrafficObservationAudit, TrafficSceneEncoder
    from .schema import PLANNER_OBSERVATION_FIELDS

_EXPORTS = {
    "MapObservationArrays": (".arrays", "MapObservationArrays"),
    "ObservationBuilder": (".builder", "ObservationBuilder"),
    "TrafficHistory": (".history", "TrafficHistory"),
    "TrafficObservationAudit": (".scene", "TrafficObservationAudit"),
    "TrafficSceneEncoder": (".scene", "TrafficSceneEncoder"),
    "PLANNER_OBSERVATION_FIELDS": (".schema", "PLANNER_OBSERVATION_FIELDS"),
}

__all__ = [
    "MapObservationArrays",
    "ObservationBuilder",
    "PLANNER_OBSERVATION_FIELDS",
    "TrafficHistory",
    "TrafficObservationAudit",
    "TrafficSceneEncoder",
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
