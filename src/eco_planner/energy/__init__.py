"""Objective-neutral energy metric providers."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .fastsim import FASTSimEnergyConfig, FASTSimEnergyProvider
    from .metrics import EnergyMetricName, EnergyMetrics, EnergyTrace
    from .protocol import EnergyMetricProvider
    from .proxy import MetaDriveFuelProxyProvider

_EXPORTS = {
    "EnergyMetricName": (".metrics", "EnergyMetricName"),
    "EnergyMetrics": (".metrics", "EnergyMetrics"),
    "EnergyTrace": (".metrics", "EnergyTrace"),
    "EnergyMetricProvider": (".protocol", "EnergyMetricProvider"),
    "MetaDriveFuelProxyProvider": (".proxy", "MetaDriveFuelProxyProvider"),
    "FASTSimEnergyConfig": (".fastsim", "FASTSimEnergyConfig"),
    "FASTSimEnergyProvider": (".fastsim", "FASTSimEnergyProvider"),
}

__all__ = [
    "EnergyMetricName",
    "EnergyMetricProvider",
    "EnergyMetrics",
    "EnergyTrace",
    "FASTSimEnergyConfig",
    "FASTSimEnergyProvider",
    "MetaDriveFuelProxyProvider",
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
