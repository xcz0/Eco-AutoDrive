"""Internal benchmark workflows and artifact analysis."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import write_benchmark_artifacts


def __getattr__(name: str) -> Any:
    if name != "write_benchmark_artifacts":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(".config", __name__), name)
    globals()[name] = value
    return value


__all__ = ["write_benchmark_artifacts"]
