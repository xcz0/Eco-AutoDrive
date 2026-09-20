"""Opt-in phase profiling with no disabled-path CUDA synchronization."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import TypeVar

import torch


@dataclass(frozen=True)
class PhaseTiming:
    """Host call wall and accelerator work for one profiled phase."""

    host_call_wall_s: float
    accelerator_s: float


@dataclass(frozen=True)
class PendingPhaseTiming:
    """Unresolved phase timing that must not read CUDA events before a sync."""

    host_call_wall_s: float
    start_event: torch.cuda.Event | None
    end_event: torch.cuda.Event | None

    def resolve(self) -> PhaseTiming:
        accelerator_s = self.host_call_wall_s
        if self.start_event is not None and self.end_event is not None:
            accelerator_s = self.start_event.elapsed_time(self.end_event) / 1000.0
        return PhaseTiming(self.host_call_wall_s, accelerator_s)


_T = TypeVar("_T")


def profile_call(
    device: torch.device,
    enabled: bool,
    operation: Callable[[], _T],
) -> tuple[_T, PendingPhaseTiming | None]:
    if not enabled:
        return operation(), None
    start_event: torch.cuda.Event | None = None
    end_event: torch.cuda.Event | None = None
    if device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record(torch.cuda.current_stream(device))
    started = perf_counter()
    result = operation()
    host_call_wall_s = perf_counter() - started
    if end_event is not None:
        end_event.record(torch.cuda.current_stream(device))
    return result, PendingPhaseTiming(host_call_wall_s, start_event, end_event)


def finish_profile(device: torch.device, enabled: bool) -> float:
    if not enabled or device.type != "cuda":
        return 0.0
    started = perf_counter()
    torch.cuda.current_stream(device).synchronize()
    return perf_counter() - started


def require_phase(timing: PendingPhaseTiming | None) -> PhaseTiming:
    if timing is None:
        raise RuntimeError("profiled phase did not return timing")
    return timing.resolve()


class PhaseProfiler:
    """Collect named phase timings and resolve them only after one finish sync."""

    def __init__(self, device: torch.device) -> None:
        self._device = device
        self._pending: dict[str, PendingPhaseTiming | None] = {}
        self._resolved: dict[str, PhaseTiming] = {}

    def measure(self, name: str, operation: Callable[[], _T]) -> _T:
        result, pending = profile_call(self._device, True, operation)
        self._pending[name] = pending
        return result

    def finish(self) -> float:
        wait = finish_profile(self._device, True)
        self._resolved = {name: require_phase(timing) for name, timing in self._pending.items()}
        self._pending = {}
        return wait

    def phase(self, name: str) -> PhaseTiming:
        if name not in self._resolved:
            raise RuntimeError(f"phase {name!r} was not profiled before finish()")
        return self._resolved[name]
