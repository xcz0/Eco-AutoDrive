"""Opt-in rollout phase profiling with no disabled-path CUDA synchronization."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Literal, TypeVar

import torch


@dataclass(frozen=True)
class RolloutPlannerPhaseTiming:
    """Host call wall and accelerator work for one profiled planner phase."""

    host_call_wall_s: float
    accelerator_s: float


@dataclass(frozen=True)
class RolloutPlannerTiming:
    """Profile-only timing for one decision or bootstrap planner batch."""

    phase: Literal["decision", "bootstrap"]
    host_to_device: RolloutPlannerPhaseTiming
    diffusion_noise: RolloutPlannerPhaseTiming
    prepare_policy_guidance: RolloutPlannerPhaseTiming
    policy_forward: RolloutPlannerPhaseTiming
    action_sampling: RolloutPlannerPhaseTiming | None
    complete_policy_guidance: RolloutPlannerPhaseTiming | None
    execution_to_host: RolloutPlannerPhaseTiming | None
    profile_sync_wait_wall_s: float


@dataclass(frozen=True)
class _PendingPhaseTiming:
    host_call_wall_s: float
    start_event: torch.cuda.Event | None
    end_event: torch.cuda.Event | None

    def resolve(self) -> RolloutPlannerPhaseTiming:
        accelerator_s = self.host_call_wall_s
        if self.start_event is not None and self.end_event is not None:
            accelerator_s = self.start_event.elapsed_time(self.end_event) / 1000.0
        return RolloutPlannerPhaseTiming(self.host_call_wall_s, accelerator_s)


_T = TypeVar("_T")


def profile_call(
    device: torch.device,
    enabled: bool,
    operation: Callable[[], _T],
) -> tuple[_T, _PendingPhaseTiming | None]:
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
    return result, _PendingPhaseTiming(host_call_wall_s, start_event, end_event)


def finish_profile(device: torch.device, enabled: bool) -> float:
    if not enabled or device.type != "cuda":
        return 0.0
    started = perf_counter()
    torch.cuda.current_stream(device).synchronize()
    return perf_counter() - started


def require_phase(timing: _PendingPhaseTiming | None) -> RolloutPlannerPhaseTiming:
    if timing is None:
        raise RuntimeError("profiled rollout phase did not return timing")
    return timing.resolve()
