"""Rollout planner and collection timing aggregation for benchmark reporting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from eco_planner.runtime.profiling import PhaseTiming

if TYPE_CHECKING:
    from eco_planner.envs.parallel import WorkerStepResult


@dataclass(frozen=True)
class RolloutPlannerTiming:
    """Profile-only timing for one decision or bootstrap planner batch."""

    phase: Literal["decision", "bootstrap"]
    host_to_device: PhaseTiming
    diffusion_noise: PhaseTiming
    prepare_policy_guidance: PhaseTiming
    policy_forward: PhaseTiming
    action_sampling: PhaseTiming | None
    complete_policy_guidance: PhaseTiming | None
    execution_to_host: PhaseTiming | None
    profile_sync_wait_wall_s: float


@dataclass(frozen=True)
class VectorRolloutRoundTiming:
    """One fixed-slot collection round, exposed for throughput profiling only."""

    phase: Literal["decision", "bootstrap"]
    active_slots: int
    capacity: int
    collate_wall_s: float
    planner_wall_s: float
    planner_timing: RolloutPlannerTiming
    environment_wall_s: float
    audit_resolve_wall_s: float
    audit_transfer_accelerator_s: float
    worker_busy_s: float
    transport_sync_s: float
    worker_imbalance_s: float


def append_decision_timing(
    timings: list[VectorRolloutRoundTiming] | None,
    steps: tuple[WorkerStepResult, ...],
    *,
    capacity: int,
    collate_s: float,
    planner_s: float,
    planner_timing: RolloutPlannerTiming,
    environment_s: float,
    audit_resolve_s: float,
    audit_transfer_accelerator_s: float,
) -> None:
    if timings is None:
        return
    worker_busy = tuple(step.timing.environment_s + step.timing.observation_s for step in steps)
    slowest = max(worker_busy)
    timings.append(
        VectorRolloutRoundTiming(
            phase="decision",
            active_slots=len(steps),
            capacity=capacity,
            collate_wall_s=collate_s,
            planner_wall_s=planner_s,
            planner_timing=planner_timing,
            environment_wall_s=environment_s,
            audit_resolve_wall_s=audit_resolve_s,
            audit_transfer_accelerator_s=audit_transfer_accelerator_s,
            worker_busy_s=sum(worker_busy),
            transport_sync_s=max(0.0, environment_s - slowest),
            worker_imbalance_s=sum(slowest - value for value in worker_busy),
        )
    )


def append_bootstrap_timing(
    timings: list[VectorRolloutRoundTiming] | None,
    *,
    active_slots: int,
    capacity: int,
    collate_s: float,
    planner_s: float,
    planner_timing: RolloutPlannerTiming,
) -> None:
    if timings is None:
        return
    timings.append(
        VectorRolloutRoundTiming(
            phase="bootstrap",
            active_slots=active_slots,
            capacity=capacity,
            collate_wall_s=collate_s,
            planner_wall_s=planner_s,
            planner_timing=planner_timing,
            environment_wall_s=0.0,
            audit_resolve_wall_s=0.0,
            audit_transfer_accelerator_s=0.0,
            worker_busy_s=0.0,
            transport_sync_s=0.0,
            worker_imbalance_s=0.0,
        )
    )


def single_planner_timing(
    timings: list[RolloutPlannerTiming], phase: Literal["decision", "bootstrap"]
) -> RolloutPlannerTiming:
    if len(timings) != 1 or timings[0].phase != phase:
        raise RuntimeError(f"profiled {phase} batch must return exactly one matching timing")
    return timings[0]
