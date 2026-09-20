"""Rollout planner phase timing aggregation for benchmark reporting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from eco_planner.runtime.profiling import PhaseTiming


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
