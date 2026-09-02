"""RL reward profiles, objectives, and profile-specific audits."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, TypeAlias

from pydantic import Field

from eco_planner.envs.domain.metrics import TransitionMetrics
from eco_planner.envs.metadrive.config import MetaDriveBuiltinRewardConfig
from eco_planner.rl.reward.audit import (
    MetaDriveBuiltinRewardAudit,
    PlannerRFTEnergyRewardAudit,
    RewardAudit,
    build_metadrive_builtin_reward_audit,
)
from eco_planner.rl.reward.config import PlannerRFTEnergyRewardConfig
from eco_planner.rl.reward.plannerrft import score_plannerrft_energy_step

RewardProfileConfig: TypeAlias = Annotated[
    MetaDriveBuiltinRewardConfig | PlannerRFTEnergyRewardConfig,
    Field(discriminator="name"),
]
RewardObjective: TypeAlias = Callable[[TransitionMetrics], tuple[float, float]]


@dataclass(frozen=True, slots=True)
class PlannerRFTEnergyRewardObjective:
    """Pickle-safe reward callable passed into Windows-spawned environment workers."""

    config: PlannerRFTEnergyRewardConfig

    def __call__(self, metrics: TransitionMetrics) -> tuple[float, float]:
        audit = score_plannerrft_energy_step(self.config, metrics)
        return audit.reward_total, audit.reward_ungated


def create_reward_objective(profile: RewardProfileConfig) -> RewardObjective | None:
    """Create a simulator-independent scorer; native MetaDrive reward needs no scorer."""

    if isinstance(profile, MetaDriveBuiltinRewardConfig):
        return None
    return PlannerRFTEnergyRewardObjective(profile)


def build_reward_audit(
    profile: RewardProfileConfig | None,
    metrics: TransitionMetrics,
    *,
    reward_total: float,
    dense_reward: float,
) -> RewardAudit:
    """Build the RL artifact audit from execution facts and a selected profile."""

    if profile is None or isinstance(profile, MetaDriveBuiltinRewardConfig):
        return build_metadrive_builtin_reward_audit(
            metrics, reward_total=reward_total, dense_reward=dense_reward
        )
    audit = score_plannerrft_energy_step(profile, metrics)
    if audit.reward_total != reward_total or audit.reward_ungated != dense_reward:
        raise ValueError("environment reward does not match the selected RL objective")
    return audit


__all__ = [
    "MetaDriveBuiltinRewardAudit",
    "MetaDriveBuiltinRewardConfig",
    "PlannerRFTEnergyRewardAudit",
    "PlannerRFTEnergyRewardConfig",
    "PlannerRFTEnergyRewardObjective",
    "RewardAudit",
    "RewardObjective",
    "RewardProfileConfig",
    "build_reward_audit",
    "create_reward_objective",
    "score_plannerrft_energy_step",
]
