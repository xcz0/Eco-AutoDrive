"""RL-owned reward configuration, evaluation, and results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from eco_planner.envs.domain import TransitionMetrics

from .config import PlannerRFTEnergyRewardConfig
from .objectives import evaluate_plannerrft_energy_step
from .result import RewardComponents, RewardDiagnostics, RewardResult

RewardProfileConfig: TypeAlias = PlannerRFTEnergyRewardConfig


@dataclass(frozen=True, slots=True)
class RewardEvaluator:
    """Evaluate one transition according to the selected RL objective."""

    config: RewardProfileConfig

    def __call__(self, metrics: TransitionMetrics) -> RewardResult:
        return evaluate_plannerrft_energy_step(self.config, metrics)


def create_reward_evaluator(profile: RewardProfileConfig) -> RewardEvaluator:
    return RewardEvaluator(profile)


__all__ = [
    "PlannerRFTEnergyRewardConfig",
    "RewardComponents",
    "RewardDiagnostics",
    "RewardEvaluator",
    "RewardProfileConfig",
    "RewardResult",
    "create_reward_evaluator",
    "evaluate_plannerrft_energy_step",
]
