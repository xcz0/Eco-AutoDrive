"""RL-owned reward configuration, evaluation, and results."""

from __future__ import annotations

from dataclasses import dataclass

from eco_planner.envs.domain import TransitionMetrics

from .components.comfort import component_score
from .components.progress import score_delta
from .config import (
    PlannerRFTEnergyRewardConfig,
    PlannerRFTNoEnergyRewardConfig,
    RewardProfileConfig,
)
from .objectives import evaluate_plannerrft_energy_step, evaluate_plannerrft_no_energy_step
from .result import RewardComponents, RewardDiagnostics, RewardResult


@dataclass(frozen=True, slots=True)
class RewardEvaluator:
    """Evaluate one transition according to the selected RL objective."""

    config: RewardProfileConfig

    def __call__(self, metrics: TransitionMetrics) -> RewardResult:
        if isinstance(self.config, PlannerRFTNoEnergyRewardConfig):
            return evaluate_plannerrft_no_energy_step(self.config, metrics)
        return evaluate_plannerrft_energy_step(self.config, metrics)


def create_reward_evaluator(profile: RewardProfileConfig) -> RewardEvaluator:
    return RewardEvaluator(profile)


__all__ = [
    "PlannerRFTEnergyRewardConfig",
    "PlannerRFTNoEnergyRewardConfig",
    "RewardComponents",
    "RewardDiagnostics",
    "RewardEvaluator",
    "RewardProfileConfig",
    "RewardResult",
    "create_reward_evaluator",
    "evaluate_plannerrft_energy_step",
    "evaluate_plannerrft_no_energy_step",
    "component_score",
    "score_delta",
]
