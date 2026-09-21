"""Pure per-transition reward evaluation over objective-neutral metrics."""

from __future__ import annotations

from dataclasses import dataclass

from eco_planner.envs.domain import TransitionMetrics

from .config import PlannerRFTNoEnergyRewardConfig, RewardProfileConfig
from .objectives import evaluate_plannerrft_energy_step, evaluate_plannerrft_no_energy_step
from .result import RewardResult


@dataclass(frozen=True, slots=True)
class RewardEvaluator:
    """Evaluate one transition according to the resolved reward objective."""

    config: RewardProfileConfig

    def __call__(self, metrics: TransitionMetrics) -> RewardResult:
        if isinstance(self.config, PlannerRFTNoEnergyRewardConfig):
            return evaluate_plannerrft_no_energy_step(self.config, metrics)
        return evaluate_plannerrft_energy_step(self.config, metrics)


def create_reward_evaluator(profile: RewardProfileConfig) -> RewardEvaluator:
    return RewardEvaluator(profile)


__all__ = ["RewardEvaluator", "create_reward_evaluator"]
