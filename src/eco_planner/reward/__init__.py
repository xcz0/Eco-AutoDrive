"""Pure reward configuration, component scores, objectives, and results.

Reward only depends on objective-neutral facts and the resolved reward profile;
it does not import rollout episodes, PPO batches, collectors, or training state.
"""

from __future__ import annotations

from .components.comfort import component_score
from .components.progress import score_delta
from .config import (
    PlannerRFTEnergyRewardConfig,
    PlannerRFTNoEnergyRewardConfig,
    RewardProfileConfig,
)
from .evaluator import RewardEvaluator, create_reward_evaluator
from .objectives import evaluate_plannerrft_energy_step, evaluate_plannerrft_no_energy_step
from .result import RewardComponents, RewardDiagnostics, RewardProfileName, RewardResult

__all__ = [
    "PlannerRFTEnergyRewardConfig",
    "PlannerRFTNoEnergyRewardConfig",
    "RewardComponents",
    "RewardDiagnostics",
    "RewardEvaluator",
    "RewardProfileConfig",
    "RewardProfileName",
    "RewardResult",
    "component_score",
    "create_reward_evaluator",
    "evaluate_plannerrft_energy_step",
    "evaluate_plannerrft_no_energy_step",
    "score_delta",
]
