"""Pure reward configuration, component scores, objectives, and results.

Reward only depends on objective-neutral facts and the resolved reward profile;
it does not import rollout episodes, PPO batches, collectors, or training state.
"""

from __future__ import annotations

from .aggregation import aggregate_transition_reward
from .calibration import (
    MOTION_LIMITS,
    CalibrationTargets,
    EnergyBandConfig,
    calibrate,
    energy_band_thresholds,
    scored_arrays,
)
from .components.comfort import component_score
from .components.progress import score_delta
from .config import (
    PlannerRFTEnergyRewardConfig,
    PlannerRFTNoEnergyRewardConfig,
    RewardProfileConfig,
)
from .evaluator import RewardEvaluator, create_reward_evaluator
from .objectives import (
    apply_safety_gate,
    combine_component_scores,
    evaluate_plannerrft_energy_step,
    evaluate_plannerrft_no_energy_step,
)
from .result import RewardComponents, RewardDiagnostics, RewardProfileName, RewardResult

__all__ = [
    "MOTION_LIMITS",
    "CalibrationTargets",
    "EnergyBandConfig",
    "PlannerRFTEnergyRewardConfig",
    "PlannerRFTNoEnergyRewardConfig",
    "RewardComponents",
    "RewardDiagnostics",
    "RewardEvaluator",
    "RewardProfileConfig",
    "RewardProfileName",
    "RewardResult",
    "aggregate_transition_reward",
    "apply_safety_gate",
    "calibrate",
    "combine_component_scores",
    "component_score",
    "create_reward_evaluator",
    "energy_band_thresholds",
    "evaluate_plannerrft_energy_step",
    "evaluate_plannerrft_no_energy_step",
    "score_delta",
    "scored_arrays",
]
