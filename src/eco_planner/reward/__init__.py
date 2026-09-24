"""Pure reward configuration, component scores, objectives, and results.

Reward only depends on objective-neutral facts and the resolved reward profile;
it does not import rollout episodes, PPO batches, collectors, or training state.
"""

from __future__ import annotations

from .aggregation import (
    SubstepReward,
    aggregate_substep_rewards,
    aggregate_transition_reward,
    substep_reward,
)
from .calibration import (
    MOTION_LIMITS,
    CalibrationTargets,
    EnergyBandConfig,
    FrozenEnergyBand,
    apply_frozen_energy_band,
    calibrate,
    energy_band_thresholds,
    scored_arrays,
)
from .components.comfort import component_score
from .components.energy import energy_score_from_fuel
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
    "FrozenEnergyBand",
    "PlannerRFTEnergyRewardConfig",
    "PlannerRFTNoEnergyRewardConfig",
    "RewardComponents",
    "RewardDiagnostics",
    "RewardEvaluator",
    "RewardProfileConfig",
    "RewardProfileName",
    "RewardResult",
    "SubstepReward",
    "aggregate_substep_rewards",
    "aggregate_transition_reward",
    "apply_safety_gate",
    "apply_frozen_energy_band",
    "calibrate",
    "combine_component_scores",
    "component_score",
    "create_reward_evaluator",
    "energy_band_thresholds",
    "energy_score_from_fuel",
    "evaluate_plannerrft_energy_step",
    "evaluate_plannerrft_no_energy_step",
    "score_delta",
    "scored_arrays",
    "substep_reward",
]
