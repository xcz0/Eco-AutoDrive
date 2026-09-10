"""PPO-guided planning organized into explicit policy, rollout, and optimization layers."""

from .artifacts import (
    TrainingRunSummary,
    TrainingUpdateSummary,
    policy_state_hash,
    write_rollout_episode,
    write_training_runtime_metadata,
)
from .config import TrainingJobConfig, parse_training_config
from .optimization import (
    PPO_BATCH_KEYS,
    PPOConfig,
    PPOUpdater,
    build_ppo_batch,
    load_exploration_policy_checkpoint,
    normalize_full_batch_advantage,
    save_exploration_policy_checkpoint,
)
from .policy import ExplorationPolicy
from .reward import PlannerRFTNoEnergyRewardConfig
from .rollout import (
    RewardProfileName,
    RolloutEpisode,
    TailKind,
    VectorRolloutCollector,
    concatenate_tensordicts,
    create_fabric_rollout_runtime,
    derive_rollout_seeds,
    rollout_audit_keys,
)

__all__ = [
    "TrainingJobConfig",
    "parse_training_config",
    "TrainingUpdateSummary",
    "policy_state_hash",
    "create_fabric_rollout_runtime",
    "load_exploration_policy_checkpoint",
    "concatenate_tensordicts",
    "PlannerRFTNoEnergyRewardConfig",
    "RolloutEpisode",
    "PPO_BATCH_KEYS",
    "PPOConfig",
    "PPOUpdater",
    "build_ppo_batch",
    "normalize_full_batch_advantage",
    "write_rollout_episode",
    "RewardProfileName",
    "TailKind",
    "rollout_audit_keys",
    "ExplorationPolicy",
    "write_training_runtime_metadata",
    "save_exploration_policy_checkpoint",
    "VectorRolloutCollector",
    "derive_rollout_seeds",
    "TrainingRunSummary",
]
