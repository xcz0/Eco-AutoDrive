"""RL entrypoints without importing execution layers during artifact inspection."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
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

_EXPORTS = {
    "TrainingRunSummary": ".artifacts",
    "TrainingUpdateSummary": ".artifacts",
    "policy_state_hash": ".artifacts",
    "write_rollout_episode": ".artifacts",
    "write_training_runtime_metadata": ".artifacts",
    "TrainingJobConfig": ".config",
    "parse_training_config": ".config",
    "PPO_BATCH_KEYS": ".optimization",
    "PPOConfig": ".optimization",
    "PPOUpdater": ".optimization",
    "build_ppo_batch": ".optimization",
    "load_exploration_policy_checkpoint": ".optimization",
    "normalize_full_batch_advantage": ".optimization",
    "save_exploration_policy_checkpoint": ".optimization",
    "ExplorationPolicy": ".policy",
    "PlannerRFTNoEnergyRewardConfig": ".reward",
    "RewardProfileName": ".rollout",
    "RolloutEpisode": ".rollout",
    "TailKind": ".rollout",
    "VectorRolloutCollector": ".rollout",
    "concatenate_tensordicts": ".rollout",
    "create_fabric_rollout_runtime": ".rollout",
    "derive_rollout_seeds": ".rollout",
    "rollout_audit_keys": ".rollout",
}

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


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
