"""Policy-guided rollout public API."""

from .collector import (
    VectorRolloutCollector,
    VectorRolloutRoundTiming,
    collect_rollout_episode,
    collect_vector_rollout_episodes,
)
from .contracts import (
    DecisionAudit,
    ExecutionTransitionAudit,
    RewardProfileName,
    RolloutEpisode,
    RolloutEpisodeBuilder,
    RolloutProvenance,
    TailKind,
    build_training_decision,
    concatenate_tensordicts,
    rollout_audit_keys,
)
from .runtime import FabricRolloutRuntime, create_fabric_rollout_runtime
from .seeds import derive_rollout_seeds

__all__ = [
    "DecisionAudit",
    "ExecutionTransitionAudit",
    "FabricRolloutRuntime",
    "RolloutEpisode",
    "RolloutEpisodeBuilder",
    "RolloutProvenance",
    "TailKind",
    "VectorRolloutCollector",
    "VectorRolloutRoundTiming",
    "build_training_decision",
    "collect_rollout_episode",
    "collect_vector_rollout_episodes",
    "create_fabric_rollout_runtime",
    "concatenate_tensordicts",
    "rollout_audit_keys",
    "RewardProfileName",
    "derive_rollout_seeds",
]
