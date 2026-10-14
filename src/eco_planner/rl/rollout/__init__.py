"""Policy-guided rollout public API."""

from .assembly import (
    RolloutEpisodeBuilder,
    build_training_decision,
    build_training_transition,
)
from .collector import (
    VectorRolloutCollector,
    collect_rollout_episode,
    collect_vector_rollout_episodes,
)
from .contracts import (
    ExecutionTransitionAudit,
    RewardProfileName,
    RolloutEpisode,
    RolloutProvenance,
    TailKind,
    rollout_audit_keys,
)
from .profiling import VectorRolloutRoundTiming
from .runtime import FabricRolloutRuntime, create_fabric_rollout_runtime
from .seeds import derive_rollout_seeds

__all__ = [
    "ExecutionTransitionAudit",
    "FabricRolloutRuntime",
    "RolloutEpisode",
    "RolloutEpisodeBuilder",
    "RolloutProvenance",
    "TailKind",
    "VectorRolloutCollector",
    "VectorRolloutRoundTiming",
    "build_training_decision",
    "build_training_transition",
    "collect_rollout_episode",
    "collect_vector_rollout_episodes",
    "create_fabric_rollout_runtime",
    "rollout_audit_keys",
    "RewardProfileName",
    "derive_rollout_seeds",
]
