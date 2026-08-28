"""Policy-guided rollout public API."""

from eco_planner.rl.rollout.collector import (
    VectorRolloutCollector,
    VectorRolloutRoundTiming,
    collect_rollout_episode,
    collect_vector_rollout_episodes,
)
from eco_planner.rl.rollout.contracts import (
    DecisionAudit,
    ExecutionTransitionAudit,
    RolloutEpisode,
    RolloutEpisodeBuilder,
    RolloutProvenance,
    TailKind,
    build_training_decision,
)
from eco_planner.rl.rollout.runtime import FabricRolloutRuntime, create_fabric_rollout_runtime

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
]
