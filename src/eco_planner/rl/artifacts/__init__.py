"""Training artifact schema, I/O, and analysis public API."""

from eco_planner.rl.artifacts.analysis import summarize_training_runs
from eco_planner.rl.artifacts.io import (
    PolicyProbeSummary,
    PPOGradientDiagnosticsSummary,
    RewardComponentMeans,
    TrainingRunSummary,
    TrainingUpdateSummary,
    build_update_summary,
    policy_state_hash,
    write_rollout_episode,
    write_training_runtime_metadata,
)
from eco_planner.rl.artifacts.schema import (
    BUILTIN_ROLLOUT_ARTIFACT_FIELDS,
    ENERGY_ROLLOUT_ARTIFACT_FIELDS,
    rollout_artifact_fields,
)

__all__ = [
    "BUILTIN_ROLLOUT_ARTIFACT_FIELDS",
    "ENERGY_ROLLOUT_ARTIFACT_FIELDS",
    "PolicyProbeSummary",
    "PPOGradientDiagnosticsSummary",
    "RewardComponentMeans",
    "TrainingRunSummary",
    "TrainingUpdateSummary",
    "build_update_summary",
    "policy_state_hash",
    "rollout_artifact_fields",
    "summarize_training_runs",
    "write_rollout_episode",
    "write_training_runtime_metadata",
]
