"""Training artifact interfaces loaded independently of execution modules."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from eco_planner.rl.artifacts.io import (
        policy_state_hash,
        write_rollout_episode,
        write_training_runtime_metadata,
    )
    from eco_planner.rl.artifacts.metrics import build_update_summary
    from eco_planner.rl.artifacts.schema import (
        ENERGY_ROLLOUT_ARTIFACT_FIELDS,
        NO_ENERGY_ROLLOUT_ARTIFACT_FIELDS,
        rollout_artifact_fields,
    )
    from eco_planner.rl.artifacts.summaries import (
        PolicyProbeSummary,
        PPOGradientDiagnosticsSummary,
        RewardComponentMeans,
        RewardDiagnosticMeans,
        TrainingRunSummary,
        TrainingUpdateSummary,
    )

_EXPORTS = {
    "policy_state_hash": "eco_planner.rl.artifacts.io",
    "write_rollout_episode": "eco_planner.rl.artifacts.io",
    "write_training_runtime_metadata": "eco_planner.rl.artifacts.io",
    "build_update_summary": "eco_planner.rl.artifacts.metrics",
    "ENERGY_ROLLOUT_ARTIFACT_FIELDS": "eco_planner.rl.artifacts.schema",
    "NO_ENERGY_ROLLOUT_ARTIFACT_FIELDS": "eco_planner.rl.artifacts.schema",
    "PolicyProbeSummary": "eco_planner.rl.artifacts.summaries",
    "PPOGradientDiagnosticsSummary": "eco_planner.rl.artifacts.summaries",
    "RewardComponentMeans": "eco_planner.rl.artifacts.summaries",
    "RewardDiagnosticMeans": "eco_planner.rl.artifacts.summaries",
    "TrainingRunSummary": "eco_planner.rl.artifacts.summaries",
    "TrainingUpdateSummary": "eco_planner.rl.artifacts.summaries",
    "rollout_artifact_fields": "eco_planner.rl.artifacts.schema",
}
__all__ = [
    "policy_state_hash",
    "write_rollout_episode",
    "write_training_runtime_metadata",
    "build_update_summary",
    "ENERGY_ROLLOUT_ARTIFACT_FIELDS",
    "NO_ENERGY_ROLLOUT_ARTIFACT_FIELDS",
    "PolicyProbeSummary",
    "PPOGradientDiagnosticsSummary",
    "RewardComponentMeans",
    "RewardDiagnosticMeans",
    "TrainingRunSummary",
    "TrainingUpdateSummary",
    "rollout_artifact_fields",
]


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(_EXPORTS[name]), name)
    globals()[name] = value
    return value
