"""Training loop state and its existing checkpoint payload boundary."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from hydra.utils import to_absolute_path

from eco_planner.rl.artifacts.schema import PolicyProbeSummary, TrainingUpdateSummary
from eco_planner.rl.config import TrainingJobConfig
from eco_planner.rl.optimization import PPOUpdater, load_training_checkpoint
from eco_planner.rl.policy import ExplorationPolicyContext
from eco_planner.rl.rollout import FabricRolloutRuntime
from eco_planner.rl.tracking import TrackingIdentity


@dataclass
class TrainingLoopState:
    completed_updates: int = 0
    total_transitions: int = 0
    update_summaries: list[TrainingUpdateSummary] = field(default_factory=list)
    probe_before: PolicyProbeSummary | None = None
    probe_contexts: tuple[ExplorationPolicyContext, ...] | None = None
    initial_policy_hash: str | None = None
    tracking: TrackingIdentity | None = None

    def checkpoint_payload(self) -> dict[str, object]:
        return {
            "tracking": self.tracking.model_dump() if self.tracking is not None else None,
            "completed_updates": self.completed_updates,
            "initial_policy_hash": self.initial_policy_hash,
            "total_transitions": self.total_transitions,
            "update_summaries": tuple(
                item.model_dump(mode="json") for item in self.update_summaries
            ),
            "probe_before": self.probe_before.model_dump(mode="json")
            if self.probe_before is not None
            else None,
            "probe_contexts": (
                tuple(_serialize_context(context) for context in self.probe_contexts)
                if self.probe_contexts is not None
                else None
            ),
        }


def resume_training_state(
    config: TrainingJobConfig, runtime: FabricRolloutRuntime, updater: PPOUpdater
) -> TrainingLoopState:
    path = config.training.resume_checkpoint_path
    if path is None:
        return TrainingLoopState()
    checkpoint_path = Path(to_absolute_path(path))
    report, loop = load_training_checkpoint(
        checkpoint_path, runtime.fabric, runtime.policy, updater
    )
    if report.completed_updates > config.training.update_count:
        raise ValueError("resume checkpoint has more updates than the configured training job")
    summaries_payload = loop["update_summaries"]
    if not isinstance(summaries_payload, (list, tuple)):
        raise TypeError("resume checkpoint has invalid update summaries")
    summaries = [
        TrainingUpdateSummary.model_validate_json(json.dumps(item)) for item in summaries_payload
    ]
    if len(summaries) != report.completed_updates:
        raise ValueError("resume checkpoint update summaries disagree with its update count")
    probe_payload = loop["probe_before"]
    contexts_payload = loop["probe_contexts"]
    probe = (
        PolicyProbeSummary.model_validate_json(json.dumps(probe_payload))
        if probe_payload is not None
        else None
    )
    if contexts_payload is not None and not isinstance(contexts_payload, (list, tuple)):
        raise TypeError("resume checkpoint has invalid policy probe contexts")
    contexts = (
        tuple(_deserialize_context(item) for item in contexts_payload)
        if contexts_payload is not None
        else None
    )
    if report.completed_updates and (probe is None or contexts is None):
        raise ValueError("resume checkpoint is missing policy probe state")
    total = loop["total_transitions"]
    if type(total) is not int or total < 0:
        raise ValueError("resume checkpoint has an invalid transition total")
    initial_policy_hash = loop["initial_policy_hash"]
    if not isinstance(initial_policy_hash, str) or len(initial_policy_hash) != 64:
        raise ValueError("resume checkpoint has an invalid initial policy hash")
    identity = TrackingIdentity.model_validate(loop["tracking"]) if loop.get("tracking") else None
    return TrainingLoopState(
        completed_updates=report.completed_updates,
        total_transitions=total,
        update_summaries=summaries,
        probe_before=probe,
        probe_contexts=contexts,
        initial_policy_hash=initial_policy_hash,
        tracking=identity,
    )


def _serialize_context(context: ExplorationPolicyContext) -> dict[str, torch.Tensor]:
    return {
        "scene_tokens": context.scene_tokens,
        "scene_padding_mask": context.scene_padding_mask,
        "navigation_tokens": context.navigation_tokens,
        "navigation_padding_mask": context.navigation_padding_mask,
        "reference_trajectory": context.reference_trajectory,
    }


def _deserialize_context(payload: Any) -> ExplorationPolicyContext:
    if not isinstance(payload, dict) or not all(
        isinstance(value, torch.Tensor) for value in payload.values()
    ):
        raise TypeError("resume checkpoint has an invalid policy context")
    return ExplorationPolicyContext(**payload)
