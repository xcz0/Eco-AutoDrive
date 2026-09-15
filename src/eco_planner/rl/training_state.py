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
    diffusion_rng_states: tuple[torch.Tensor, ...] = ()
    policy_rng_states: tuple[torch.Tensor, ...] = ()

    def capture_rollout_rng(
        self,
        diffusion_generators: tuple[torch.Generator, ...],
        policy_generators: tuple[torch.Generator, ...],
    ) -> None:
        self.diffusion_rng_states = tuple(g.get_state().clone() for g in diffusion_generators)
        self.policy_rng_states = tuple(g.get_state().clone() for g in policy_generators)

    def restore_rollout_rng(
        self,
        diffusion_generators: tuple[torch.Generator, ...],
        policy_generators: tuple[torch.Generator, ...],
    ) -> None:
        for generators, states in (
            (diffusion_generators, self.diffusion_rng_states),
            (policy_generators, self.policy_rng_states),
        ):
            for generator, state in zip(generators, states, strict=True):
                expected_seed = generator.initial_seed()
                generator.set_state(state.cpu())
                if generator.initial_seed() != expected_seed:
                    raise ValueError("resume rollout RNG seed does not match its logical slot")

    def checkpoint_payload(self) -> dict[str, object]:
        return {
            "diffusion_rng_states": self.diffusion_rng_states,
            "policy_rng_states": self.policy_rng_states,
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
    diffusion_rng_states = _rollout_rng_states(loop, "diffusion_rng_states", len(config.scenarios))
    policy_rng_states = _rollout_rng_states(loop, "policy_rng_states", len(config.scenarios))
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
        diffusion_rng_states=diffusion_rng_states,
        policy_rng_states=policy_rng_states,
    )


def _rollout_rng_states(
    loop: dict[str, object], key: str, scenario_count: int
) -> tuple[torch.Tensor, ...]:
    if key not in loop:
        raise ValueError(f"resume checkpoint is missing {key}; exact rollout resume is impossible")
    states = loop[key]
    if not isinstance(states, (tuple, list)) or len(states) != scenario_count:
        raise ValueError(f"resume checkpoint {key} must match the logical scenario count")
    if not all(
        isinstance(state, torch.Tensor) and state.dtype == torch.uint8 and state.ndim == 1
        for state in states
    ):
        raise TypeError(f"resume checkpoint {key} must contain one-dimensional uint8 RNG states")
    return tuple(state.cpu().clone() for state in states)


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
