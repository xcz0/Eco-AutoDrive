"""Strict Stage-6 training summaries and transition artifacts."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt

from eco_planner.rl.policy import ExplorationPolicy
from eco_planner.rl.ppo import PPOUpdateReport
from eco_planner.rl.rollout import RolloutEpisode, RolloutTransition

TRAINING_ARTIFACT_SCHEMA_VERSION = 1


class _ArtifactModel(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)


class Stage6UpdateSummary(_ArtifactModel):
    update_index: StrictInt = Field(ge=0)
    sample_count: Literal[32]
    episode_count: StrictInt = Field(gt=0)
    total_reward: StrictFloat
    dense_reward: StrictFloat
    terminal_override: StrictFloat
    route_completion_delta: StrictFloat
    distance_m: StrictFloat = Field(ge=0.0)
    mean_speed_mps: StrictFloat = Field(ge=0.0)
    stopped_fraction: StrictFloat = Field(ge=0.0, le=1.0)
    collision_count: StrictInt = Field(ge=0)
    out_of_road_count: StrictInt = Field(ge=0)
    maximum_position_error_m: StrictFloat = Field(ge=0.0)
    maximum_heading_error_rad: StrictFloat = Field(ge=0.0)
    beta_alpha_mean: tuple[StrictFloat, StrictFloat]
    beta_beta_mean: tuple[StrictFloat, StrictFloat]
    action_mean: tuple[StrictFloat, StrictFloat]
    action_variance: tuple[StrictFloat, StrictFloat]
    mean_policy_loss: StrictFloat
    mean_value_loss: StrictFloat
    mean_entropy_loss: StrictFloat
    mean_total_loss: StrictFloat
    mean_approximate_kl: StrictFloat
    mean_clip_fraction: StrictFloat
    mean_entropy: StrictFloat
    mean_explained_variance: StrictFloat
    maximum_pre_clip_gradient_norm: StrictFloat
    final_learning_rate: StrictFloat = Field(ge=0.0)


class Stage6ProbeSummary(_ArtifactModel):
    alpha: tuple[tuple[StrictFloat, StrictFloat], ...]
    beta: tuple[tuple[StrictFloat, StrictFloat], ...]
    guidance_mean: tuple[tuple[StrictFloat, StrictFloat], ...]
    boundary_mass: tuple[tuple[StrictFloat, StrictFloat], ...]


class Stage6RunSummary(_ArtifactModel):
    schema_version: Literal[1] = TRAINING_ARTIFACT_SCHEMA_VERSION
    status: Literal["completed"] = "completed"
    training_seed: StrictInt = Field(ge=0)
    replay_id: StrictInt = Field(ge=0, le=1)
    noise_seeds: tuple[StrictInt, StrictInt]
    policy_action_seeds: tuple[StrictInt, StrictInt]
    total_transitions: Literal[128]
    initial_policy_hash: str = Field(min_length=64, max_length=64)
    final_policy_hash: str = Field(min_length=64, max_length=64)
    frozen_planner_hash_before: str = Field(min_length=64, max_length=64)
    frozen_planner_hash_after: str = Field(min_length=64, max_length=64)
    probe_before: Stage6ProbeSummary
    probe_after: Stage6ProbeSummary
    updates: tuple[Stage6UpdateSummary, ...]


def policy_state_hash(policy: ExplorationPolicy) -> str:
    """Hash one policy state dict in stable name order."""

    if not isinstance(policy, ExplorationPolicy):
        raise TypeError("policy hash requires ExplorationPolicy")
    digest = hashlib.sha256()
    for name, value in sorted(policy.state_dict().items()):
        host = value.detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(host.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def write_rollout_episode(path: Path, episode: RolloutEpisode) -> None:
    """Persist every strict transition field without a DDIM denoise chain."""

    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = _transition_arrays(episode.transitions)
    arrays.update(
        {
            "schema_version": np.asarray(TRAINING_ARTIFACT_SCHEMA_VERSION, dtype=np.int64),
            "tail_kind": np.asarray(episode.tail_kind),
            "tail_bootstrap_value": episode.tail_bootstrap_value.numpy(),
        }
    )
    np.savez(path, **arrays)


def write_partial_rollout(path: Path, transitions: tuple[RolloutTransition, ...]) -> None:
    """Persist the collected prefix of a classified episode failure."""

    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = _transition_arrays(transitions) if transitions else {}
    arrays.update(
        {
            "schema_version": np.asarray(TRAINING_ARTIFACT_SCHEMA_VERSION, dtype=np.int64),
            "trace_status": np.asarray("partial" if transitions else "empty"),
        }
    )
    np.savez(path, **arrays)


def _transition_arrays(
    transitions: tuple[RolloutTransition, ...],
) -> dict[str, np.ndarray]:
    return {
        "scene_tokens": _tensor_stack(transitions, "scene_tokens"),
        "scene_padding_mask": _tensor_stack(transitions, "scene_padding_mask"),
        "navigation_tokens": _tensor_stack(transitions, "navigation_tokens"),
        "navigation_padding_mask": _tensor_stack(transitions, "navigation_padding_mask"),
        "reference_trajectory": _tensor_stack(transitions, "reference_trajectory"),
        "base_action": _stack([step.base_action[0] for step in transitions]),
        "guidance_action": _stack([step.guidance_action[0] for step in transitions]),
        "beta_alpha": _stack([step.beta_alpha[0] for step in transitions]),
        "beta_beta": _stack([step.beta_beta[0] for step in transitions]),
        "old_joint_guidance_log_prob": _stack(
            [step.old_joint_guidance_log_prob for step in transitions]
        ),
        "old_value": _stack([step.old_value for step in transitions]),
        "initial_noise": _stack([step.initial_noise[0] for step in transitions]),
        "diffusion_rng_state": _stack([step.diffusion_rng_state for step in transitions]),
        "policy_rng_state": _stack([step.policy_rng_state for step in transitions]),
        "reward": _scalars([step.reward.total_score.item() for step in transitions]),
        "dense_reward": _scalars([step.reward.dense_step_scores.item() for step in transitions]),
        "terminal_override": _scalars(
            [step.reward.terminal_override_deltas.item() for step in transitions]
        ),
        "route_completion_delta": _scalars(
            [step.audit.route_completion_delta for step in transitions]
        ),
        "distance_m": _scalars([step.audit.distance_m for step in transitions]),
        "speed_mps": _scalars([step.audit.speed_mps for step in transitions]),
        "stopped": np.asarray([step.audit.stopped for step in transitions], dtype=np.bool_),
        "position_error_m": _scalars([step.audit.position_error_m for step in transitions]),
        "heading_error_rad": _scalars([step.audit.heading_error_rad for step in transitions]),
        "terminated": np.asarray([step.terminated for step in transitions], dtype=np.bool_),
        "truncated": np.asarray([step.truncated for step in transitions], dtype=np.bool_),
        "out_of_road": np.asarray([step.audit.out_of_road for step in transitions], dtype=np.bool_),
        "collision": np.asarray(
            [
                step.audit.crash_vehicle
                or step.audit.crash_object
                or step.audit.crash_building
                or step.audit.crash_human
                for step in transitions
            ],
            dtype=np.bool_,
        ),
        "map_seed": np.asarray([step.map_seed for step in transitions], dtype=np.int64),
        "noise_seed": np.asarray([step.noise_seed for step in transitions], dtype=np.int64),
        "policy_action_seed": np.asarray(
            [step.policy_action_seed for step in transitions], dtype=np.int64
        ),
        "planning_cycle_index": np.asarray(
            [step.planning_cycle_index for step in transitions], dtype=np.int64
        ),
    }


def build_update_summary(
    update_index: int,
    episodes: tuple[RolloutEpisode, ...],
    report: PPOUpdateReport,
) -> Stage6UpdateSummary:
    transitions = tuple(step for episode in episodes for step in episode.transitions)
    if len(transitions) != 32:
        raise ValueError("Stage-6 update summary requires exactly 32 transitions")
    actions = torch.cat([step.guidance_action for step in transitions], dim=0)
    alpha = torch.cat([step.beta_alpha for step in transitions], dim=0)
    beta = torch.cat([step.beta_beta for step in transitions], dim=0)
    collision_count = sum(
        step.audit.crash_vehicle
        or step.audit.crash_object
        or step.audit.crash_building
        or step.audit.crash_human
        for step in transitions
    )
    payload = {
        "update_index": update_index,
        "sample_count": 32,
        "episode_count": len(episodes),
        "total_reward": sum(step.reward.total_score.item() for step in transitions),
        "dense_reward": sum(step.reward.dense_step_scores.item() for step in transitions),
        "terminal_override": sum(
            step.reward.terminal_override_deltas.item() for step in transitions
        ),
        "route_completion_delta": sum(step.audit.route_completion_delta for step in transitions),
        "distance_m": sum(step.audit.distance_m for step in transitions),
        "mean_speed_mps": sum(step.audit.speed_mps for step in transitions) / len(transitions),
        "stopped_fraction": sum(step.audit.stopped for step in transitions) / len(transitions),
        "collision_count": collision_count,
        "out_of_road_count": sum(step.audit.out_of_road for step in transitions),
        "maximum_position_error_m": max(step.audit.position_error_m for step in transitions),
        "maximum_heading_error_rad": max(step.audit.heading_error_rad for step in transitions),
        "beta_alpha_mean": tuple(float(value) for value in alpha.mean(dim=0)),
        "beta_beta_mean": tuple(float(value) for value in beta.mean(dim=0)),
        "action_mean": tuple(float(value) for value in actions.mean(dim=0)),
        "action_variance": tuple(float(value) for value in actions.var(dim=0, correction=0)),
    }
    payload.update(asdict(report))
    payload.pop("optimizer_step_count")
    return Stage6UpdateSummary.model_validate(payload)


def write_json(path: Path, payload: BaseModel | dict[str, object]) -> None:
    if isinstance(payload, BaseModel):
        value = payload.model_dump(mode="json")
    else:
        value = payload
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )


def write_training_runtime_metadata(path: Path, runtime: object) -> None:
    root = path.parent
    metadata = {
        "schema_version": TRAINING_ARTIFACT_SCHEMA_VERSION,
        "git_head": _git(root, "rev-parse", "HEAD").strip(),
        "git_status_short": _git(root, "status", "--short").splitlines(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "lightning": version("lightning"),
        "metadrive": version("metadrive-simulator"),
        "runtime": asdict(runtime.report),
        "checkpoint": asdict(runtime.checkpoint_report),
        "sampler": asdict(runtime.sampler_report),
        "guidance": asdict(runtime.guidance_config),
    }
    write_json(path, metadata)
    (root / "tracked_diff.patch").write_text(
        _git(root, "diff", "--binary", "--no-ext-diff"), encoding="utf-8"
    )


def _git(output_dir: Path, *args: str) -> str:
    repository = Path.cwd()
    result = subprocess.run(
        ["git", *args], cwd=repository, check=True, capture_output=True, text=True, encoding="utf-8"
    )
    return result.stdout


def _tensor_stack(transitions: tuple[object, ...], name: str) -> np.ndarray:
    tensors = [getattr(step.policy_context, name)[0] for step in transitions]
    return _stack(tensors)


def _stack(values: list[torch.Tensor]) -> np.ndarray:
    return torch.stack(values, dim=0).detach().cpu().numpy()


def _scalars(values: list[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)
