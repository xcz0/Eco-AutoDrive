"""RL-specific summaries and TensorDict-to-NPZ artifact adapters."""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import torch
from hydra.utils import to_absolute_path
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt
from tensordict import TensorDictBase

from eco_planner.artifacts import (
    collect_repository_metadata,
    write_json,
    write_npz,
    write_tracked_diff,
)
from eco_planner.rl.artifacts.schema import rollout_artifact_fields
from eco_planner.rl.optimization.ppo import PPOUpdateReport
from eco_planner.rl.policy import ExplorationPolicy
from eco_planner.rl.rollout.contracts import (
    RolloutEpisode,
    concatenate_tensordicts,
    rollout_audit_keys,
)
from eco_planner.runtime.resources import ResourceProfileConfig

if TYPE_CHECKING:
    from eco_planner.rl.rollout.runtime import FabricRolloutRuntime


class _ArtifactModel(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)


class RewardComponentMeans(_ArtifactModel):
    reward_gate: StrictFloat = Field(ge=0.0, le=1.0)
    collision_score: StrictFloat = Field(ge=0.0, le=1.0)
    drivable_score: StrictFloat = Field(ge=0.0, le=1.0)
    wrong_direction_score: StrictFloat = Field(ge=0.0, le=1.0)
    ttc_score: StrictFloat = Field(ge=0.0, le=1.0)
    progress_score: StrictFloat = Field(ge=0.0, le=1.0)
    comfort_score: StrictFloat = Field(ge=0.0, le=1.0)
    speed_score: StrictFloat = Field(ge=0.0, le=1.0)
    energy_score: StrictFloat = Field(ge=0.0, le=1.0)


class PPOGradientDiagnosticsSummary(_ArtifactModel):
    actor_head_policy: StrictFloat = Field(ge=0.0)
    shared_trunk_policy: StrictFloat = Field(ge=0.0)
    value_head_critic: StrictFloat = Field(ge=0.0)
    shared_trunk_critic: StrictFloat = Field(ge=0.0)
    actor_head_entropy: StrictFloat = Field(ge=0.0)
    shared_trunk_entropy: StrictFloat = Field(ge=0.0)


class TrainingUpdateSummary(_ArtifactModel):
    update_index: StrictInt = Field(ge=0)
    sample_count: StrictInt = Field(gt=0)
    episode_count: StrictInt = Field(gt=0)
    mean_episode_length: StrictFloat = Field(gt=0.0)
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
    beta_alpha_min: tuple[StrictFloat, StrictFloat]
    beta_alpha_max: tuple[StrictFloat, StrictFloat]
    beta_beta_mean: tuple[StrictFloat, StrictFloat]
    beta_beta_min: tuple[StrictFloat, StrictFloat]
    beta_beta_max: tuple[StrictFloat, StrictFloat]
    action_mean: tuple[StrictFloat, StrictFloat]
    action_std: tuple[StrictFloat, StrictFloat]
    action_min: tuple[StrictFloat, StrictFloat]
    action_max: tuple[StrictFloat, StrictFloat]
    mean_state_value: StrictFloat
    std_state_value: StrictFloat
    mean_policy_loss: StrictFloat
    mean_value_loss: StrictFloat
    mean_entropy_loss: StrictFloat
    mean_total_loss: StrictFloat
    mean_approximate_kl: StrictFloat
    mean_clip_fraction: StrictFloat
    mean_entropy: StrictFloat
    mean_explained_variance: StrictFloat
    maximum_pre_clip_gradient_norm: StrictFloat
    evaluated_minibatch_count: StrictInt = Field(gt=0)
    optimizer_step_count: StrictInt = Field(ge=0)
    final_learning_rate: StrictFloat = Field(ge=0.0)
    raw_advantage_mean: StrictFloat
    raw_advantage_std: StrictFloat = Field(ge=0.0)
    normalized_advantage_mean: StrictFloat
    normalized_advantage_std: StrictFloat = Field(ge=0.0)
    mean_value_target: StrictFloat
    std_value_target: StrictFloat
    kl_early_stopped: StrictBool
    kl_early_stop_trigger: StrictFloat | None
    cumulative_kl_early_stop_count: StrictInt = Field(ge=0)
    policy_ratio_mean: StrictFloat = Field(gt=0.0)
    policy_ratio_std: StrictFloat = Field(ge=0.0)
    policy_ratio_p95: StrictFloat = Field(gt=0.0)
    policy_ratio_max: StrictFloat = Field(gt=0.0)
    gradient_diagnostics: PPOGradientDiagnosticsSummary | None
    reward_profile: Literal["metadrive_builtin_v1", "plannerrft_energy_v1"]
    native_step_energy_total_ml: StrictFloat = Field(ge=0.0)
    executed_fuel_proxy_total_ml: StrictFloat = Field(ge=0.0)
    executed_fuel_proxy_distance_m: StrictFloat = Field(ge=0.0)
    executed_fuel_proxy_ml_per_km: StrictFloat | None = Field(ge=0.0)
    reward_component_means: RewardComponentMeans | None


class PolicyProbeSummary(_ArtifactModel):
    alpha: tuple[tuple[StrictFloat, StrictFloat], ...]
    beta: tuple[tuple[StrictFloat, StrictFloat], ...]
    guidance_mean: tuple[tuple[StrictFloat, StrictFloat], ...]
    boundary_mass: tuple[tuple[StrictFloat, StrictFloat], ...]


class TrainingRunSummary(_ArtifactModel):
    status: Literal["completed"]
    training_seed: StrictInt = Field(ge=0)
    replay_id: StrictInt = Field(ge=0)
    noise_seeds: tuple[StrictInt, ...]
    policy_action_seeds: tuple[StrictInt, ...]
    total_transitions: StrictInt = Field(gt=0)
    initial_policy_hash: str = Field(min_length=64, max_length=64)
    final_policy_hash: str = Field(min_length=64, max_length=64)
    frozen_planner_hash_before: str = Field(min_length=64, max_length=64)
    frozen_planner_hash_after: str = Field(min_length=64, max_length=64)
    probe_before: PolicyProbeSummary
    probe_after: PolicyProbeSummary
    updates: tuple[TrainingUpdateSummary, ...]
    reward_profile: Literal["metadrive_builtin_v1", "plannerrft_energy_v1"]


def policy_state_hash(policy: ExplorationPolicy) -> str:
    """Hash one policy state dict in stable name order."""

    digest = hashlib.sha256()
    for name, value in sorted(policy.state_dict().items()):
        host = value.detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(host.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def write_rollout_episode(path: Path, episode: RolloutEpisode) -> None:
    """Persist the complete audit trajectory through the stable NumPy artifact boundary."""

    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = _trajectory_arrays(episode)
    arrays.update(
        {
            "reward_profile": np.asarray(episode.reward_profile),
            "tail_kind": np.asarray(episode.tail_kind),
            "tail_bootstrap_value": episode.tail_bootstrap_value.cpu().numpy(),
        }
    )
    expected_fields = set(rollout_artifact_fields(episode.reward_profile))
    if set(arrays) != expected_fields:
        raise RuntimeError("rollout artifact payload does not match its explicit schema")
    write_npz(path, arrays)


def build_update_summary(
    update_index: int, episodes: tuple[RolloutEpisode, ...], report: PPOUpdateReport
) -> TrainingUpdateSummary:
    if not episodes:
        raise ValueError("training update summary requires at least one episode")
    reward_profiles = {episode.reward_profile for episode in episodes}
    if len(reward_profiles) != 1:
        raise ValueError("training update cannot mix rollout reward profiles")
    reward_profile = reward_profiles.pop()
    trajectory = concatenate_tensordicts([episode.audit for episode in episodes])
    sample_count = trajectory.batch_size[0]
    episode_count = len(episodes)
    mean_episode_length = sample_count / episode_count
    collision = (
        _tensor(trajectory, "crash_vehicle")
        | _tensor(trajectory, "crash_object")
        | _tensor(trajectory, "crash_building")
        | _tensor(trajectory, "crash_human")
    )
    collision |= _tensor(trajectory, "crash_sidewalk")
    state_value = _tensor(trajectory, "state_value")
    beta_alpha = _tensor(trajectory, "beta_alpha")
    beta_beta = _tensor(trajectory, "beta_beta")
    guidance_action = _tensor(trajectory, "guidance_action")
    payload = {
        "update_index": update_index,
        "sample_count": sample_count,
        "episode_count": episode_count,
        "mean_episode_length": float(mean_episode_length),
        "total_reward": float(_tensor(trajectory, "reward").sum()),
        "dense_reward": float(_tensor(trajectory, "dense_reward").sum()),
        "terminal_override": float(_tensor(trajectory, "terminal_override").sum()),
        "route_completion_delta": float(_tensor(trajectory, "route_completion_delta").sum()),
        "distance_m": float(_tensor(trajectory, "distance_m").sum()),
        "mean_speed_mps": float(_tensor(trajectory, "speed_mps").mean()),
        "stopped_fraction": float(_tensor(trajectory, "stopped").float().mean()),
        "collision_count": int(collision.sum()),
        "out_of_road_count": int(_tensor(trajectory, "out_of_road").sum()),
        "maximum_position_error_m": float(_tensor(trajectory, "position_error_m").max()),
        "maximum_heading_error_rad": float(_tensor(trajectory, "heading_error_rad").max()),
        "beta_alpha_mean": tuple(float(value) for value in beta_alpha.mean(dim=0)),
        "beta_alpha_min": tuple(float(value) for value in beta_alpha.min(dim=0).values),
        "beta_alpha_max": tuple(float(value) for value in beta_alpha.max(dim=0).values),
        "beta_beta_mean": tuple(float(value) for value in beta_beta.mean(dim=0)),
        "beta_beta_min": tuple(float(value) for value in beta_beta.min(dim=0).values),
        "beta_beta_max": tuple(float(value) for value in beta_beta.max(dim=0).values),
        "action_mean": tuple(float(value) for value in guidance_action.mean(dim=0)),
        "action_std": tuple(float(value) for value in guidance_action.std(dim=0, correction=0)),
        "action_min": tuple(float(value) for value in guidance_action.min(dim=0).values),
        "action_max": tuple(float(value) for value in guidance_action.max(dim=0).values),
        "mean_state_value": float(state_value.mean()),
        "std_state_value": float(state_value.std(correction=0)),
    }
    proxy_total = float(_tensor(trajectory, "executed_fuel_proxy_step_energy_ml").sum())
    distance_total = float(_tensor(trajectory, "step_distance_m").sum())
    payload.update(
        {
            "reward_profile": reward_profile,
            "native_step_energy_total_ml": float(
                _tensor(trajectory, "native_step_energy_ml").sum()
            ),
            "executed_fuel_proxy_total_ml": proxy_total,
            "executed_fuel_proxy_distance_m": distance_total,
            "executed_fuel_proxy_ml_per_km": (
                proxy_total * 1000.0 / distance_total if distance_total > 0.0 else None
            ),
            "reward_component_means": None,
        }
    )
    if reward_profile == "plannerrft_energy_v1":
        payload.update(
            {
                "reward_component_means": {
                    name: float(_tensor(trajectory, name).mean())
                    for name in (
                        "reward_gate",
                        "collision_score",
                        "drivable_score",
                        "wrong_direction_score",
                        "ttc_score",
                        "progress_score",
                        "comfort_score",
                        "speed_score",
                        "energy_score",
                    )
                },
            }
        )
    payload.update(asdict(report))
    return TrainingUpdateSummary.model_validate(payload)


def write_training_runtime_metadata(
    path: Path, runtime: FabricRolloutRuntime, resources: ResourceProfileConfig
) -> None:
    """Record common reproducibility metadata plus the RL runtime selections."""

    repository_root = Path(to_absolute_path("."))
    metadata = {
        **collect_repository_metadata(repository_root),
        "runtime": asdict(runtime.report),
        "checkpoint": asdict(runtime.checkpoint_report),
        "sampler": asdict(runtime.sampler_report),
        "guidance": asdict(runtime.guidance_config),
        "resources": resources.model_dump(mode="json"),
    }
    write_json(path, metadata)
    write_tracked_diff(path.parent / "tracked_diff.patch", repository_root)


def _trajectory_arrays(episode: RolloutEpisode) -> dict[str, np.ndarray]:
    return {
        name: _tensor(episode.audit, name).detach().cpu().numpy()
        for name in rollout_audit_keys(episode.reward_profile)
    }


def _tensor(trajectory: TensorDictBase, key: str) -> torch.Tensor:
    return cast(torch.Tensor, trajectory[key])
