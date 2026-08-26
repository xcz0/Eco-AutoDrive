"""RL-specific summaries and TensorDict-to-NPZ artifact adapters."""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from hydra.utils import to_absolute_path
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt
from tensordict import TensorDictBase

from eco_planner.artifacts import collect_repository_metadata, write_json, write_tracked_diff
from eco_planner.rl.policy import ExplorationPolicy
from eco_planner.rl.ppo import PPOUpdateReport
from eco_planner.rl.rollout import (
    AUDIT_FIELD_KEYS,
    ENERGY_REWARD_AUDIT_FIELD_KEYS,
    RolloutEpisode,
)
from eco_planner.runtime_resources import ResourceProfileConfig


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


class TrainingUpdateSummary(_ArtifactModel):
    update_index: StrictInt = Field(ge=0)
    sample_count: StrictInt = Field(gt=0)
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
    reward_profile: Literal["metadrive_builtin_v1", "plannerrft_energy_v1"] = (
        "metadrive_builtin_v1"
    )
    native_step_energy_total_ml: StrictFloat = Field(default=0.0, ge=0.0)
    executed_fuel_proxy_total_ml: StrictFloat = Field(default=0.0, ge=0.0)
    executed_fuel_proxy_ml_per_km: StrictFloat | None = Field(default=None, ge=0.0)
    reward_component_means: RewardComponentMeans | None = None


class PolicyProbeSummary(_ArtifactModel):
    alpha: tuple[tuple[StrictFloat, StrictFloat], ...]
    beta: tuple[tuple[StrictFloat, StrictFloat], ...]
    guidance_mean: tuple[tuple[StrictFloat, StrictFloat], ...]
    boundary_mass: tuple[tuple[StrictFloat, StrictFloat], ...]


class TrainingRunSummary(_ArtifactModel):
    status: str = "completed"
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
    reward_profile: Literal["metadrive_builtin_v1", "plannerrft_energy_v1"] = (
        "metadrive_builtin_v1"
    )


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
    """Persist the complete audit trajectory through the stable NumPy artifact boundary."""

    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = _trajectory_arrays(episode.audit)
    arrays.update(
        {
            "reward_profile": np.asarray(
                "plannerrft_energy_v1"
                if "reward_gate" in episode.audit.keys()
                else "metadrive_builtin_v1"
            ),
            "tail_kind": np.asarray(episode.tail_kind),
            "tail_bootstrap_value": episode.tail_bootstrap_value.cpu().numpy(),
        }
    )
    np.savez(path, **arrays)


def build_update_summary(
    update_index: int, episodes: tuple[RolloutEpisode, ...], report: PPOUpdateReport
) -> TrainingUpdateSummary:
    if not episodes:
        raise ValueError("training update summary requires at least one episode")
    trajectory = torch.cat([episode.audit for episode in episodes], dim=0)
    sample_count = trajectory.batch_size[0]
    collision = (
        trajectory["crash_vehicle"]
        | trajectory["crash_object"]
        | trajectory["crash_building"]
        | trajectory["crash_human"]
    )
    collision |= trajectory["crash_sidewalk"]
    payload = {
        "update_index": update_index,
        "sample_count": sample_count,
        "episode_count": len(episodes),
        "total_reward": float(trajectory["reward"].sum()),
        "dense_reward": float(trajectory["dense_reward"].sum()),
        "terminal_override": float(trajectory["terminal_override"].sum()),
        "route_completion_delta": float(trajectory["route_completion_delta"].sum()),
        "distance_m": float(trajectory["distance_m"].sum()),
        "mean_speed_mps": float(trajectory["speed_mps"].mean()),
        "stopped_fraction": float(trajectory["stopped"].float().mean()),
        "collision_count": int(collision.sum()),
        "out_of_road_count": int(trajectory["out_of_road"].sum()),
        "maximum_position_error_m": float(trajectory["position_error_m"].max()),
        "maximum_heading_error_rad": float(trajectory["heading_error_rad"].max()),
        "beta_alpha_mean": tuple(float(value) for value in trajectory["beta_alpha"].mean(dim=0)),
        "beta_beta_mean": tuple(float(value) for value in trajectory["beta_beta"].mean(dim=0)),
        "action_mean": tuple(float(value) for value in trajectory["guidance_action"].mean(dim=0)),
        "action_variance": tuple(
            float(value) for value in trajectory["guidance_action"].var(dim=0, correction=0)
        ),
    }
    proxy_total = float(trajectory["executed_fuel_proxy_step_energy_ml"].sum())
    distance_total = float(trajectory["step_distance_m"].sum())
    payload.update(
        {
            "reward_profile": (
                "plannerrft_energy_v1"
                if "reward_gate" in trajectory.keys()
                else "metadrive_builtin_v1"
            ),
            "native_step_energy_total_ml": float(trajectory["native_step_energy_ml"].sum()),
            "executed_fuel_proxy_total_ml": proxy_total,
            "executed_fuel_proxy_ml_per_km": (
                proxy_total * 1000.0 / distance_total if distance_total > 0.0 else None
            ),
        }
    )
    if "reward_gate" in trajectory.keys():
        payload.update(
            {
                "reward_component_means": {
                    name: float(trajectory[name].mean())
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
    payload.pop("optimizer_step_count")
    return TrainingUpdateSummary.model_validate(payload)


def write_training_runtime_metadata(
    path: Path, runtime: object, resources: ResourceProfileConfig
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


def _trajectory_arrays(trajectory: TensorDictBase) -> dict[str, np.ndarray]:
    keys = (
        ENERGY_REWARD_AUDIT_FIELD_KEYS
        if "reward_gate" in trajectory.keys()
        else AUDIT_FIELD_KEYS
    )
    return {name: trajectory[name].detach().cpu().numpy() for name in keys}
