"""Reward-independent closed-loop evaluation for Exploration Policy checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, cast

import numpy as np
import torch
from hydra.utils import to_absolute_path
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt

from eco_planner.artifacts import write_json
from eco_planner.envs.metadrive.reward import MetaDriveBuiltinRewardConfig
from eco_planner.evaluation.config import ScenarioConfig
from eco_planner.rl.artifacts import policy_state_hash, write_rollout_episode
from eco_planner.rl.config import TrainingJobConfig
from eco_planner.rl.optimization import load_exploration_policy_checkpoint
from eco_planner.rl.rollout import (
    RolloutEpisode,
    VectorRolloutCollector,
    create_fabric_rollout_runtime,
)
from eco_planner.rl.rollout.contracts import concatenate_tensordicts

_EVALUATION_SEED_NAMESPACE = 7_602_024


class _ArtifactModel(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)


class PolicyEvaluationSummary(_ArtifactModel):
    checkpoint_label: Literal["initial", "final"]
    checkpoint_path: str
    policy_hash: str = Field(min_length=64, max_length=64)
    evaluation_seed: StrictInt = Field(ge=0)
    scenarios: tuple[str, ...]
    noise_seeds: tuple[StrictInt, ...]
    transition_count: StrictInt = Field(gt=0)
    episode_count: StrictInt = Field(gt=0)
    mean_episode_length: StrictFloat = Field(gt=0.0)
    collision_count: StrictInt = Field(ge=0)
    out_of_road_count: StrictInt = Field(ge=0)
    route_completion_delta: StrictFloat
    distance_m: StrictFloat = Field(ge=0.0)
    mean_speed_mps: StrictFloat = Field(ge=0.0)
    stopped_fraction: StrictFloat = Field(ge=0.0, le=1.0)
    action_mean: tuple[StrictFloat, StrictFloat]
    action_std: tuple[StrictFloat, StrictFloat]
    action_min: tuple[StrictFloat, StrictFloat]
    action_max: tuple[StrictFloat, StrictFloat]
    beta_alpha_mean: tuple[StrictFloat, StrictFloat]
    beta_beta_mean: tuple[StrictFloat, StrictFloat]


class PolicyEvaluationComparison(_ArtifactModel):
    initial: PolicyEvaluationSummary
    final: PolicyEvaluationSummary
    episode_length_retention: StrictFloat = Field(ge=0.0)
    route_progress_retention: StrictFloat = Field(ge=0.0)
    collision_count_not_increased: bool
    out_of_road_count_not_increased: bool
    passed: bool


def evaluate_policy_checkpoint(
    config: TrainingJobConfig,
    checkpoint_path: Path,
    *,
    label: Literal["initial", "final"],
    scenarios: tuple[ScenarioConfig, ...],
    transitions_per_scenario: int,
    evaluation_seed: int,
    output_dir: Path,
) -> PolicyEvaluationSummary:
    """Evaluate one policy with deterministic mean actions on fixed scenarios and seeds."""

    if not isinstance(config.reward, MetaDriveBuiltinRewardConfig):
        raise ValueError("PPO stability evaluation requires metadrive_builtin_v1")
    if not scenarios:
        raise ValueError("policy evaluation requires at least one scenario")
    if type(transitions_per_scenario) is not int or transitions_per_scenario <= 0:
        raise ValueError("transitions_per_scenario must be a positive integer")
    if type(evaluation_seed) is not int or evaluation_seed < 0:
        raise ValueError("evaluation_seed must be a non-negative integer")
    output_dir.mkdir(parents=True, exist_ok=False)
    noise_seeds, policy_seeds = _derive_evaluation_seeds(evaluation_seed, len(scenarios))
    runtime = create_fabric_rollout_runtime(
        config.runtime,
        config.sampler,
        config.guidance,
        config.policy,
        Path(to_absolute_path(config.model.args_path)),
        Path(to_absolute_path(config.model.checkpoint_path)),
        policy_seeds[0],
        planner_compile_mode=config.training.planner_compile_mode,
    )
    load_exploration_policy_checkpoint(checkpoint_path, runtime.policy)
    diffusion_generators = tuple(runtime.new_noise_generator(seed) for seed in noise_seeds)
    policy_generators = tuple(runtime.new_policy_generator(seed) for seed in policy_seeds)
    with VectorRolloutCollector(
        scenarios,
        runtime,
        config.env,
        mode=config.training.mode,
        map_query_radius_m=config.map_query_radius_m,
        history_warmup_steps=config.training.history_warmup_steps,
        physical_slot_count=config.resources.rollout_worker_count,
        torch_threads_per_worker=config.resources.torch_threads_per_worker,
        reward_profile=config.reward,
    ) as collector:
        by_slot = collector.collect(
            transitions_per_slot=transitions_per_scenario,
            stopped_speed_threshold_mps=config.training.stopped_speed_threshold_mps,
            diffusion_generators=diffusion_generators,
            policy_generators=policy_generators,
            noise_seeds=noise_seeds,
            policy_action_seeds=policy_seeds,
            policy_sampling="mean",
        )
    episodes = tuple(episode for slot in by_slot for episode in slot)
    for slot, slot_episodes in enumerate(by_slot):
        for episode_index, episode in enumerate(slot_episodes):
            write_rollout_episode(output_dir / f"slot-{slot}-episode-{episode_index}.npz", episode)
    summary = _summarize_policy_evaluation(
        episodes,
        label=label,
        checkpoint_path=checkpoint_path,
        policy_hash=policy_state_hash(runtime.policy),
        evaluation_seed=evaluation_seed,
        scenarios=scenarios,
        noise_seeds=noise_seeds,
    )
    write_json(output_dir / "summary.json", summary)
    return summary


def compare_policy_evaluations(
    initial: PolicyEvaluationSummary,
    final: PolicyEvaluationSummary,
    *,
    minimum_retention: float = 0.9,
) -> PolicyEvaluationComparison:
    """Apply the fixed safety and progress gates to initial/final checkpoint summaries."""

    if (
        initial.evaluation_seed != final.evaluation_seed
        or initial.scenarios != final.scenarios
        or initial.noise_seeds != final.noise_seeds
    ):
        raise ValueError("policy evaluations must use identical scenarios and diffusion seeds")
    if initial.route_completion_delta <= 0.0:
        raise ValueError("initial policy evaluation must make positive route progress")
    episode_retention = final.mean_episode_length / initial.mean_episode_length
    route_retention = final.route_completion_delta / initial.route_completion_delta
    collision_ok = final.collision_count <= initial.collision_count
    out_of_road_ok = final.out_of_road_count <= initial.out_of_road_count
    passed = (
        collision_ok
        and out_of_road_ok
        and episode_retention >= minimum_retention
        and route_retention >= minimum_retention
    )
    return PolicyEvaluationComparison(
        initial=initial,
        final=final,
        episode_length_retention=float(episode_retention),
        route_progress_retention=float(route_retention),
        collision_count_not_increased=collision_ok,
        out_of_road_count_not_increased=out_of_road_ok,
        passed=passed,
    )


def _summarize_policy_evaluation(
    episodes: tuple[RolloutEpisode, ...],
    *,
    label: Literal["initial", "final"],
    checkpoint_path: Path,
    policy_hash: str,
    evaluation_seed: int,
    scenarios: tuple[ScenarioConfig, ...],
    noise_seeds: tuple[int, ...],
) -> PolicyEvaluationSummary:
    if not episodes:
        raise ValueError("policy evaluation produced no episodes")
    trajectory = concatenate_tensordicts([episode.audit for episode in episodes])
    transition_count = trajectory.batch_size[0]
    collision = trajectory["crash_vehicle"] | trajectory["crash_object"]
    collision |= trajectory["crash_building"] | trajectory["crash_human"]
    collision |= trajectory["crash_sidewalk"]
    action = cast(torch.Tensor, trajectory["guidance_action"])
    alpha = cast(torch.Tensor, trajectory["beta_alpha"])
    beta = cast(torch.Tensor, trajectory["beta_beta"])
    return PolicyEvaluationSummary(
        checkpoint_label=label,
        checkpoint_path=str(checkpoint_path),
        policy_hash=policy_hash,
        evaluation_seed=evaluation_seed,
        scenarios=tuple(f"{item.name}:{item.map}:{item.seed}" for item in scenarios),
        noise_seeds=noise_seeds,
        transition_count=transition_count,
        episode_count=len(episodes),
        mean_episode_length=float(transition_count / len(episodes)),
        collision_count=int(collision.sum()),
        out_of_road_count=int(trajectory["out_of_road"].sum()),
        route_completion_delta=float(trajectory["route_completion_delta"].sum()),
        distance_m=float(trajectory["distance_m"].sum()),
        mean_speed_mps=float(trajectory["speed_mps"].mean()),
        stopped_fraction=float(trajectory["stopped"].float().mean()),
        action_mean=_pair(action.mean(dim=0)),
        action_std=_pair(action.std(dim=0, correction=0)),
        action_min=_pair(action.min(dim=0).values),
        action_max=_pair(action.max(dim=0).values),
        beta_alpha_mean=_pair(alpha.mean(dim=0)),
        beta_beta_mean=_pair(beta.mean(dim=0)),
    )


def _derive_evaluation_seeds(
    evaluation_seed: int, scenario_count: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    sequence = np.random.SeedSequence([_EVALUATION_SEED_NAMESPACE, evaluation_seed])
    values = tuple(
        int(child.generate_state(1, dtype=np.uint32)[0])
        for child in sequence.spawn(2 * scenario_count)
    )
    return values[:scenario_count], values[scenario_count:]


def _pair(value: torch.Tensor) -> tuple[float, float]:
    host = value.detach().cpu()
    return float(host[0]), float(host[1])
