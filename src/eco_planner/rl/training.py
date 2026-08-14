"""Stage-6 serial closed-loop PPO smoke-training orchestration."""

from __future__ import annotations

import traceback
from pathlib import Path

import numpy as np
import torch
from hydra.utils import to_absolute_path

from eco_planner.rl.checkpoint import save_exploration_policy_checkpoint
from eco_planner.rl.collector import RolloutCollectionFailure, collect_rollout_episode
from eco_planner.rl.policy import (
    BetaGuidanceDistribution,
    BetaGuidanceParameters,
    ExplorationPolicyContext,
)
from eco_planner.rl.ppo import PPOUpdater
from eco_planner.rl.rollout import RolloutEpisode
from eco_planner.rl.runtime import create_fabric_rollout_runtime
from eco_planner.rl.training_artifact import (
    Stage6ProbeSummary,
    Stage6RunSummary,
    build_update_summary,
    policy_state_hash,
    write_json,
    write_partial_rollout,
    write_rollout_episode,
    write_training_runtime_metadata,
)
from eco_planner.rl.training_config import Stage6TrainingJobConfig

_SEED_NAMESPACE = 6_002_024


def run_stage6_training(config: Stage6TrainingJobConfig, output_dir: Path) -> Stage6RunSummary:
    """Run the fixed 2x16x4 Stage-6 smoke profile and persist all transitions."""

    if not isinstance(config, Stage6TrainingJobConfig):
        raise TypeError("config must be Stage6TrainingJobConfig")
    if not isinstance(output_dir, Path):
        raise TypeError("output_dir must be pathlib.Path")
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "summary.json").exists():
        raise FileExistsError(f"Stage-6 output already contains a summary: {output_dir}")
    if config.training.deterministic:
        torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("high")
    noise_seeds, policy_seeds = _derive_rollout_seeds(config.runtime.seed)
    runtime = create_fabric_rollout_runtime(
        config.runtime,
        config.sampler,
        config.guidance,
        config.policy,
        Path(to_absolute_path(config.model.args_path)),
        Path(to_absolute_path(config.model.checkpoint_path)),
        policy_seeds[0],
    )
    updater = PPOUpdater(runtime.policy, config.train)
    diffusion_generators = tuple(runtime.new_noise_generator(seed) for seed in noise_seeds)
    policy_generators = tuple(runtime.new_policy_generator(seed) for seed in policy_seeds)
    planner_hash_before = runtime.frozen_planner_hash()
    initial_policy_hash = policy_state_hash(runtime.policy)
    save_exploration_policy_checkpoint(output_dir / "policy-initial.pt", runtime.policy)
    update_summaries = []
    probe_contexts: tuple[ExplorationPolicyContext, ...] | None = None
    probe_before: Stage6ProbeSummary | None = None
    total_transitions = 0

    for update_index in range(config.training.update_count):
        update_episodes: list[RolloutEpisode] = []
        for slot, scenario in enumerate(config.scenarios):
            remaining = config.training.transitions_per_environment
            episode_index = 0
            while remaining:
                try:
                    episode = collect_rollout_episode(
                        scenario,
                        runtime,
                        config.env,
                        mode="no_traffic",
                        map_query_radius_m=config.map_query_radius_m,
                        history_warmup_steps=0,
                        max_transitions=remaining,
                        stopped_speed_threshold_mps=config.training.stopped_speed_threshold_mps,
                        diffusion_generator=diffusion_generators[slot],
                        policy_generator=policy_generators[slot],
                        noise_seed=noise_seeds[slot],
                        policy_action_seed=policy_seeds[slot],
                    )
                except RolloutCollectionFailure as failure:
                    failure_dir = output_dir / "failures" / f"update-{update_index:03d}-slot-{slot}"
                    write_partial_rollout(failure_dir / "trace.npz", failure.transitions)
                    write_json(
                        failure_dir / "failure.json",
                        {
                            "phase": failure.phase,
                            "exception_type": type(failure.cause).__name__,
                            "message": str(failure.cause),
                            "traceback": traceback.format_exc(),
                        },
                    )
                    write_training_runtime_metadata(output_dir / "runtime_metadata.json", runtime)
                    raise
                write_rollout_episode(
                    output_dir
                    / "updates"
                    / f"update-{update_index:03d}"
                    / f"slot-{slot}-episode-{episode_index}.npz",
                    episode,
                )
                update_episodes.append(episode)
                transition_count = len(episode.transitions)
                total_transitions += transition_count
                remaining -= transition_count
                episode_index += 1
        episode_tuple = tuple(update_episodes)
        if probe_contexts is None:
            probe_contexts = tuple(
                next(
                    step.policy_context
                    for episode in episode_tuple
                    for step in episode.transitions
                    if step.scenario_name == scenario.name
                )
                for scenario in config.scenarios
            )
            probe_before = _probe_policy(
                runtime,
                probe_contexts,
                config.training.boundary_sample_count,
                config.training.boundary_distance,
                config.training.diagnostic_seed,
            )
        report = updater.update(episode_tuple)
        update_summaries.append(build_update_summary(update_index, episode_tuple, report))
        save_exploration_policy_checkpoint(
            output_dir / f"policy-update-{update_index:03d}.pt", runtime.policy
        )

    if probe_contexts is None or probe_before is None:
        raise RuntimeError("Stage-6 training did not capture fixed probe contexts")
    probe_after = _probe_policy(
        runtime,
        probe_contexts,
        config.training.boundary_sample_count,
        config.training.boundary_distance,
        config.training.diagnostic_seed,
    )
    final_policy_hash = policy_state_hash(runtime.policy)
    planner_hash_after = runtime.frozen_planner_hash()
    save_exploration_policy_checkpoint(output_dir / "policy-final.pt", runtime.policy)
    if total_transitions != 128:
        raise RuntimeError(
            f"Stage-6 training collected {total_transitions} transitions, expected 128"
        )
    if planner_hash_after != planner_hash_before:
        raise RuntimeError("Stage-6 PPO mutated the frozen planner")
    summary = Stage6RunSummary(
        training_seed=config.runtime.seed,
        replay_id=config.training.replay_id,
        noise_seeds=noise_seeds,
        policy_action_seeds=policy_seeds,
        total_transitions=128,
        initial_policy_hash=initial_policy_hash,
        final_policy_hash=final_policy_hash,
        frozen_planner_hash_before=planner_hash_before,
        frozen_planner_hash_after=planner_hash_after,
        probe_before=probe_before,
        probe_after=probe_after,
        updates=tuple(update_summaries),
    )
    write_json(output_dir / "summary.json", summary)
    write_training_runtime_metadata(output_dir / "runtime_metadata.json", runtime)
    return summary


def _derive_rollout_seeds(training_seed: int) -> tuple[tuple[int, int], tuple[int, int]]:
    sequence = np.random.SeedSequence([_SEED_NAMESPACE, training_seed])
    children = sequence.spawn(4)
    values = tuple(int(child.generate_state(1, dtype=np.uint32)[0]) for child in children)
    if len(set(values)) != 4:
        raise RuntimeError("Stage-6 seed derivation produced duplicate random streams")
    return (values[0], values[1]), (values[2], values[3])


def _probe_policy(
    runtime: object,
    contexts: tuple[ExplorationPolicyContext, ...],
    sample_count: int,
    boundary_distance: float,
    diagnostic_seed: int,
) -> Stage6ProbeSummary:
    alpha_values: list[tuple[float, float]] = []
    beta_values: list[tuple[float, float]] = []
    means: list[tuple[float, float]] = []
    masses: list[tuple[float, float]] = []
    for index, host_context in enumerate(contexts):
        context = _context_to_device(host_context, runtime.device)
        with torch.no_grad():
            output = runtime.policy(context)
        alpha = output.parameters.alpha
        beta = output.parameters.beta
        mean = output.distribution.mean().guidance_action
        expanded = BetaGuidanceDistribution(
            BetaGuidanceParameters(
                alpha=alpha.expand(sample_count, -1),
                beta=beta.expand(sample_count, -1),
            )
        )
        generator = runtime.new_policy_generator(diagnostic_seed + index)
        samples = expanded.sample(generator).base_action
        boundary = (samples <= boundary_distance) | (samples >= 1.0 - boundary_distance)
        alpha_values.append(tuple(float(value) for value in alpha[0].cpu()))
        beta_values.append(tuple(float(value) for value in beta[0].cpu()))
        means.append(tuple(float(value) for value in mean[0].cpu()))
        masses.append(tuple(float(value) for value in boundary.float().mean(dim=0).cpu()))
    return Stage6ProbeSummary(
        alpha=tuple(alpha_values),
        beta=tuple(beta_values),
        guidance_mean=tuple(means),
        boundary_mass=tuple(masses),
    )


def _context_to_device(
    context: ExplorationPolicyContext, device: torch.device
) -> ExplorationPolicyContext:
    return ExplorationPolicyContext(
        scene_tokens=context.scene_tokens.to(device),
        scene_padding_mask=context.scene_padding_mask.to(device),
        navigation_tokens=context.navigation_tokens.to(device),
        navigation_padding_mask=context.navigation_padding_mask.to(device),
        reference_trajectory=context.reference_trajectory.to(device),
    )
