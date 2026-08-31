"""Fixed-slot vector closed-loop PPO training orchestration."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from hydra.utils import to_absolute_path

from eco_planner.artifacts import write_json
from eco_planner.rl.artifacts import (
    PolicyProbeSummary,
    TrainingRunSummary,
    TrainingUpdateSummary,
    build_update_summary,
    policy_state_hash,
    write_rollout_episode,
    write_training_runtime_metadata,
)
from eco_planner.rl.config import TrainingJobConfig
from eco_planner.rl.optimization import (
    PPOUpdater,
    load_training_checkpoint,
    save_exploration_policy_checkpoint,
    save_training_checkpoint,
)
from eco_planner.rl.policy import ExplorationPolicyContext, policy_context_tensordict
from eco_planner.rl.policy.distribution import AffineBeta
from eco_planner.rl.rollout import (
    FabricRolloutRuntime,
    RolloutEpisode,
    VectorRolloutCollector,
    create_fabric_rollout_runtime,
)
from eco_planner.runtime_resources import ResourceProfileConfig

_SEED_NAMESPACE = 6_002_024
TrainingUpdateObserver = Callable[[TrainingUpdateSummary], None]


def train(
    config: TrainingJobConfig,
    output_dir: Path,
    *,
    update_observer: TrainingUpdateObserver | None = None,
) -> TrainingRunSummary:
    """Run a configured closed-loop PPO job and persist policies and research artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "summary.json").exists():
        raise FileExistsError(f"training output already contains a summary: {output_dir}")
    if config.training.deterministic:
        torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("high")
    resources = cast(ResourceProfileConfig, config.resources)
    scenario_count = len(config.scenarios)
    noise_seeds, policy_seeds = _derive_rollout_seeds(config.runtime.seed, scenario_count)
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
    updater = PPOUpdater(runtime.policy, config.ppo)
    (
        start_update,
        total_transitions,
        update_summaries,
        probe_before,
        probe_contexts,
        resumed_initial_policy_hash,
    ) = _resume_state(config, runtime, updater)
    diffusion_generators = tuple(runtime.new_noise_generator(seed) for seed in noise_seeds)
    policy_generators = tuple(runtime.new_policy_generator(seed) for seed in policy_seeds)
    planner_hash_before = runtime.frozen_planner_hash()
    initial_policy_hash = resumed_initial_policy_hash or policy_state_hash(runtime.policy)
    if start_update == 0:
        save_exploration_policy_checkpoint(output_dir / "policy-initial.pt", runtime.policy)

    with VectorRolloutCollector(
        config.scenarios,
        runtime,
        config.env,
        mode=config.training.mode,
        map_query_radius_m=config.map_query_radius_m,
        history_warmup_steps=config.training.history_warmup_steps,
        physical_slot_count=resources.rollout_worker_count,
        torch_threads_per_worker=resources.torch_threads_per_worker,
        reward_profile=config.reward,
    ) as rollout_collector:
        for update_index in range(start_update, config.training.update_count):
            update_episodes: list[RolloutEpisode] = []
            update_contexts: list[ExplorationPolicyContext] = []
            slot_episodes = rollout_collector.collect(
                transitions_per_slot=config.training.transitions_per_environment,
                stopped_speed_threshold_mps=config.training.stopped_speed_threshold_mps,
                diffusion_generators=diffusion_generators,
                policy_generators=policy_generators,
                noise_seeds=noise_seeds,
                policy_action_seeds=policy_seeds,
            )
            for slot, episodes in enumerate(slot_episodes):
                for episode_index, episode in enumerate(episodes):
                    write_rollout_episode(
                        output_dir
                        / "updates"
                        / f"update-{update_index:03d}"
                        / f"slot-{slot}-episode-{episode_index}.npz",
                        episode,
                    )
                    if episode_index == 0:
                        item = episode.training[0]
                        update_contexts.append(
                            ExplorationPolicyContext(
                                scene_tokens=item["scene_tokens"].unsqueeze(0),
                                scene_padding_mask=item["scene_padding_mask"].unsqueeze(0),
                                navigation_tokens=item["navigation_tokens"].unsqueeze(0),
                                navigation_padding_mask=item["navigation_padding_mask"].unsqueeze(
                                    0
                                ),
                                reference_trajectory=item["reference_trajectory"].unsqueeze(0),
                            )
                        )
                    update_episodes.append(episode)
                    total_transitions += episode.transition_count
            if probe_contexts is None:
                if len(update_contexts) != scenario_count:
                    raise RuntimeError(
                        "training did not capture one fixed probe context per scenario"
                    )
                probe_contexts = tuple(update_contexts)
                probe_before = _probe_policy(
                    runtime,
                    probe_contexts,
                    config.training.boundary_sample_count,
                    config.training.boundary_distance,
                    config.training.diagnostic_seed,
                )
            report = updater.update(tuple(update_episodes))
            update_summary = build_update_summary(update_index, tuple(update_episodes), report)
            update_summaries.append(update_summary)
            save_exploration_policy_checkpoint(
                output_dir / f"policy-update-{update_index:03d}.pt", runtime.policy
            )
            save_training_checkpoint(
                output_dir / "training-state.ckpt",
                runtime.fabric,
                runtime.policy,
                updater,
                _loop_state(
                    update_index + 1,
                    total_transitions,
                    update_summaries,
                    probe_before,
                    probe_contexts,
                    initial_policy_hash,
                ),
            )
            if update_observer is not None:
                update_observer(update_summary)

    probe_contexts = cast(tuple[ExplorationPolicyContext, ...], probe_contexts)
    probe_before = cast(PolicyProbeSummary, probe_before)
    expected_total = config.training.update_count * config.ppo.batch_size
    if total_transitions != expected_total:
        raise RuntimeError(
            f"training collected {total_transitions} transitions, expected {expected_total}"
        )
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
    if planner_hash_after != planner_hash_before:
        raise RuntimeError("PPO mutated the frozen planner")
    summary = TrainingRunSummary(
        status="completed",
        training_seed=config.runtime.seed,
        replay_id=config.training.replay_id,
        noise_seeds=noise_seeds,
        policy_action_seeds=policy_seeds,
        total_transitions=total_transitions,
        initial_policy_hash=initial_policy_hash,
        final_policy_hash=final_policy_hash,
        frozen_planner_hash_before=planner_hash_before,
        frozen_planner_hash_after=planner_hash_after,
        probe_before=probe_before,
        probe_after=probe_after,
        updates=tuple(update_summaries),
        reward_profile=config.reward.name,
    )
    write_json(output_dir / "summary.json", summary)
    write_training_runtime_metadata(output_dir / "runtime_metadata.json", runtime, resources)
    return summary


def _resume_state(
    config: TrainingJobConfig, runtime: FabricRolloutRuntime, updater: PPOUpdater
) -> tuple[
    int,
    int,
    list[TrainingUpdateSummary],
    PolicyProbeSummary | None,
    tuple[ExplorationPolicyContext, ...] | None,
    str | None,
]:
    path = config.training.resume_checkpoint_path
    if path is None:
        return 0, 0, [], None, None, None
    checkpoint_path = Path(to_absolute_path(path))
    report, loop = load_training_checkpoint(
        checkpoint_path, runtime.fabric, runtime.policy, updater
    )
    if report.completed_updates > config.training.update_count:
        raise ValueError("resume checkpoint has more updates than the configured training job")
    summaries_payload = loop["update_summaries"]
    if not isinstance(summaries_payload, (list, tuple)):
        raise TypeError("resume checkpoint has invalid update summaries")
    summaries = [TrainingUpdateSummary.model_validate(item) for item in summaries_payload]
    if len(summaries) != report.completed_updates:
        raise ValueError("resume checkpoint update summaries disagree with its update count")
    probe_payload = loop["probe_before"]
    contexts_payload = loop["probe_contexts"]
    probe = PolicyProbeSummary.model_validate(probe_payload) if probe_payload is not None else None
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
    return report.completed_updates, total, summaries, probe, contexts, initial_policy_hash


def _loop_state(
    completed_updates: int,
    total_transitions: int,
    updates: list[TrainingUpdateSummary],
    probe_before: PolicyProbeSummary | None,
    probe_contexts: tuple[ExplorationPolicyContext, ...] | None,
    initial_policy_hash: str,
) -> dict[str, object]:
    return {
        "completed_updates": completed_updates,
        "initial_policy_hash": initial_policy_hash,
        "total_transitions": total_transitions,
        "update_summaries": tuple(item.model_dump(mode="json") for item in updates),
        "probe_before": probe_before.model_dump(mode="json") if probe_before is not None else None,
        "probe_contexts": (
            tuple(_serialize_context(context) for context in probe_contexts)
            if probe_contexts is not None
            else None
        ),
    }


def _derive_rollout_seeds(
    training_seed: int, scenario_count: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if type(scenario_count) is not int or scenario_count <= 0:
        raise ValueError("scenario_count must be a positive integer")
    sequence = np.random.SeedSequence([_SEED_NAMESPACE, training_seed])
    values = tuple(
        int(child.generate_state(1, dtype=np.uint32)[0])
        for child in sequence.spawn(2 * scenario_count)
    )
    if len(set(values)) != len(values):
        raise RuntimeError("seed derivation produced duplicate random streams")
    return values[:scenario_count], values[scenario_count:]


def _probe_policy(
    runtime: FabricRolloutRuntime,
    contexts: tuple[ExplorationPolicyContext, ...],
    sample_count: int,
    boundary_distance: float,
    diagnostic_seed: int,
) -> PolicyProbeSummary:
    alpha_values: list[tuple[float, float]] = []
    beta_values: list[tuple[float, float]] = []
    means: list[tuple[float, float]] = []
    masses: list[tuple[float, float]] = []
    for index, host_context in enumerate(contexts):
        context = _context_to_device(host_context, runtime.device)
        with torch.no_grad():
            outputs = runtime.policy.forward_tensordict(policy_context_tensordict(context))
            output = runtime.policy.output_from_tensordict(outputs)
        alpha = output.distribution.parameters.alpha
        beta = output.distribution.parameters.beta
        expanded = AffineBeta(alpha.expand(sample_count, -1), beta.expand(sample_count, -1))
        generator = runtime.new_policy_generator(diagnostic_seed + index)
        samples = expanded.sample(generator).base_action
        boundary = (samples <= boundary_distance) | (samples >= 1.0 - boundary_distance)
        alpha_values.append(_tensor_pair(alpha[0]))
        beta_values.append(_tensor_pair(beta[0]))
        means.append(_tensor_pair(expanded.mean[0]))
        masses.append(_tensor_pair(boundary.float().mean(dim=0)))
    return PolicyProbeSummary(
        alpha=tuple(alpha_values),
        beta=tuple(beta_values),
        guidance_mean=tuple(means),
        boundary_mass=tuple(masses),
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


def _tensor_pair(value: torch.Tensor) -> tuple[float, float]:
    if tuple(value.shape) != (2,):
        raise ValueError("policy probe statistic must have shape [2]")
    host = value.detach().cpu()
    return float(host[0]), float(host[1])
