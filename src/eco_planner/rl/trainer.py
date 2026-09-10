"""Fixed-slot vector closed-loop PPO training orchestration."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import cast

import torch
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf

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
    save_exploration_policy_checkpoint,
    save_training_checkpoint,
)
from eco_planner.rl.policy import ExplorationPolicyContext
from eco_planner.rl.probing import capture_probe_contexts, probe_policy
from eco_planner.rl.rollout import (
    RolloutEpisode,
    VectorRolloutCollector,
    create_fabric_rollout_runtime,
)
from eco_planner.rl.rollout.seeds import derive_rollout_seeds
from eco_planner.rl.tracking import TrainingTracking
from eco_planner.rl.training_state import resume_training_state
from eco_planner.runtime.resources import ResourceProfileConfig

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
    if not (output_dir / "resolved_config.yaml").exists():
        OmegaConf.save(
            OmegaConf.create(config.model_dump(mode="json")), output_dir / "resolved_config.yaml"
        )
    with TrainingTracking(config, output_dir) as tracking:
        tracking.start_new()
        return _train(config, output_dir, tracking, update_observer)


def _train(
    config: TrainingJobConfig,
    output_dir: Path,
    tracking: TrainingTracking,
    update_observer: TrainingUpdateObserver | None,
) -> TrainingRunSummary:
    if config.training.deterministic:
        torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("high")
    resources = cast(ResourceProfileConfig, config.resources)
    scenario_count = len(config.scenarios)
    noise_seeds, policy_seeds = derive_rollout_seeds(config.runtime.seed, scenario_count)
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
    state = resume_training_state(config, runtime, updater)
    start_update = state.completed_updates
    tracking.attach(runtime.fabric, state.update_summaries, state.tracking)
    state.tracking = tracking.identity
    write_training_runtime_metadata(output_dir / "runtime_metadata.json", runtime, resources)
    tracking.runtime_metadata()
    diffusion_generators = tuple(runtime.new_noise_generator(seed) for seed in noise_seeds)
    policy_generators = tuple(runtime.new_policy_generator(seed) for seed in policy_seeds)
    planner_hash_before = runtime.frozen_planner_hash()
    state.initial_policy_hash = state.initial_policy_hash or policy_state_hash(runtime.policy)
    if start_update == 0:
        save_exploration_policy_checkpoint(output_dir / "policy-initial.pt", runtime.policy)
        tracking.artifact("policy-initial.pt")

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
                    update_episodes.append(episode)
                    state.total_transitions += episode.transition_count
            if state.probe_contexts is None:
                state.probe_contexts = capture_probe_contexts(slot_episodes, scenario_count)
                state.probe_before = probe_policy(
                    runtime,
                    state.probe_contexts,
                    config.training.boundary_sample_count,
                    config.training.boundary_distance,
                    config.training.diagnostic_seed,
                )
                tracking.probe("before", state.probe_before, update_index)
            report = updater.update(tuple(update_episodes))
            update_summary = build_update_summary(update_index, tuple(update_episodes), report)
            state.update_summaries.append(update_summary)
            state.completed_updates = update_index + 1
            save_exploration_policy_checkpoint(
                output_dir / f"policy-update-{update_index:03d}.pt", runtime.policy
            )
            save_training_checkpoint(
                output_dir / "training-state.ckpt",
                runtime.fabric,
                runtime.policy,
                updater,
                state.checkpoint_payload(),
            )
            tracking.update(update_summary)
            if update_observer is not None:
                update_observer(update_summary)

    state.probe_contexts = cast(tuple[ExplorationPolicyContext, ...], state.probe_contexts)
    state.probe_before = cast(PolicyProbeSummary, state.probe_before)
    expected_total = config.training.update_count * config.ppo.batch_size
    if state.total_transitions != expected_total:
        raise RuntimeError(
            f"training collected {state.total_transitions} transitions, expected {expected_total}"
        )
    probe_after = probe_policy(
        runtime,
        state.probe_contexts,
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
        total_transitions=state.total_transitions,
        initial_policy_hash=state.initial_policy_hash,
        final_policy_hash=final_policy_hash,
        frozen_planner_hash_before=planner_hash_before,
        frozen_planner_hash_after=planner_hash_after,
        probe_before=state.probe_before,
        probe_after=probe_after,
        updates=tuple(state.update_summaries),
        reward_profile=config.reward.name,
    )
    write_json(output_dir / "summary.json", summary)
    if start_update == config.training.update_count:
        save_training_checkpoint(
            output_dir / "training-state.ckpt",
            runtime.fabric,
            runtime.policy,
            updater,
            state.checkpoint_payload(),
        )
    tracking.probe("after", probe_after, config.training.update_count - 1)
    tracking.artifact("summary.json")
    tracking.artifact("policy-final.pt")
    tracking.artifact("training-state.ckpt")
    return summary
