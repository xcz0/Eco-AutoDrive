"""Fixed-slot vector closed-loop PPO training orchestration."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import cast

from omegaconf import OmegaConf

from eco_planner.artifacts import write_json
from eco_planner.planning.policy import (
    ExplorationPolicyContext,
    policy_state_hash,
    save_exploration_policy_checkpoint,
)
from eco_planner.rl.artifacts.io import write_rollout_episode, write_training_runtime_metadata
from eco_planner.rl.artifacts.metrics import build_update_summary
from eco_planner.rl.artifacts.summaries import (
    PolicyProbeSummary,
    TrainingRunSummary,
    TrainingUpdateSummary,
)
from eco_planner.rl.optimization.checkpoint import save_training_checkpoint
from eco_planner.rl.optimization.ppo import PPOUpdater
from eco_planner.runtime.resources import require_resource_profile

from .config import TrainingJobConfig
from .probing import capture_probe_contexts, probe_policy
from .rollout.collector import VectorRolloutCollector
from .rollout.contracts import RolloutEpisode
from .rollout.runtime import FabricRolloutRuntime
from .rollout.seeds import RolloutRandomStreams, create_rollout_streams, derive_rollout_seeds
from .setup import (
    configure_training_execution,
    create_training_collector,
    create_training_runtime,
)
from .tracking import TrainingTracking
from .training_state import TrainingLoopState, resume_training_state

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
    configure_training_execution(config)
    resources = require_resource_profile(config.resources)
    noise_seeds, policy_seeds = derive_rollout_seeds(config.runtime.seed, len(config.scenarios))
    runtime = create_training_runtime(config, policy_seeds[0])
    updater = PPOUpdater(runtime.policy, config.ppo)
    state = resume_training_state(config, runtime, updater)
    start_update = state.completed_updates
    tracking.attach(runtime.fabric, state.tracking)
    state.tracking = tracking.identity
    write_training_runtime_metadata(
        output_dir / "runtime_metadata.json",
        runtime,
        resources,
    )
    tracking.runtime_metadata()
    streams = create_rollout_streams(runtime, noise_seeds, policy_seeds)
    if config.training.resume_checkpoint_path is not None:
        state.restore_rollout_rng(streams)
    planner_hash_before = runtime.frozen_planner_hash()
    state.initial_policy_hash = state.initial_policy_hash or policy_state_hash(runtime.policy)
    if start_update == 0:
        save_exploration_policy_checkpoint(output_dir / "policy-initial.pt", runtime.policy)
        tracking.artifact("policy-initial.pt")

    with create_training_collector(config, runtime) as rollout_collector:
        for update_index in range(start_update, config.training.update_count):
            update_summary = _run_update(
                config,
                output_dir,
                update_index=update_index,
                runtime=runtime,
                updater=updater,
                rollout_collector=rollout_collector,
                state=state,
                tracking=tracking,
                streams=streams,
            )
            state.update_summaries.append(update_summary)
            state.completed_updates = update_index + 1
            state.capture_rollout_rng(streams)
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


def _run_update(
    config: TrainingJobConfig,
    output_dir: Path,
    *,
    update_index: int,
    runtime: FabricRolloutRuntime,
    updater: PPOUpdater,
    rollout_collector: VectorRolloutCollector,
    state: TrainingLoopState,
    tracking: TrainingTracking,
    streams: tuple[RolloutRandomStreams, ...],
) -> TrainingUpdateSummary:
    """Own one update's storage and episode views until its summary is built."""

    update_episodes: list[RolloutEpisode] = []
    batch = rollout_collector.collect(
        transitions_per_slot=config.training.transitions_per_environment,
        streams=streams,
    )
    for slot, episodes in enumerate(batch.episodes_by_slot):
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
        state.probe_contexts = capture_probe_contexts(batch.episodes_by_slot, len(config.scenarios))
        state.probe_before = probe_policy(
            runtime,
            state.probe_contexts,
            config.training.boundary_sample_count,
            config.training.boundary_distance,
            config.training.diagnostic_seed,
        )
        tracking.probe("before", state.probe_before, update_index)
    report = updater.update(batch.training)
    return build_update_summary(update_index, tuple(update_episodes), report)
