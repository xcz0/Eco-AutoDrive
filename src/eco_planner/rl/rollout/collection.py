"""Collect and persist one update-0 batch for independent offline diagnostics."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from eco_planner.artifacts import write_json
from eco_planner.jobs import write_job_configuration
from eco_planner.planning.policy import (
    policy_state_hash,
    save_exploration_policy_checkpoint,
)
from eco_planner.rl.artifacts.io import write_training_runtime_metadata
from eco_planner.rl.config import TrainingJobConfig

from ..setup import (
    configure_training_execution,
    create_training_collector,
    create_training_runtime,
)
from .fixed_batch import write_batch
from .seeds import create_rollout_streams, derive_rollout_seeds


def collect(resolved: DictConfig, config: TrainingJobConfig, output_dir: Path) -> dict[str, Any]:
    if config.resources is None:
        raise ValueError("diagnostic requires explicit resource profile")
    if config.training.resume_checkpoint_path is not None:
        raise ValueError("update-0 diagnostic cannot resume a trained policy")
    output_dir.mkdir(parents=True, exist_ok=False)
    write_job_configuration(resolved, output_dir)
    configure_training_execution(config)
    start = time.perf_counter()
    noise_seeds, policy_seeds = derive_rollout_seeds(config.runtime.seed, len(config.scenarios))
    runtime = create_training_runtime(config, policy_seeds[0])
    write_training_runtime_metadata(output_dir / "runtime_metadata.json", runtime, config.resources)
    initial_hash = policy_state_hash(runtime.policy)
    planner_hash = runtime.frozen_planner_hash()
    save_exploration_policy_checkpoint(output_dir / "policy-initial.pt", runtime.policy)
    print("Collecting one update-0 batch (no optimizer steps).", flush=True)
    with create_training_collector(config, runtime) as collector:
        batch = collector.collect(
            transitions_per_slot=config.training.transitions_per_environment,
            streams=create_rollout_streams(runtime, noise_seeds, policy_seeds),
        )
    episodes, sample_index = write_batch(output_dir, batch.episodes_by_slot, config.scenarios)
    if policy_state_hash(runtime.policy) != initial_hash:
        raise RuntimeError("initial policy changed")
    if runtime.frozen_planner_hash() != planner_hash:
        raise RuntimeError("frozen planner changed")
    if any(p.grad is not None for p in runtime.planner.parameters()):
        raise RuntimeError("diagnostic created planner gradients")
    summary = {
        "status": "completed",
        "kind": "fixed-batch",
        "optimizer_steps": 0,
        "sample_count": len(sample_index),
        "training_seed": config.runtime.seed,
        "noise_seeds": noise_seeds,
        "policy_action_seeds": policy_seeds,
        "initial_policy_hash": initial_hash,
        "final_policy_hash": initial_hash,
        "frozen_planner_hash": planner_hash,
        "policy_unchanged": True,
        "planner_unchanged": True,
        "planner_gradients_absent": True,
        "episode_count": len(episodes),
        "wall_time_s": time.perf_counter() - start,
        "batch_source": "new single update-0 collection",
    }
    write_json(output_dir / "summary.json", summary)
    return {
        "status": "completed",
        "output_dir": str(output_dir),
        "sample_count": len(sample_index),
        "optimizer_steps": 0,
    }
