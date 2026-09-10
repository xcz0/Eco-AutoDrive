"""Collect and persist one update-0 batch for independent offline diagnostics."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf

from eco_planner._repository import REPOSITORY_ROOT
from eco_planner.artifacts import write_json
from eco_planner.configuration import load_resolved_yaml_mapping
from eco_planner.experiments.scalar_reward.composition import compose_arm_training_config
from eco_planner.experiments.scalar_reward.config import load_scalar_reward_protocol
from eco_planner.rl import (
    PlannerRFTNoEnergyRewardConfig,
    VectorRolloutCollector,
    create_fabric_rollout_runtime,
    derive_rollout_seeds,
    policy_state_hash,
    save_exploration_policy_checkpoint,
    write_training_runtime_metadata,
)

from .artifacts import SHARED_SOURCES, copy_sources, write_batch
from .collection import CollectionConfig


def collect(config_path: Path, output_dir: Path) -> dict[str, Any]:
    study = CollectionConfig.model_validate(load_resolved_yaml_mapping(config_path))
    protocol = load_scalar_reward_protocol(Path(to_absolute_path(study.protocol)))
    resolved, config = compose_arm_training_config(
        protocol,
        "a1",
        study.training_seed,
        study.overrides,
    )
    if not isinstance(config.reward, PlannerRFTNoEnergyRewardConfig):
        raise ValueError("fixed batch collection requires the R0 profile")
    if config.resources is None:
        raise ValueError("diagnostic requires explicit resource profile")
    if config.training.resume_checkpoint_path is not None:
        raise ValueError("update-0 diagnostic cannot resume a trained policy")
    output_dir.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(resolved, output_dir / "resolved_config.yaml")
    OmegaConf.save(OmegaConf.create(study.model_dump()), output_dir / "collection_config.yaml")
    copy_sources(
        output_dir,
        (
            *SHARED_SOURCES,
            Path(__file__),
            Path(__file__).with_name("collection_config.py"),
            REPOSITORY_ROOT / "src/eco_planner/experiments/scalar_reward/composition.py",
            REPOSITORY_ROOT / "src/eco_planner/rl/rollout/seeds.py",
            REPOSITORY_ROOT / "scripts/experiments/__main__.py",
        ),
    )
    if config.training.deterministic:
        torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("high")
    start = time.perf_counter()
    noise_seeds, policy_seeds = derive_rollout_seeds(config.runtime.seed, len(config.scenarios))
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
    write_training_runtime_metadata(output_dir / "runtime_metadata.json", runtime, config.resources)
    initial_hash = policy_state_hash(runtime.policy)
    planner_hash = runtime.frozen_planner_hash()
    save_exploration_policy_checkpoint(output_dir / "policy-initial.pt", runtime.policy)
    print("Collecting one update-0 batch (no optimizer steps).", flush=True)
    with VectorRolloutCollector(
        config.scenarios,
        runtime,
        config.env,
        mode=config.training.mode,
        map_query_radius_m=config.map_query_radius_m,
        history_warmup_steps=config.training.history_warmup_steps,
        physical_slot_count=config.resources.rollout_worker_count,
        torch_threads_per_worker=config.resources.torch_threads_per_worker,
        reward_profile=config.reward,
    ) as collector:
        slots = collector.collect(
            transitions_per_slot=config.training.transitions_per_environment,
            stopped_speed_threshold_mps=config.training.stopped_speed_threshold_mps,
            diffusion_generators=tuple(runtime.new_noise_generator(s) for s in noise_seeds),
            policy_generators=tuple(runtime.new_policy_generator(s) for s in policy_seeds),
            noise_seeds=noise_seeds,
            policy_action_seeds=policy_seeds,
        )
    episodes, sample_index = write_batch(output_dir, slots, config.scenarios)
    if policy_state_hash(runtime.policy) != initial_hash:
        raise RuntimeError("initial policy changed")
    if runtime.frozen_planner_hash() != planner_hash:
        raise RuntimeError("frozen planner changed")
    if any(p.grad is not None for p in runtime._planner.parameters()):
        raise RuntimeError("diagnostic created planner gradients")
    summary = {
        "status": "completed",
        "kind": "fixed-batch",
        "optimizer_steps": 0,
        "sample_count": len(sample_index),
        "training_seed": study.training_seed,
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
