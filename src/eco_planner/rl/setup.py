"""Typed configuration factories shared by training and fixed-batch collection."""

from pathlib import Path

import torch
from hydra.utils import to_absolute_path

from eco_planner.runtime.resources import require_resource_profile

from .config import TrainingJobConfig
from .rollout.collector import VectorRolloutCollector
from .rollout.runtime import FabricRolloutRuntime, create_fabric_rollout_runtime


def configure_training_execution(config: TrainingJobConfig) -> None:
    if config.training.deterministic:
        torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("high")


def create_training_runtime(
    config: TrainingJobConfig,
    policy_action_seed: int,
) -> FabricRolloutRuntime:
    return create_fabric_rollout_runtime(
        config.runtime,
        config.sampler,
        config.guidance,
        config.policy,
        Path(to_absolute_path(config.model.args_path)),
        Path(to_absolute_path(config.model.checkpoint_path)),
        policy_action_seed,
        planner_compile_mode=config.training.planner_compile_mode,
    )


def create_training_collector(
    config: TrainingJobConfig,
    runtime: FabricRolloutRuntime,
) -> VectorRolloutCollector:
    resources = require_resource_profile(config.resources)
    return VectorRolloutCollector(
        config.scenarios,
        runtime,
        config.env,
        mode=config.training.mode,
        map_query_radius_m=config.map_query_radius_m,
        history_warmup_steps=config.training.history_warmup_steps,
        physical_slot_count=resources.rollout_worker_count,
        torch_threads_per_worker=resources.torch_threads_per_worker,
        reward_profile=config.reward,
    )
