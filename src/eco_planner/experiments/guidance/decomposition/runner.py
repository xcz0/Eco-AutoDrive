"""Matched lon/lat guidance-constant decomposition without constructing a policy."""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from eco_planner._repository import REPOSITORY_ROOT
from eco_planner.analysis import publish
from eco_planner.artifacts import collect_repository_metadata, write_json
from eco_planner.configuration import load_resolved_yaml_mapping
from eco_planner.contracts import ExecutionMode
from eco_planner.evaluation import parse_evaluation_config
from eco_planner.evaluation.inference.runtime import (
    create_fabric_inference_runtime,
)
from eco_planner.evaluation.intervention import InterventionExecution, collect_group
from eco_planner.experiments.guidance.decomposition.diagnostics import (
    DecompositionConfig,
    SeedArms,
    analyze_episodes,
)
from eco_planner.experiments.protocol.composition import compose_arm_training_config
from eco_planner.experiments.protocol.config import load_protocol
from eco_planner.jobs import compose_job_config
from eco_planner.models import Ddim5SamplerConfig
from eco_planner.runtime.envs import (
    VectorEnvScenario,
    VectorMetaDriveEnv,
)
from eco_planner.runtime.resources import require_resource_profile


def run(config_path: Path, output_dir: Path, *, figures: bool = True) -> dict[str, Any]:
    study = DecompositionConfig.model_validate(load_resolved_yaml_mapping(config_path))
    protocol = load_protocol(REPOSITORY_ROOT / study.protocol)
    composed = compose_job_config(study.job, study.overrides)
    job = parse_evaluation_config(composed)
    resources = require_resource_profile(job.resources)
    if job.runtime.seed != study.runtime_seed:
        raise ValueError("decomposition runtime seed must match the composed job")
    if job.evaluation.mode != "no_traffic" or job.evaluation.history_warmup_steps != 0:
        raise ValueError("decomposition requires zero-warmup no-traffic execution")
    if not isinstance(job.sampler, Ddim5SamplerConfig) or job.sampler.ddim_stochasticity != 0.0:
        raise ValueError("decomposition requires deterministic DDIM-5")
    scenarios = tuple(VectorEnvScenario(s.name, s.map, s.seed) for s in job.scenarios)
    workers = resources.rollout_worker_count
    if len(scenarios) % workers:
        raise ValueError("scenario count must fill fixed worker batches")
    _, reward_job = compose_arm_training_config(protocol, "r0", study.training_seeds[0])
    cycles = job.evaluation.evaluated_horizon_steps
    output_dir.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(composed, output_dir / "resolved_config.yaml", resolve=True)
    write_json(output_dir / "intervention_config.json", study.model_dump())
    write_json(output_dir / "scenarios.json", {"scenarios": [asdict(s) for s in scenarios]})
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(resources.torch_threads_per_worker)
    started = time.perf_counter()
    runtime = create_fabric_inference_runtime(
        job.runtime,
        job.sampler,
        job.guidance,
        REPOSITORY_ROOT / job.model.args_path,
        REPOSITORY_ROOT / job.model.checkpoint_path,
    )
    write_json(
        output_dir / "runtime_metadata.json",
        {
            **collect_repository_metadata(REPOSITORY_ROOT),
            "runtime": asdict(runtime.report),
            "resources": resources.model_dump(),
            "checkpoint": asdict(runtime.checkpoint_report),
            "sampler": asdict(runtime.sampler_report),
            "policy_created": False,
            "optimizer_steps": 0,
            "execution_mode": "rollout",
            "cycles": cycles,
            "execution_steps": 1,
            "deterministic": True,
        },
    )
    env = VectorMetaDriveEnv(
        tuple({**job.env, "map": s.map} for s in scenarios[:workers]),
        mode="no_traffic",
        execution_mode=ExecutionMode.ROLLOUT,
        map_query_radius_m=job.map_query_radius_m,
        history_warmup_steps=0,
        scenarios=scenarios,
        torch_threads_per_worker=resources.torch_threads_per_worker,
    )
    episodes: list[dict[str, Any]] = []
    try:
        group = 0
        for offset in range(0, len(scenarios), workers):
            batch = scenarios[offset : offset + workers]
            for spec in study.seed_arms:
                actions = tuple((arm.lateral, arm.longitudinal) for arm in spec.arms)
                for seed in study.noise_seeds:
                    print(
                        f"Group {group + 1}: seed {spec.seed}: {[s.name for s in batch]}, "
                        f"noise={seed}",
                        flush=True,
                    )
                    rows = collect_group(
                        env,
                        runtime,
                        batch,
                        seed,
                        InterventionExecution(actions, cycles, 1),
                        reward_job.reward.energy,
                        output_dir,
                        group,
                    )
                    _annotate(rows, spec)
                    episodes.extend(rows)
                    write_json(output_dir / "episodes.json", {"episodes": episodes})
                    group += 1
                    print(
                        f"Saved {len(episodes)} episodes; "
                        f"elapsed {time.perf_counter() - started:.1f}s",
                        flush=True,
                    )
    finally:
        env.close()
    write_json(
        output_dir / "run.json",
        {
            "status": "completed",
            "wall_time_s": time.perf_counter() - started,
            "optimizer_steps": 0,
            "policy_created": False,
            "episode_count": len(episodes),
        },
    )
    save_decisions(episodes, study, [s.name for s in scenarios], output_dir)
    return publish("guidance-decomposition", output_dir, output_dir, figures=figures)


def _annotate(rows: list[dict[str, Any]], spec: SeedArms) -> None:
    by_action = {(arm.lateral, arm.longitudinal): arm.name for arm in spec.arms}
    for row in rows:
        row["training_seed"] = spec.seed
        row["arm"] = by_action[(row["g_lat"], row["g_lon"])]


def save_decisions(
    episodes: list[dict[str, Any]],
    study: DecompositionConfig,
    scenario_names: list[str],
    output_dir: Path,
) -> None:
    result = analyze_episodes(episodes, study, scenario_names)
    write_json(output_dir / "decisions.json", {"verdict": result["verdict"]})
