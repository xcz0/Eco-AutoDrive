"""Execution-horizon causal intervention without constructing a policy or optimizer."""

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
from eco_planner.evaluation.inference.runtime import (
    create_fabric_inference_runtime,
)
from eco_planner.evaluation.intervention import InterventionExecution, collect_group
from eco_planner.experiments.guidance.horizon.diagnostics import (
    HorizonInterventionConfig,
    analyze_episodes,
)
from eco_planner.experiments.protocol.composition import compose_arm_training_config
from eco_planner.experiments.protocol.config import load_protocol
from eco_planner.runtime.envs import (
    VectorEnvScenario,
    VectorMetaDriveEnv,
)


def run(config_path: Path, output_dir: Path, *, figures: bool = True) -> dict[str, Any]:
    study = HorizonInterventionConfig.model_validate(load_resolved_yaml_mapping(config_path))
    protocol = load_protocol(REPOSITORY_ROOT / study.protocol)
    resolved, job = compose_arm_training_config(protocol, "a1", study.runtime_seed, study.overrides)
    if job.resources is None:
        raise ValueError("horizon requires an explicit resource profile")
    if job.training.mode != "no_traffic" or not job.training.deterministic:
        raise ValueError("horizon requires deterministic no-traffic execution")
    if job.env["horizon"] <= study.total_window_steps:
        raise ValueError("environment horizon must exceed the fixed intervention window")
    if job.sampler.name != "ddim5" or job.sampler.ddim_stochasticity != 0:
        raise ValueError("horizon requires deterministic DDIM-5")
    scenarios = tuple(VectorEnvScenario(s.name, s.map, s.seed) for s in job.scenarios)
    workers = job.resources.rollout_worker_count
    if len(scenarios) % workers or study.required_scenarios > len(scenarios):
        raise ValueError("scenario count must fill fixed batches and support the majority gate")
    output_dir.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(resolved, output_dir / "resolved_config.yaml", resolve=True)
    write_json(output_dir / "intervention_config.json", study.model_dump())
    write_json(output_dir / "scenarios.json", {"scenarios": [asdict(s) for s in scenarios]})
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(job.resources.torch_threads_per_worker)
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
            "resources": job.resources.model_dump(),
            "checkpoint": asdict(runtime.checkpoint_report),
            "sampler": asdict(runtime.sampler_report),
            "policy_created": False,
            "optimizer_steps": 0,
            "execution_steps": list(study.execution_horizons),
            "execution_horizons": list(study.execution_horizons),
            "deterministic": True,
        },
    )
    episodes: list[dict[str, Any]] = []
    group = 0
    for horizon in study.execution_horizons:
        env = VectorMetaDriveEnv(
            tuple({**job.env, "map": s.map} for s in scenarios[:workers]),
            mode="no_traffic",
            map_query_radius_m=job.map_query_radius_m,
            history_warmup_steps=0,
            scenarios=scenarios,
            torch_threads_per_worker=job.resources.torch_threads_per_worker,
            execution_steps=horizon,
        )
        try:
            for offset in range(0, len(scenarios), workers):
                for seed in study.noise_seeds:
                    batch = scenarios[offset : offset + workers]
                    print(
                        f"Horizon {horizon}: group {group + 1}: "
                        f"{[s.name for s in batch]}, noise={seed}",
                        flush=True,
                    )
                    episodes.extend(
                        collect_group(
                            env,
                            runtime,
                            batch,
                            seed,
                            InterventionExecution(
                                tuple(
                                    (study.lateral_action, action)
                                    for action in study.longitudinal_actions
                                ),
                                study.total_window_steps // horizon,
                                horizon,
                            ),
                            job.reward.energy,
                            output_dir,
                            group,
                        )
                    )
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
    return publish("guidance-horizon", output_dir, output_dir, figures=figures)


def save_decisions(
    episodes: list[dict[str, Any]],
    study: HorizonInterventionConfig,
    scenario_names: list[str],
    output_dir: Path,
) -> None:
    result = analyze_episodes(episodes, study, scenario_names)
    decisions: dict[str, Any] = {
        "gate_a": result["gate_a"],
        "horizons": {},
        "planner_response": {},
    }
    for horizon, data in result["horizons"].items():
        decisions["horizons"][horizon] = {
            "direction": data["direction"],
            "metrics": {
                metric: {
                    **{
                        key: metrics[key]
                        for key in (
                            "passed",
                            "positive_pass_count",
                            "negative_pass_count",
                            "direction",
                        )
                    },
                    "scenarios": {
                        name: {"passed": row["passed"]}
                        for name, row in metrics["scenarios"].items()
                    },
                }
                for metric, metrics in data["metrics"].items()
            },
        }
    for checkpoint, response in result["planner_response"].items():
        decisions["planner_response"][checkpoint] = {
            "passed": response["passed"],
            "positive_pass_count": response["positive_pass_count"],
            "negative_pass_count": response["negative_pass_count"],
            "direction": response["direction"],
            "scenarios": {
                name: {"passed": row["passed"]} for name, row in response["scenarios"].items()
            },
        }
    write_json(output_dir / "decisions.json", decisions)
