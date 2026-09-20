"""Frozen-policy execution-contract bridge without training or reward changes.

Part A replays the E-040 final R0/Rstress checkpoints in a matched closed loop
across explicit execution prefixes; Part B audits both policies' local guidance
actions on identical held-out contexts. The two artifacts connect the learned
policy difference to the guidance temporal-response mechanism found in E-043/E-045.
"""

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
from eco_planner.evaluation.policy_intervention import (
    collect_policy_pair,
    collect_same_state_audit,
)
from eco_planner.experiments.guidance.execution_bridge.diagnostics import (
    ExecutionBridgeConfig,
    analyze_episodes,
    arm_runs,
)
from eco_planner.experiments.protocol.composition import compose_arm_training_config
from eco_planner.experiments.protocol.config import load_protocol
from eco_planner.jobs import compose_job_config
from eco_planner.planning import create_policy_guidance_runtime
from eco_planner.planning.diffusion import Ddim5SamplerConfig, OrthogonalPolicyGuidanceConfig
from eco_planner.planning.policy import (
    load_exploration_policy_checkpoint,
    policy_state_hash,
)
from eco_planner.runtime.envs import VectorEnvScenario, VectorMetaDriveEnv
from eco_planner.runtime.resources import require_resource_profile


def run(
    source: Path, config_path: Path, output_dir: Path, *, figures: bool = True
) -> dict[str, Any]:
    study = ExecutionBridgeConfig.model_validate(load_resolved_yaml_mapping(config_path))
    protocol = load_protocol(REPOSITORY_ROOT / study.protocol)
    composed = compose_job_config(study.job, study.overrides)
    job = parse_evaluation_config(composed)
    resources = require_resource_profile(job.resources)
    if job.runtime.seed != study.runtime_seed:
        raise ValueError("bridge runtime seed must match the composed job")
    if job.evaluation.mode != "no_traffic" or job.evaluation.history_warmup_steps != 0:
        raise ValueError("bridge requires zero-warmup no-traffic execution")
    if not isinstance(job.sampler, Ddim5SamplerConfig) or job.sampler.ddim_stochasticity != 0.0:
        raise ValueError("bridge requires deterministic DDIM-5")
    if not isinstance(job.guidance, OrthogonalPolicyGuidanceConfig):
        raise ValueError("bridge requires orthogonal_policy guidance")
    horizon = job.evaluation.evaluated_horizon_steps
    for execution_horizon in study.execution_horizons:
        if horizon % execution_horizon:
            raise ValueError("execution horizons must divide the evaluated horizon")
    if horizon % study.same_state_contract_steps:
        raise ValueError("same-state contract steps must divide the evaluated horizon")
    scenarios = tuple(VectorEnvScenario(s.name, s.map, s.seed) for s in job.scenarios)
    workers = resources.rollout_worker_count
    if len(scenarios) % workers or study.required_scenarios > len(scenarios):
        raise ValueError("scenario count must fill fixed batches and support the majority gate")

    training_seeds = sorted({run.training_seed for run in study.runs})
    _, reference_training = compose_arm_training_config(protocol, "r0", training_seeds[0])
    if reference_training.policy is None:
        raise ValueError("bridge requires the trained-arm policy architecture")
    checkpoint_meta: dict[str, dict[str, Any]] = {}
    for seed in training_seeds:
        reference, counterfactual = arm_runs(study, seed)
        for run_config in (reference, counterfactual):
            path = (source / run_config.path).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"missing frozen checkpoint: {path}")
            _, training = compose_arm_training_config(protocol, run_config.arm, seed)
            if training.policy != reference_training.policy:
                raise ValueError("bridge arms must share one policy architecture")
            checkpoint_meta[run_config.label] = {
                "arm": run_config.arm,
                "training_seed": seed,
                "path": str(path),
                "reward_profile": run_config.reward_profile,
            }
    output_dir.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(composed, output_dir / "resolved_config.yaml", resolve=True)
    write_json(
        output_dir / "intervention_config.json",
        {**study.model_dump(), "checkpoints": checkpoint_meta, "evaluated_horizon_steps": horizon},
    )
    write_json(output_dir / "scenarios.json", {"scenarios": [asdict(s) for s in scenarios]})
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(resources.torch_threads_per_worker)
    started = time.perf_counter()
    args_path = REPOSITORY_ROOT / job.model.args_path
    model_path = REPOSITORY_ROOT / job.model.checkpoint_path
    runtime = create_policy_guidance_runtime(
        job.runtime,
        job.sampler,
        job.guidance,
        reference_training.policy,
        args_path,
        model_path,
        study.runtime_seed,
        planner_compile_mode="eager",
    )

    loaded_hashes: dict[str, str] = {}

    def load_policy(label: str) -> None:
        checkpoint = Path(checkpoint_meta[label]["path"])
        load_exploration_policy_checkpoint(checkpoint, runtime.policy)
        digest = policy_state_hash(runtime.policy)
        recorded = loaded_hashes.setdefault(label, digest)
        if recorded != digest:
            raise RuntimeError(f"policy checkpoint {label!r} changed between loads")

    for seed in training_seeds:
        reference, counterfactual = arm_runs(study, seed)
        for run_config in (reference, counterfactual):
            load_policy(run_config.label)
    energy_configs = {seed: _energy_config(protocol, seed) for seed in training_seeds}

    write_json(
        output_dir / "runtime_metadata.json",
        {
            **collect_repository_metadata(REPOSITORY_ROOT),
            "runtime": asdict(runtime.report),
            "resources": resources.model_dump(),
            "checkpoint": asdict(runtime.checkpoint_report),
            "sampler": asdict(runtime.sampler_report),
            "policy_created": True,
            "optimizer_steps": 0,
            "execution_mode": "rollout",
            "execution_horizons": list(study.execution_horizons),
            "same_state_contract_steps": study.same_state_contract_steps,
            "reference_arm": study.reference_arm,
            "counterfactual_arm": study.counterfactual_arm,
            "policy_hashes": loaded_hashes,
            "deterministic": True,
        },
    )

    episodes: list[dict[str, Any]] = []
    group = 0
    for execution_horizon in study.execution_horizons:
        env = _make_env(job, scenarios, workers, resources, execution_horizon)
        try:
            for offset in range(0, len(scenarios), workers):
                batch = scenarios[offset : offset + workers]
                for seed in training_seeds:
                    reference, counterfactual = arm_runs(study, seed)
                    arm_runs_meta = _arm_metadata(
                        study, loaded_hashes, seed, reference, counterfactual
                    )
                    for noise_seed in study.noise_seeds:
                        print(
                            f"Part A horizon={execution_horizon} seed={seed} "
                            f"group={group + 1}: {[s.name for s in batch]}, noise={noise_seed}",
                            flush=True,
                        )
                        rows = collect_policy_pair(
                            env,
                            runtime,
                            batch,
                            noise_seed,
                            horizon // execution_horizon,
                            execution_horizon,
                            energy_configs[seed],
                            output_dir,
                            group,
                            arm_runs_meta,
                            lambda index, meta=arm_runs_meta: load_policy(
                                meta[index]["policy_label"]
                            ),
                        )
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

    contexts: list[dict[str, Any]] = []
    if study.include_same_state_audit:
        env = _make_env(job, scenarios, workers, resources, study.same_state_contract_steps)
        try:
            for offset in range(0, len(scenarios), workers):
                batch = scenarios[offset : offset + workers]
                for seed in training_seeds:
                    reference, counterfactual = arm_runs(study, seed)
                    arm_runs_meta = _arm_metadata(
                        study, loaded_hashes, seed, reference, counterfactual
                    )
                    print(
                        f"Part B seed={seed} group={group + 1}: {[s.name for s in batch]}",
                        flush=True,
                    )
                    contexts.extend(
                        collect_same_state_audit(
                            env,
                            runtime,
                            batch,
                            study.noise_seeds[0],
                            horizon // study.same_state_contract_steps,
                            study.same_state_contract_steps,
                            output_dir,
                            group,
                            arm_runs_meta,
                            lambda index, meta=arm_runs_meta: load_policy(
                                meta[index]["policy_label"]
                            ),
                            reference_index=0,
                            counterfactual_index=1,
                        )
                    )
                    write_json(output_dir / "same_state.json", {"contexts": contexts})
                    group += 1
                    print(
                        f"Saved {len(contexts)} same-state contexts; "
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
            "policy_created": True,
            "episode_count": len(episodes),
            "same_state_context_count": len(contexts),
        },
    )
    save_decisions(episodes, contexts, study, [s.name for s in scenarios], output_dir)
    return publish("guidance-execution-bridge", output_dir, output_dir, figures=figures)


def _make_env(
    job: Any,
    scenarios: tuple[VectorEnvScenario, ...],
    workers: int,
    resources: Any,
    execution_steps: int,
) -> VectorMetaDriveEnv:
    return VectorMetaDriveEnv(
        tuple({**job.env, "map": s.map} for s in scenarios[:workers]),
        mode="no_traffic",
        execution_mode=ExecutionMode.ROLLOUT,
        map_query_radius_m=job.map_query_radius_m,
        history_warmup_steps=0,
        scenarios=scenarios,
        torch_threads_per_worker=resources.torch_threads_per_worker,
        execution_steps=execution_steps,
    )


def _energy_config(protocol: Any, training_seed: int) -> Any:
    _, training = compose_arm_training_config(protocol, "r0", training_seed)
    return training.reward.energy


def _arm_metadata(
    study: ExecutionBridgeConfig,
    hashes: dict[str, str],
    training_seed: int,
    reference: Any,
    counterfactual: Any,
) -> list[dict[str, Any]]:
    return [
        {
            "arm": run_config.arm,
            "policy_label": run_config.label,
            "policy_hash": hashes[run_config.label],
            "training_seed": training_seed,
        }
        for run_config in (reference, counterfactual)
    ]


def save_decisions(
    episodes: list[dict[str, Any]],
    contexts: list[dict[str, Any]],
    study: ExecutionBridgeConfig,
    scenario_names: list[str],
    output_dir: Path,
) -> None:
    result = analyze_episodes(episodes, contexts, study, scenario_names)
    write_json(output_dir / "decisions.json", {"gate": result["gate"]})
