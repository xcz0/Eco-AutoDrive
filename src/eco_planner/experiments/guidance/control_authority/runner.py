"""Collect manual-guidance interventions without constructing a policy or optimizer."""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import TensorDictBase

from eco_planner._repository import REPOSITORY_ROOT
from eco_planner.analysis.reporting.guidance import publish
from eco_planner.artifacts import collect_repository_metadata, write_json, write_npz
from eco_planner.configuration import load_resolved_yaml_mapping
from eco_planner.contracts import SIMULATOR_STEP_S, ExecutionMode
from eco_planner.envs.domain import TrajectoryExecutionResult
from eco_planner.evaluation.inference.runtime import (
    FabricInferenceRuntime,
    create_fabric_inference_runtime,
)
from eco_planner.experiments.guidance.control_authority.diagnostics import (
    InterventionConfig,
    analyze_episodes,
)
from eco_planner.experiments.reward.scalar.composition import compose_arm_training_config
from eco_planner.experiments.reward.scalar.config import load_scalar_reward_protocol
from eco_planner.rl.reward.components.energy import energy_score
from eco_planner.rl.reward.config import EnergyRewardConfig
from eco_planner.runtime.envs import (
    VectorEnvScenario,
    VectorMetaDriveEnv,
    WorkerResetResult,
    WorkerStepResult,
    operation_results,
)


def transition_record(
    result: TrajectoryExecutionResult,
    energy_config: EnergyRewardConfig,
    audit: TensorDictBase,
    slot: int,
) -> dict[str, Any]:
    if len(result.metrics) != 1:
        raise RuntimeError("Task D requires exactly one 0.1 s rollout transition")
    m = result.metrics[0]
    if m.input.timestep_s != SIMULATOR_STEP_S:
        raise RuntimeError("Task D timestep differs from rollout ABI")
    score, intensity, distance_valid = energy_score(energy_config, m)
    e = result.execution
    collision = any(
        getattr(e, name)
        for name in (
            "crash_vehicle",
            "crash_object",
            "crash_building",
            "crash_human",
            "crash_sidewalk",
        )
    )
    arrays = {
        name: audit[name][slot].numpy()
        for name in (
            "prediction",
            "reference_prediction",
            "longitudinal_target_speed_delta_mps",
            "applied_gradient_l2",
            "longitudinal_objective_delta",
        )
    }

    def along_speed(trajectory: np.ndarray) -> np.ndarray:
        ego = trajectory[0]
        tangent = ego[:, 2:4] / np.linalg.norm(ego[:, 2:4], axis=1, keepdims=True)
        velocity = np.diff(np.vstack((np.zeros((1, 2)), ego[:, :2])), axis=0) / SIMULATOR_STEP_S
        return np.sum(velocity * tangent, axis=1)

    guided = along_speed(arrays["prediction"])
    reference = along_speed(arrays["reference_prediction"])
    return {
        "dt_s": m.input.timestep_s,
        "speed_mps": m.speed_mps,
        "position_xy_m": list(m.input.position_xy_m),
        "heading_rad": m.input.heading_rad,
        "velocity_xy_mps": list(m.input.velocity_xy_mps),
        "distance_m": m.step_distance_m,
        "fuel_ml": m.energy.fuel_ml,
        "energy_ml_per_km": m.energy.fuel_ml_per_km,
        "energy_score": score,
        "score_intensity_ml_per_km": intensity,
        "energy_distance_valid": distance_valid,
        "progress_m": m.input.route_progress_delta_m,
        "acceleration_mps2": m.longitudinal_acceleration_mps2,
        "jerk_mps3": m.jerk_mps3,
        "position_error_m": m.position_error_m,
        "heading_error_rad": m.heading_error_rad,
        "collision": collision,
        "out_of_road": e.out_of_road,
        "terminated": result.terminated,
        "truncated": result.truncated,
        "arrive_dest": e.arrive_dest,
        "max_step": e.max_step,
        "planner_first_speed_mps": float(guided[0]),
        "planner_mean_speed_mps": float(guided.mean()),
        "reference_first_speed_mps": float(reference[0]),
        "reference_mean_speed_mps": float(reference.mean()),
        "target_first_delta_mps": float(arrays["longitudinal_target_speed_delta_mps"][0]),
        "target_mean_delta_mps": float(arrays["longitudinal_target_speed_delta_mps"].mean()),
        "applied_gradient_l2": arrays["applied_gradient_l2"].tolist(),
        "longitudinal_objective_delta": arrays["longitudinal_objective_delta"].tolist(),
    }


def collect_group(
    env: VectorMetaDriveEnv,
    runtime: FabricInferenceRuntime,
    scenarios: tuple[VectorEnvScenario, ...],
    noise_seed: int,
    config: InterventionConfig,
    energy_config: EnergyRewardConfig,
    output: Path,
    group: int,
) -> list[dict[str, Any]]:
    """Keep the inference batch fixed even after a terminal slot stops executing."""
    batch = len(scenarios)
    all_episodes: list[dict[str, Any]] = []
    baseline_observation: TensorDictBase | None = None
    baseline_state: list[np.ndarray] | None = None
    baseline_noises: list[np.ndarray] = []
    for arm, action in enumerate(config.longitudinal_actions):
        folder = output / "raw" / f"group-{group:03d}-arm-{arm}"
        folder.mkdir(parents=True)
        generators = tuple(
            torch.Generator(device=runtime.device).manual_seed(noise_seed) for _ in scenarios
        )
        resets = env.reset(scenarios)
        reset_records = operation_results(resets, WorkerResetResult)
        observation = cast(TensorDictBase, resets["observation"])
        states = [r.initial_state for r in reset_records]
        if baseline_observation is None:
            baseline_observation = observation.clone()
            baseline_state = [s.copy() for s in states]
        else:
            for key in observation.keys():
                if not torch.equal(observation[key], baseline_observation[key]):
                    raise RuntimeError(f"unmatched initial observation: {key}")
            if baseline_state is None or not all(
                np.array_equal(a, b) for a, b in zip(states, baseline_state, strict=True)
            ):
                raise RuntimeError("unmatched initial simulator state")
        write_npz(
            folder / "initial_observation.npz",
            {str(k): observation[k].numpy() for k in observation.keys()},
        )
        episodes = [
            {
                "id": f"group-{group:03d}-arm-{arm}-slot-{i}",
                "scenario": spec.name,
                "map": spec.map,
                "map_seed": spec.seed,
                "simulator_seed": spec.seed,
                "noise_seed": noise_seed,
                "physical_slot": i,
                "g_lon": action,
                "g_lat": config.lateral_action,
                "raw_dir": folder.relative_to(output).as_posix(),
                "initial_state": states[i].tolist(),
                "status": "running",
                "steps": [],
            }
            for i, spec in enumerate(scenarios)
        ]
        active = list(range(batch))
        actions = torch.tensor(
            [[config.lateral_action, action]] * batch, dtype=torch.float32, device=runtime.device
        )
        try:
            for cycle in range(config.window_steps):
                noise = runtime.sample_noise(generators)
                host_noise = noise.cpu().numpy()
                if arm == 0:
                    baseline_noises.append(host_noise.copy())
                elif not np.array_equal(host_noise, baseline_noises[cycle]):
                    raise RuntimeError("unmatched planner noise across intervention arms")
                decision = runtime.infer_batch(
                    observation, noise, generators, guidance_action=actions
                )
                trajectories = decision.ego_trajectories
                if not np.isfinite(trajectories).all() or np.any(
                    np.linalg.norm(trajectories[..., 2:4], axis=-1) == 0
                ):
                    raise RuntimeError("invalid planner trajectory")
                # Resolve once and retain full planner evidence, including inactive slots.
                audit = decision.audit_result()
                write_npz(
                    folder / f"cycle-{cycle:02d}.npz",
                    {str(k): audit[k].numpy() for k in audit.keys()},
                )
                if not active:
                    continue
                stepped = env.step(trajectories[active], slots=tuple(active))
                results = operation_results(stepped, WorkerStepResult)
                next_observation = cast(TensorDictBase, stepped["observation"])
                remaining = []
                for local, slot in enumerate(active):
                    result = results[local].step
                    episodes[slot]["steps"].append(
                        transition_record(result, energy_config, audit, slot)
                    )
                    if result.terminated or result.truncated:
                        episodes[slot]["status"] = "episode_terminated"
                    else:
                        observation[slot] = next_observation[local]
                        remaining.append(slot)
                active = remaining
            for slot in active:
                episodes[slot]["status"] = "window_complete"
        finally:
            # Preserve partial evidence on a hard error; never silently resume that run.
            write_json(folder / "episodes.json", {"episodes": episodes})
        all_episodes.extend(episodes)
    return all_episodes


def run(config_path: Path, output_dir: Path, *, figures: bool = True) -> dict[str, Any]:
    study = InterventionConfig.model_validate(load_resolved_yaml_mapping(config_path))
    protocol = load_scalar_reward_protocol(REPOSITORY_ROOT / study.protocol)
    resolved, job = compose_arm_training_config(protocol, "a1", study.runtime_seed, study.overrides)
    if job.resources is None:
        raise ValueError("Task D requires an explicit resource profile")
    if job.training.mode != "no_traffic" or not job.training.deterministic:
        raise ValueError("Task D requires deterministic no-traffic execution")
    if job.env["horizon"] <= study.window_steps:
        raise ValueError("environment horizon must exceed the fixed intervention window")
    if job.sampler.name != "ddim5" or job.sampler.ddim_stochasticity != 0:
        raise ValueError("Task D requires deterministic DDIM-5")
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
            "execution_mode": "rollout",
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
        torch_threads_per_worker=job.resources.torch_threads_per_worker,
    )
    episodes: list[dict[str, Any]] = []
    try:
        group = 0
        for offset in range(0, len(scenarios), workers):
            for seed in study.noise_seeds:
                batch = scenarios[offset : offset + workers]
                print(f"Group {group + 1}: {[s.name for s in batch]}, noise={seed}", flush=True)
                episodes.extend(
                    collect_group(
                        env, runtime, batch, seed, study, job.reward.energy, output_dir, group
                    )
                )
                write_json(output_dir / "episodes.json", {"episodes": episodes})
                group += 1
                print(
                    f"Saved {len(episodes)} episodes; elapsed {time.perf_counter() - started:.1f}s",
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
    return publish(output_dir, output_dir, figures=figures)


def save_decisions(
    episodes: list[dict[str, Any]],
    study: InterventionConfig,
    scenario_names: list[str],
    output_dir: Path,
) -> None:
    result = analyze_episodes(episodes, study, scenario_names)
    decisions = {"gate_d": result["gate_d"], "windows": {}}
    for window, metrics in result["windows"].items():
        decisions["windows"][window] = {
            metric: {
                **{
                    key: data[key]
                    for key in ("passed", "positive_pass_count", "negative_pass_count")
                },
                "scenarios": {
                    name: {"passed": row["passed"]} for name, row in data["scenarios"].items()
                },
            }
            for metric, data in metrics.items()
        }
    write_json(output_dir / "decisions.json", decisions)
