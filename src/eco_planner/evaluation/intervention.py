"""Collect manual-guidance interventions without constructing a policy or optimizer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import torch
from tensordict import TensorDictBase

from eco_planner.artifacts import write_json, write_npz
from eco_planner.contracts import SIMULATOR_STEP_S
from eco_planner.envs.domain import (
    TrajectoryExecutionRecord,
    TransitionMetrics,
)
from eco_planner.evaluation.inference import DiffusionEvaluationAgent
from eco_planner.planning import DiffusionRuntime
from eco_planner.reward.components.energy import EnergyRewardConfig, energy_score
from eco_planner.runtime.envs import (
    VectorEnvScenario,
    VectorMetaDriveEnv,
    WorkerResetResult,
    WorkerStepResult,
    operation_results,
)

PLANNER_RESPONSE_CHECKPOINT_STEPS: Final = (1, 2, 5, 10, 20, 40, 80)


@dataclass(frozen=True)
class InterventionExecution:
    """Per-arm manual guidance actions ordered ``(lateral, longitudinal)``."""

    arm_actions: tuple[tuple[float, float], ...]
    cycles: int
    execution_steps: int


def _along_speed(trajectory: np.ndarray) -> np.ndarray:
    ego = trajectory[0]
    tangent = ego[:, 2:4] / np.linalg.norm(ego[:, 2:4], axis=1, keepdims=True)
    velocity = np.diff(np.vstack((np.zeros((1, 2)), ego[:, :2])), axis=0) / SIMULATOR_STEP_S
    return np.sum(velocity * tangent, axis=1)


def planner_cycle_record(
    audit: TensorDictBase, slot: int, *, include_waypoints: bool = False
) -> dict[str, Any]:
    """Summarize one plan's predicted forward response at fixed checkpoints."""
    guided = _along_speed(audit["prediction"][slot].numpy())
    displacement = np.cumsum(guided * SIMULATOR_STEP_S)
    record = {
        "checkpoint_steps": list(PLANNER_RESPONSE_CHECKPOINT_STEPS),
        "forward_displacement_m": [
            float(displacement[step - 1]) for step in PLANNER_RESPONSE_CHECKPOINT_STEPS
        ],
        "first_speed_mps": float(guided[0]),
        "mean_speed_mps": float(guided.mean()),
    }
    if include_waypoints:
        record["waypoint_forward_displacement_m"] = [float(value) for value in displacement]
    return record


def transition_record(
    metrics: TransitionMetrics,
    execution: TrajectoryExecutionRecord,
    energy_config: EnergyRewardConfig,
    audit: TensorDictBase,
    slot: int,
    *,
    plan_cycle: int,
    substep: int,
) -> dict[str, Any]:
    m = metrics
    if m.input.timestep_s != SIMULATOR_STEP_S:
        raise RuntimeError("guidance intervention timestep differs from rollout ABI")
    score, intensity, distance_valid = energy_score(energy_config, m)
    e = execution
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
    guided = _along_speed(audit["prediction"][slot].numpy())
    reference = _along_speed(audit["reference_prediction"][slot].numpy())
    target_delta = audit["longitudinal_target_speed_delta_mps"][slot].numpy()
    return {
        "dt_s": m.input.timestep_s,
        "plan_cycle": plan_cycle,
        "substep": substep,
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
        "terminated": bool(e.substep_terminated[substep]),
        "truncated": bool(e.substep_truncated[substep]),
        "arrive_dest": e.arrive_dest,
        "route_completion": e.route_completion,
        "max_step": e.max_step,
        "planner_first_speed_mps": float(guided[0]),
        "planner_mean_speed_mps": float(guided.mean()),
        "reference_first_speed_mps": float(reference[0]),
        "reference_mean_speed_mps": float(reference.mean()),
        "target_first_delta_mps": float(target_delta[0]),
        "target_mean_delta_mps": float(target_delta.mean()),
        "applied_gradient_l2": audit["applied_gradient_l2"][slot].numpy().tolist(),
        "longitudinal_objective_delta": (
            audit["longitudinal_objective_delta"][slot].numpy().tolist()
        ),
    }


def collect_group(
    env: VectorMetaDriveEnv,
    runtime: DiffusionRuntime,
    scenarios: tuple[VectorEnvScenario, ...],
    noise_seed: int,
    config: InterventionExecution,
    energy_config: EnergyRewardConfig,
    output: Path,
    group: int,
    *,
    include_waypoints: bool = False,
) -> list[dict[str, Any]]:
    """Keep the inference batch fixed even after a terminal slot stops executing."""
    agent = DiffusionEvaluationAgent(runtime)
    batch = len(scenarios)
    all_episodes: list[dict[str, Any]] = []
    baseline_observation: TensorDictBase | None = None
    baseline_state: list[np.ndarray] | None = None
    baseline_noises: list[np.ndarray] = []
    for arm, (lateral_action, longitudinal_action) in enumerate(config.arm_actions):
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
                "g_lat": lateral_action,
                "g_lon": longitudinal_action,
                "execution_horizon": config.execution_steps,
                "cycles": config.cycles,
                "raw_dir": folder.relative_to(output).as_posix(),
                "initial_state": states[i].tolist(),
                "status": "running",
                "steps": [],
                "planner_cycles": [],
            }
            for i, spec in enumerate(scenarios)
        ]
        active = list(range(batch))
        actions = torch.tensor(
            [[lateral_action, longitudinal_action]] * batch,
            dtype=torch.float32,
            device=runtime.device,
        )
        try:
            for cycle in range(config.cycles):
                noise = runtime.sample_noise(generators)
                host_noise = noise.cpu().numpy()
                if arm == 0:
                    baseline_noises.append(host_noise.copy())
                elif not np.array_equal(host_noise, baseline_noises[cycle]):
                    raise RuntimeError("unmatched planner noise across intervention arms")
                decision = agent.infer_batch(
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
                    episodes[slot]["planner_cycles"].append(
                        {
                            "plan_cycle": cycle,
                            **planner_cycle_record(
                                audit, slot, include_waypoints=include_waypoints
                            ),
                        }
                    )
                    for substep, metric in enumerate(result.metrics):
                        episodes[slot]["steps"].append(
                            transition_record(
                                metric,
                                result.execution,
                                energy_config,
                                audit,
                                slot,
                                plan_cycle=cycle,
                                substep=substep,
                            )
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
