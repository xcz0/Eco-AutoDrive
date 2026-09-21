"""Matched frozen-policy interventions with an explicit execution prefix.

This is the policy-driven sibling of :mod:`eco_planner.evaluation.intervention`.
It keeps the same matched-group semantics (identical reset state, physical slot,
batch shape, and per-cycle diffusion noise across arms) but builds each arm's
guidance action from a frozen exploration-policy checkpoint instead of a manual
constant, and additionally records the same-state counterfactual response needed
to bridge the learned policy difference to the planner's temporal response.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from tensordict import TensorDictBase

from eco_planner.artifacts import write_json, write_npz
from eco_planner.contracts import SIMULATOR_STEP_S
from eco_planner.evaluation.inference.decision import prepare_learned_inference_decision
from eco_planner.evaluation.intervention import (
    PLANNER_RESPONSE_CHECKPOINT_STEPS,
    planner_cycle_record,
    transition_record,
)
from eco_planner.planning import PolicyGuidanceDecisionResult, PolicyGuidanceRuntime
from eco_planner.reward.config import EnergyRewardConfig
from eco_planner.runtime.envs import (
    VectorEnvScenario,
    VectorMetaDriveEnv,
    WorkerResetResult,
    WorkerStepResult,
    operation_results,
)
from eco_planner.runtime.host_transfer import HostTransfer

_PLANNER_AUDIT_KEYS = (
    "prediction",
    "reference_prediction",
    "guidance_action",
    "initial_noise",
    "longitudinal_target_speed_delta_mps",
    "applied_gradient_l2",
    "longitudinal_objective_delta",
)

ArmMetadata = dict[str, Any]


def _prediction_response(prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return cumulative forward and lateral displacement of one predicted ego path."""
    ego = prediction[0]
    heading = ego[:, 2:4] / np.linalg.norm(ego[:, 2:4], axis=1, keepdims=True)
    normal = np.stack((-heading[:, 1], heading[:, 0]), axis=1)
    velocity = np.diff(np.vstack((np.zeros((1, 2)), ego[:, :2])), axis=0) / SIMULATOR_STEP_S
    forward = np.cumsum(np.sum(velocity * heading, axis=1) * SIMULATOR_STEP_S)
    lateral = np.cumsum(np.sum(velocity * normal, axis=1) * SIMULATOR_STEP_S)
    return forward, lateral


def _checkpoint_indices() -> list[int]:
    return [step - 1 for step in PLANNER_RESPONSE_CHECKPOINT_STEPS]


def _mean_decision(
    runtime: PolicyGuidanceRuntime,
    observation: TensorDictBase,
    generators: Sequence[torch.Generator],
) -> Any:
    """Run one deterministic guidance decision and adapt it to host audit tensors."""

    result: PolicyGuidanceDecisionResult = runtime.decide_batch_mean(observation, generators)
    return prepare_learned_inference_decision(result, HostTransfer(runtime.device))


def collect_policy_pair(
    env: VectorMetaDriveEnv,
    runtime: PolicyGuidanceRuntime,
    scenarios: tuple[VectorEnvScenario, ...],
    noise_seed: int,
    cycles: int,
    execution_steps: int,
    energy_config: EnergyRewardConfig,
    output: Path,
    group: int,
    arm_runs: Sequence[ArmMetadata],
    load_policy: Callable[[int], None],
    *,
    include_waypoints: bool = False,
) -> list[dict[str, Any]]:
    """Collect matched frozen-policy arms under one execution prefix.

    ``arm_runs`` declares the ordered arms (``arm``, ``policy_label``,
    ``policy_hash``); ``load_policy(index)`` loads that arm's checkpoint into the
    shared runtime. The inference batch and noise draws stay fixed even after a
    terminal slot stops executing, so every arm sees the same planner noise.
    """
    batch = len(scenarios)
    all_episodes: list[dict[str, Any]] = []
    baseline_observation: TensorDictBase | None = None
    baseline_state: list[np.ndarray] | None = None
    baseline_noises: list[np.ndarray] = []
    for arm_index, arm_run in enumerate(arm_runs):
        load_policy(arm_index)
        folder = output / "raw" / f"group-{group:03d}-arm-{arm_index}"
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
                "id": f"group-{group:03d}-arm-{arm_index}-slot-{i}",
                "scenario": spec.name,
                "map": spec.map,
                "map_seed": spec.seed,
                "simulator_seed": spec.seed,
                "noise_seed": noise_seed,
                "physical_slot": i,
                "training_seed": arm_run["training_seed"],
                "arm": arm_run["arm"],
                "policy_label": arm_run["policy_label"],
                "policy_hash": arm_run["policy_hash"],
                "execution_horizon": execution_steps,
                "cycles": cycles,
                "raw_dir": folder.relative_to(output).as_posix(),
                "initial_state": states[i].tolist(),
                "status": "running",
                "steps": [],
                "planner_cycles": [],
            }
            for i, spec in enumerate(scenarios)
        ]
        active = list(range(batch))
        try:
            for cycle in range(cycles):
                decision = _mean_decision(runtime, observation, generators)
                audit = decision.audit_result()
                host_noise = audit["initial_noise"].numpy()
                if arm_index == 0:
                    baseline_noises.append(host_noise.copy())
                elif not np.array_equal(host_noise, baseline_noises[cycle]):
                    raise RuntimeError("unmatched planner noise across policy arms")
                trajectories = decision.ego_trajectories
                if not np.isfinite(trajectories).all() or np.any(
                    np.linalg.norm(trajectories[..., 2:4], axis=-1) == 0
                ):
                    raise RuntimeError("invalid planner trajectory")
                write_npz(
                    folder / f"cycle-{cycle:02d}.npz",
                    {key: audit[key].numpy() for key in _PLANNER_AUDIT_KEYS},
                )
                if not active:
                    continue
                stepped = env.step(trajectories[active], slots=tuple(active))
                results = operation_results(stepped, WorkerStepResult)
                next_observation = cast(TensorDictBase, stepped["observation"])
                remaining = []
                for local, slot in enumerate(active):
                    result = results[local].step
                    guidance = audit["guidance_action"][slot].numpy().tolist()
                    episodes[slot]["planner_cycles"].append(
                        {
                            "plan_cycle": cycle,
                            "guidance_action": guidance,
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
            write_json(folder / "episodes.json", {"episodes": episodes})
        all_episodes.extend(episodes)
    return all_episodes


def _audit_contexts(
    runtime: PolicyGuidanceRuntime,
    contexts_by_cycle: list[list[dict[str, Any]]],
    noise_seed: int,
) -> dict[tuple[int, int], dict[str, Any]]:
    """Run one frozen policy over saved same-state contexts with deterministic noise."""
    responses: dict[tuple[int, int], dict[str, Any]] = {}
    checkpoint_indices = _checkpoint_indices()
    for cycle, contexts in enumerate(contexts_by_cycle):
        if not contexts:
            continue
        observation = cast(
            TensorDictBase, TensorDictBase.stack([context["observation"] for context in contexts])
        )
        generators = tuple(
            torch.Generator(device=runtime.device).manual_seed(noise_seed) for _ in contexts
        )
        decision = _mean_decision(runtime, observation, generators)
        audit = decision.audit_result()
        for index, context in enumerate(contexts):
            forward, lateral = _prediction_response(audit["prediction"][index].numpy())
            responses[cycle, context["slot"]] = {
                "forward_displacement_m": forward.tolist(),
                "lateral_displacement_m": [float(lateral[i]) for i in checkpoint_indices],
                "guidance_action": audit["guidance_action"][index].numpy().tolist(),
            }
    return responses


def collect_same_state_audit(
    env: VectorMetaDriveEnv,
    runtime: PolicyGuidanceRuntime,
    scenarios: tuple[VectorEnvScenario, ...],
    noise_seed: int,
    cycles: int,
    execution_steps: int,
    output: Path,
    group: int,
    arm_runs: Sequence[ArmMetadata],
    load_policy: Callable[[int], None],
    *,
    reference_index: int,
    counterfactual_index: int,
) -> list[dict[str, Any]]:
    """Collect same-state counterfactual planner responses for two frozen policies.

    The environment is stepped by the reference policy only; every saved context
    is then evaluated by both policies with identical observation, planner, and
    diffusion noise, so only the policy-produced guidance differs.
    """
    batch = len(scenarios)
    load_policy(reference_index)
    generators = tuple(
        torch.Generator(device=runtime.device).manual_seed(noise_seed) for _ in scenarios
    )
    resets = env.reset(scenarios)
    observation = cast(TensorDictBase, resets["observation"])
    active = list(range(batch))
    contexts_by_cycle: list[list[dict[str, Any]]] = []
    for cycle in range(cycles):
        decision = _mean_decision(runtime, observation, generators)
        trajectories = decision.ego_trajectories
        contexts_by_cycle.append(
            [
                {
                    "slot": slot,
                    "cycle": cycle,
                    "observation": observation[slot].detach().cpu().clone(),
                }
                for slot in active
            ]
        )
        if not active:
            continue
        stepped = env.step(trajectories[active], slots=tuple(active))
        results = operation_results(stepped, WorkerStepResult)
        next_observation = cast(TensorDictBase, stepped["observation"])
        remaining = []
        for local, slot in enumerate(active):
            result = results[local].step
            if result.terminated or result.truncated:
                continue
            observation[slot] = next_observation[local]
            remaining.append(slot)
        active = remaining

    load_policy(reference_index)
    reference = _audit_contexts(runtime, contexts_by_cycle, noise_seed)
    load_policy(counterfactual_index)
    counterfactual = _audit_contexts(runtime, contexts_by_cycle, noise_seed)

    records: list[dict[str, Any]] = []
    for contexts in contexts_by_cycle:
        for context in contexts:
            key = (context["cycle"], context["slot"])
            spec = scenarios[context["slot"]]
            ref = reference[key]
            cf = counterfactual[key]
            records.append(
                {
                    "training_seed": arm_runs[reference_index]["training_seed"],
                    "scenario": spec.name,
                    "map": spec.map,
                    "map_seed": spec.seed,
                    "cycle": context["cycle"],
                    "reference_arm": arm_runs[reference_index]["arm"],
                    "counterfactual_arm": arm_runs[counterfactual_index]["arm"],
                    "reference_label": arm_runs[reference_index]["policy_label"],
                    "counterfactual_label": arm_runs[counterfactual_index]["policy_label"],
                    "reference_policy_hash": arm_runs[reference_index]["policy_hash"],
                    "counterfactual_policy_hash": arm_runs[counterfactual_index]["policy_hash"],
                    "g_lat_reference": ref["guidance_action"][0],
                    "g_lon_reference": ref["guidance_action"][1],
                    "g_lat_counterfactual": cf["guidance_action"][0],
                    "g_lon_counterfactual": cf["guidance_action"][1],
                    "forward_reference_m": ref["forward_displacement_m"],
                    "forward_counterfactual_m": cf["forward_displacement_m"],
                    "lateral_reference_m": ref["lateral_displacement_m"],
                    "lateral_counterfactual_m": cf["lateral_displacement_m"],
                }
            )
    write_json(output / f"same-state-group-{group:03d}.json", {"contexts": records})
    return records
