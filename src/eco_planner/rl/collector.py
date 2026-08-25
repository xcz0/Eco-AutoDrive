"""Closed-loop rollout collectors for serial and fixed-slot vector execution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import Literal
from weakref import finalize

import numpy as np
import torch
from tensordict import TensorDictBase

from eco_planner.envs import (
    MetaDriveEnvSlot,
    PlannerObservationSpec,
    TrajectoryExecutionRecord,
    VectorEnvScenario,
    VectorMetaDriveEnv,
    collate_observations,
)
from eco_planner.evaluation.config import ScenarioConfig
from eco_planner.rl.rollout import (
    RolloutEpisode,
    build_rollout_audit,
    build_training_transition,
    finalize_rollout_episode,
    set_training_transition_next_state_value,
)
from eco_planner.rl.runtime import FabricRolloutRuntime


@dataclass(frozen=True)
class VectorRolloutRoundTiming:
    """One fixed-slot collection round, exposed for throughput profiling only."""

    phase: Literal["decision", "bootstrap"]
    active_slots: int
    capacity: int
    planner_wall_s: float
    environment_wall_s: float
    worker_busy_s: float
    worker_wait_s: float
    worker_imbalance_s: float


def collect_rollout_episode(
    spec: ScenarioConfig,
    runtime: FabricRolloutRuntime,
    env_config: Mapping[str, object],
    *,
    mode: Literal["no_traffic", "traffic"],
    map_query_radius_m: float,
    history_warmup_steps: int,
    max_transitions: int,
    stopped_speed_threshold_mps: float = 0.1,
    diffusion_generator: torch.Generator | None = None,
    policy_generator: torch.Generator | None = None,
    noise_seed: int | None = None,
    policy_action_seed: int | None = None,
) -> RolloutEpisode:
    """Collect one bounded episode without reset, batching, artifacts, or policy updates."""

    if type(max_transitions) is not int or max_transitions <= 0:
        raise ValueError("max_transitions must be a positive integer")
    if type(history_warmup_steps) is not int or history_warmup_steps < 0:
        raise ValueError("history_warmup_steps must be a non-negative integer")
    if (
        type(stopped_speed_threshold_mps) is not float
        or not np.isfinite(stopped_speed_threshold_mps)
        or stopped_speed_threshold_mps <= 0.0
    ):
        raise ValueError("stopped_speed_threshold_mps must be a positive finite float")
    configured = dict(env_config)
    configured["map"] = spec.map
    if configured.get("trajectory_execution_steps") != 1:
        raise ValueError("rollout requires env.trajectory_execution_steps=1")
    env_slot = MetaDriveEnvSlot(
        configured,
        mode=mode,
        observation_spec=PlannerObservationSpec.from_planner_config(runtime.planner_config),
        map_query_radius_m=map_query_radius_m,
        history_warmup_steps=history_warmup_steps,
    )
    resolved_noise_seed = runtime.noise_seed if noise_seed is None else _seed(noise_seed, "noise")
    resolved_policy_seed = (
        runtime.policy_action_seed
        if policy_action_seed is None
        else _seed(policy_action_seed, "policy action")
    )
    if diffusion_generator is None:
        diffusion_generator = runtime.new_noise_generator()
    if policy_generator is None:
        policy_generator = runtime.new_policy_generator()
    training_transitions = []
    audit_transitions = []
    try:
        env_slot.reset(map_name=spec.map, seed=spec.seed)
        tuple(env_slot.warmup())
        previous_route_completion = env_slot.env.route_completion

        for cycle in range(max_transitions):
            observation = collate_observations([env_slot.observe().observation])
            decision = runtime.decide(observation, diffusion_generator, policy_generator)
            training_decision = decision.training_decision
            if training_transitions:
                set_training_transition_next_state_value(
                    training_transitions[-1], training_decision["state_value"]
                )
            step = env_slot.step(decision.ego_trajectory)
            terminated = step.terminated
            truncated = step.truncated
            audit_result = decision.audit_result()
            execution = step.execution
            if execution.substep_states.shape[0] != 1:
                raise RuntimeError("rollout transition must execute exactly one substep")
            audit = _transition_audit(
                execution,
                previous_route_completion,
                stopped_speed_threshold_mps,
            )
            reward = float(execution.substep_rewards.sum())
            training_transitions.append(
                build_training_transition(
                    training_decision,
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                )
            )
            audit_transitions.append(
                build_rollout_audit(
                    policy_context=audit_result.policy_context,
                    base_action=audit_result.base_action,
                    guidance_action=audit_result.guidance_action,
                    old_joint_guidance_log_prob=audit_result.old_joint_guidance_log_prob,
                    state_value=audit_result.old_value,
                    beta_alpha=audit_result.beta_alpha,
                    beta_beta=audit_result.beta_beta,
                    initial_noise=audit_result.initial_noise,
                    diffusion_rng_state=audit_result.diffusion_rng_state,
                    policy_rng_state=audit_result.policy_rng_state,
                    reward=reward,
                    dense_reward=float(execution.substep_dense_rewards.sum()),
                    terminal_override=float(
                        (execution.substep_rewards - execution.substep_dense_rewards).sum()
                    ),
                    terminated=terminated,
                    truncated=truncated,
                    map_seed=spec.seed,
                    noise_seed=resolved_noise_seed,
                    policy_action_seed=resolved_policy_seed,
                    planning_cycle_index=cycle,
                    **audit,
                )
            )
            previous_route_completion = execution.route_completion
            if terminated:
                return finalize_rollout_episode(
                    training_transitions, audit_transitions, "terminated", torch.zeros(1)
                )
            if truncated:
                next_observation = collate_observations([env_slot.observe().observation])
                return finalize_rollout_episode(
                    training_transitions,
                    audit_transitions,
                    "truncated",
                    runtime.bootstrap_value(next_observation, diffusion_generator),
                )
        next_observation = collate_observations([env_slot.observe().observation])
        return finalize_rollout_episode(
            training_transitions,
            audit_transitions,
            "rollout_limit",
            runtime.bootstrap_value(next_observation, diffusion_generator),
        )
    finally:
        env_slot.close()


class VectorRolloutCollector:
    """Own a fixed MetaDrive worker pool across one or more PPO collections."""

    def __init__(
        self,
        specs: tuple[ScenarioConfig, ...],
        runtime: FabricRolloutRuntime,
        env_config: Mapping[str, object],
        *,
        mode: Literal["no_traffic", "traffic"],
        map_query_radius_m: float,
        history_warmup_steps: int,
        physical_slot_count: int | None = None,
    ) -> None:
        if not specs:
            raise ValueError("vector rollout requires at least one scenario")
        if env_config.get("trajectory_execution_steps") != 1:
            raise ValueError("rollout requires env.trajectory_execution_steps=1")
        if physical_slot_count is None:
            physical_slot_count = len(specs)
        if type(physical_slot_count) is not int or physical_slot_count <= 0:
            raise ValueError("physical_slot_count must be a positive integer")
        self._specs = specs
        self._runtime = runtime
        self._physical_slot_count = min(physical_slot_count, len(specs))
        self._scenarios = tuple(VectorEnvScenario(spec.name, spec.map, spec.seed) for spec in specs)
        configured_envs = tuple(
            {**env_config, "map": spec.map} for spec in specs[: self._physical_slot_count]
        )
        self._envs = VectorMetaDriveEnv(
            configured_envs,
            mode=mode,
            observation_spec=PlannerObservationSpec.from_planner_config(runtime.planner_config),
            map_query_radius_m=map_query_radius_m,
            history_warmup_steps=history_warmup_steps,
        )
        self._close_finalizer = finalize(self, self._envs.close)

    def __enter__(self) -> VectorRolloutCollector:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the worker pool owned by this collector."""

        self._close_finalizer()

    def collect(
        self,
        *,
        transitions_per_slot: int,
        stopped_speed_threshold_mps: float,
        diffusion_generators: tuple[torch.Generator, ...],
        policy_generators: tuple[torch.Generator, ...],
        noise_seeds: tuple[int, ...],
        policy_action_seeds: tuple[int, ...],
        timings: list[VectorRolloutRoundTiming] | None = None,
    ) -> tuple[tuple[RolloutEpisode, ...], ...]:
        """Collect one PPO batch while retaining workers for a subsequent call."""

        specs = self._specs
        runtime = self._runtime
        envs = self._envs
        slot_count = len(specs)
        if type(transitions_per_slot) is not int or transitions_per_slot <= 0:
            raise ValueError("transitions_per_slot must be a positive integer")
        _validate_vector_slots(
            slot_count,
            diffusion_generators,
            policy_generators,
            noise_seeds,
            policy_action_seeds,
        )
        if self._physical_slot_count < slot_count:
            return self._collect_waves(
                transitions_per_slot=transitions_per_slot,
                stopped_speed_threshold_mps=stopped_speed_threshold_mps,
                diffusion_generators=diffusion_generators,
                policy_generators=policy_generators,
                noise_seeds=noise_seeds,
                policy_action_seeds=policy_action_seeds,
                timings=timings,
            )
        scenarios = self._scenarios
        episodes: list[list[RolloutEpisode]] = [[] for _ in specs]
        training: list[list[TensorDictBase]] = [[] for _ in specs]
        audit: list[list[TensorDictBase]] = [[] for _ in specs]
        collected = [0] * slot_count
        episode_cycles = [0] * slot_count

        resets = envs.reset(scenarios)
        observations = [reset.observation for reset in resets]
        previous_route_completion = [reset.route_completion for reset in resets]
        while collected[0] < transitions_per_slot:
            if any(count != collected[0] for count in collected):
                raise RuntimeError("fixed-slot vector rollout consumed unequal transition counts")
            profile = timings is not None
            planner_started = perf_counter() if profile else 0.0
            decision = runtime.decide_batch(
                collate_observations(observations), diffusion_generators, policy_generators
            )
            for slot, transitions in enumerate(training):
                if transitions:
                    set_training_transition_next_state_value(
                        transitions[-1], decision.slot(slot).training_decision["state_value"]
                    )
            planner_s = perf_counter() - planner_started if profile else 0.0
            environment_started = perf_counter() if profile else 0.0
            steps = envs.step(decision.ego_trajectories)
            environment_s = perf_counter() - environment_started if profile else 0.0
            if timings is not None:
                worker_busy = tuple(
                    step.timing.environment_s + step.timing.observation_s for step in steps
                )
                slowest = max(worker_busy)
                timings.append(
                    VectorRolloutRoundTiming(
                        phase="decision",
                        active_slots=len(steps),
                        capacity=slot_count,
                        planner_wall_s=planner_s,
                        environment_wall_s=environment_s,
                        worker_busy_s=sum(worker_busy),
                        worker_wait_s=sum(step.timing.worker_wait_s for step in steps),
                        worker_imbalance_s=sum(slowest - value for value in worker_busy),
                    )
                )
            tails: list[tuple[int, Literal["terminated", "truncated", "rollout_limit"]]] = []
            for slot, step in enumerate(steps):
                slot_decision = decision.slot(slot)
                execution = step.execution
                if execution.substep_states.shape[0] != 1:
                    raise RuntimeError("rollout transition must execute exactly one substep")
                reward = float(execution.substep_rewards.sum())
                training[slot].append(
                    build_training_transition(
                        slot_decision.training_decision,
                        reward=reward,
                        terminated=step.terminated,
                        truncated=step.truncated,
                    )
                )
                audit_result = slot_decision.audit_result()
                audit[slot].append(
                    build_rollout_audit(
                        policy_context=audit_result.policy_context,
                        base_action=audit_result.base_action,
                        guidance_action=audit_result.guidance_action,
                        old_joint_guidance_log_prob=audit_result.old_joint_guidance_log_prob,
                        state_value=audit_result.old_value,
                        beta_alpha=audit_result.beta_alpha,
                        beta_beta=audit_result.beta_beta,
                        initial_noise=audit_result.initial_noise,
                        diffusion_rng_state=audit_result.diffusion_rng_state,
                        policy_rng_state=audit_result.policy_rng_state,
                        reward=reward,
                        dense_reward=float(execution.substep_dense_rewards.sum()),
                        terminal_override=float(
                            (execution.substep_rewards - execution.substep_dense_rewards).sum()
                        ),
                        terminated=step.terminated,
                        truncated=step.truncated,
                        map_seed=specs[slot].seed,
                        noise_seed=noise_seeds[slot],
                        policy_action_seed=policy_action_seeds[slot],
                        planning_cycle_index=episode_cycles[slot],
                        **_transition_audit(
                            execution,
                            previous_route_completion[slot],
                            stopped_speed_threshold_mps,
                        ),
                    )
                )
                collected[slot] += 1
                episode_cycles[slot] += 1
                previous_route_completion[slot] = execution.route_completion
                if step.terminated:
                    tails.append((slot, "terminated"))
                elif step.truncated:
                    tails.append((slot, "truncated"))
                elif collected[slot] == transitions_per_slot:
                    tails.append((slot, "rollout_limit"))

            bootstrap_slots = [slot for slot, kind in tails if kind != "terminated"]
            bootstrap_values: dict[int, torch.Tensor] = {}
            if bootstrap_slots:
                bootstrap_started = perf_counter() if timings is not None else 0.0
                values = runtime.bootstrap_value_batch(
                    collate_observations([steps[slot].observation for slot in bootstrap_slots]),
                    tuple(diffusion_generators[slot] for slot in bootstrap_slots),
                )
                if timings is not None:
                    timings.append(
                        VectorRolloutRoundTiming(
                            phase="bootstrap",
                            active_slots=len(bootstrap_slots),
                            capacity=slot_count,
                            planner_wall_s=perf_counter() - bootstrap_started,
                            environment_wall_s=0.0,
                            worker_busy_s=0.0,
                            worker_wait_s=0.0,
                            worker_imbalance_s=0.0,
                        )
                    )
                for index, slot in enumerate(bootstrap_slots):
                    bootstrap_values[slot] = values[index : index + 1]
            for slot, kind in tails:
                episodes[slot].append(
                    finalize_rollout_episode(
                        training[slot],
                        audit[slot],
                        kind,
                        torch.zeros(1) if kind == "terminated" else bootstrap_values[slot],
                    )
                )
                training[slot] = []
                audit[slot] = []
                if collected[slot] < transitions_per_slot:
                    reset = envs.reset_at(slot, scenarios[slot])
                    observations[slot] = reset.observation
                    previous_route_completion[slot] = reset.route_completion
                    episode_cycles[slot] = 0
            for slot, step in enumerate(steps):
                if collected[slot] < transitions_per_slot and not any(
                    tail_slot == slot for tail_slot, _ in tails
                ):
                    observations[slot] = step.observation

        if any(training) or any(audit):
            raise RuntimeError("vector rollout ended with an unfinished episode")
        return tuple(tuple(slot_episodes) for slot_episodes in episodes)

    def _collect_waves(
        self,
        *,
        transitions_per_slot: int,
        stopped_speed_threshold_mps: float,
        diffusion_generators: tuple[torch.Generator, ...],
        policy_generators: tuple[torch.Generator, ...],
        noise_seeds: tuple[int, ...],
        policy_action_seeds: tuple[int, ...],
        timings: list[VectorRolloutRoundTiming] | None,
    ) -> tuple[tuple[RolloutEpisode, ...], ...]:
        """Collect logical scenario slots in deterministic physical-worker waves."""

        collected: list[tuple[RolloutEpisode, ...]] = []
        for start in range(0, len(self._specs), self._physical_slot_count):
            stop = min(start + self._physical_slot_count, len(self._specs))
            collected.extend(
                self._collect_wave(
                    self._specs[start:stop],
                    self._scenarios[start:stop],
                    transitions_per_slot=transitions_per_slot,
                    stopped_speed_threshold_mps=stopped_speed_threshold_mps,
                    diffusion_generators=diffusion_generators[start:stop],
                    policy_generators=policy_generators[start:stop],
                    noise_seeds=noise_seeds[start:stop],
                    policy_action_seeds=policy_action_seeds[start:stop],
                    timings=timings,
                )
            )
        return tuple(collected)

    def _collect_wave(
        self,
        specs: tuple[ScenarioConfig, ...],
        scenarios: tuple[VectorEnvScenario, ...],
        *,
        transitions_per_slot: int,
        stopped_speed_threshold_mps: float,
        diffusion_generators: tuple[torch.Generator, ...],
        policy_generators: tuple[torch.Generator, ...],
        noise_seeds: tuple[int, ...],
        policy_action_seeds: tuple[int, ...],
        timings: list[VectorRolloutRoundTiming] | None,
    ) -> tuple[tuple[RolloutEpisode, ...], ...]:
        """Collect one wave without changing logical scenario RNG ownership."""

        active_slots = tuple(range(len(specs)))
        envs = self._envs
        resets = tuple(envs.reset_at(slot, scenario) for slot, scenario in enumerate(scenarios))
        observations = [reset.observation for reset in resets]
        previous_route_completion = [reset.route_completion for reset in resets]
        episodes: list[list[RolloutEpisode]] = [[] for _ in specs]
        training: list[list[TensorDictBase]] = [[] for _ in specs]
        audit: list[list[TensorDictBase]] = [[] for _ in specs]
        collected = [0] * len(specs)
        episode_cycles = [0] * len(specs)

        while collected[0] < transitions_per_slot:
            if any(count != collected[0] for count in collected):
                raise RuntimeError("wave rollout consumed unequal transition counts")
            profile = timings is not None
            planner_started = perf_counter() if profile else 0.0
            decision = self._runtime.decide_batch(
                collate_observations(observations), diffusion_generators, policy_generators
            )
            for slot, transitions in enumerate(training):
                if transitions:
                    set_training_transition_next_state_value(
                        transitions[-1], decision.slot(slot).training_decision["state_value"]
                    )
            planner_s = perf_counter() - planner_started if profile else 0.0
            environment_started = perf_counter() if profile else 0.0
            steps = envs.step_slots(active_slots, decision.ego_trajectories)
            environment_s = perf_counter() - environment_started if profile else 0.0
            if timings is not None:
                worker_busy = tuple(
                    step.timing.environment_s + step.timing.observation_s for step in steps
                )
                slowest = max(worker_busy)
                timings.append(
                    VectorRolloutRoundTiming(
                        phase="decision",
                        active_slots=len(steps),
                        capacity=self._physical_slot_count,
                        planner_wall_s=planner_s,
                        environment_wall_s=environment_s,
                        worker_busy_s=sum(worker_busy),
                        worker_wait_s=sum(step.timing.worker_wait_s for step in steps),
                        worker_imbalance_s=sum(slowest - value for value in worker_busy),
                    )
                )
            tails: list[tuple[int, Literal["terminated", "truncated", "rollout_limit"]]] = []
            for slot, step in enumerate(steps):
                execution = step.execution
                if execution.substep_states.shape[0] != 1:
                    raise RuntimeError("rollout transition must execute exactly one substep")
                slot_decision = decision.slot(slot)
                reward = float(execution.substep_rewards.sum())
                training[slot].append(
                    build_training_transition(
                        slot_decision.training_decision,
                        reward=reward,
                        terminated=step.terminated,
                        truncated=step.truncated,
                    )
                )
                audit_result = slot_decision.audit_result()
                audit[slot].append(
                    build_rollout_audit(
                        policy_context=audit_result.policy_context,
                        base_action=audit_result.base_action,
                        guidance_action=audit_result.guidance_action,
                        old_joint_guidance_log_prob=audit_result.old_joint_guidance_log_prob,
                        state_value=audit_result.old_value,
                        beta_alpha=audit_result.beta_alpha,
                        beta_beta=audit_result.beta_beta,
                        initial_noise=audit_result.initial_noise,
                        diffusion_rng_state=audit_result.diffusion_rng_state,
                        policy_rng_state=audit_result.policy_rng_state,
                        reward=reward,
                        dense_reward=float(execution.substep_dense_rewards.sum()),
                        terminal_override=float(
                            (execution.substep_rewards - execution.substep_dense_rewards).sum()
                        ),
                        terminated=step.terminated,
                        truncated=step.truncated,
                        map_seed=specs[slot].seed,
                        noise_seed=noise_seeds[slot],
                        policy_action_seed=policy_action_seeds[slot],
                        planning_cycle_index=episode_cycles[slot],
                        **_transition_audit(
                            execution,
                            previous_route_completion[slot],
                            stopped_speed_threshold_mps,
                        ),
                    )
                )
                collected[slot] += 1
                episode_cycles[slot] += 1
                previous_route_completion[slot] = execution.route_completion
                if step.terminated:
                    tails.append((slot, "terminated"))
                elif step.truncated:
                    tails.append((slot, "truncated"))
                elif collected[slot] == transitions_per_slot:
                    tails.append((slot, "rollout_limit"))

            bootstrap_slots = [slot for slot, kind in tails if kind != "terminated"]
            bootstrap_values: dict[int, torch.Tensor] = {}
            if bootstrap_slots:
                bootstrap_started = perf_counter() if timings is not None else 0.0
                values = self._runtime.bootstrap_value_batch(
                    collate_observations([steps[slot].observation for slot in bootstrap_slots]),
                    tuple(diffusion_generators[slot] for slot in bootstrap_slots),
                )
                if timings is not None:
                    timings.append(
                        VectorRolloutRoundTiming(
                            phase="bootstrap",
                            active_slots=len(bootstrap_slots),
                            capacity=self._physical_slot_count,
                            planner_wall_s=perf_counter() - bootstrap_started,
                            environment_wall_s=0.0,
                            worker_busy_s=0.0,
                            worker_wait_s=0.0,
                            worker_imbalance_s=0.0,
                        )
                    )
                for index, slot in enumerate(bootstrap_slots):
                    bootstrap_values[slot] = values[index : index + 1]
            for slot, kind in tails:
                episodes[slot].append(
                    finalize_rollout_episode(
                        training[slot],
                        audit[slot],
                        kind,
                        torch.zeros(1) if kind == "terminated" else bootstrap_values[slot],
                    )
                )
                training[slot] = []
                audit[slot] = []
                if collected[slot] < transitions_per_slot:
                    reset = envs.reset_at(slot, scenarios[slot])
                    observations[slot] = reset.observation
                    previous_route_completion[slot] = reset.route_completion
                    episode_cycles[slot] = 0
            for slot, step in enumerate(steps):
                if collected[slot] < transitions_per_slot and not any(
                    tail_slot == slot for tail_slot, _ in tails
                ):
                    observations[slot] = step.observation

        if any(training) or any(audit):
            raise RuntimeError("wave rollout ended with an unfinished episode")
        return tuple(tuple(slot_episodes) for slot_episodes in episodes)


def collect_vector_rollout_episodes(
    specs: tuple[ScenarioConfig, ...],
    runtime: FabricRolloutRuntime,
    env_config: Mapping[str, object],
    *,
    mode: Literal["no_traffic", "traffic"],
    map_query_radius_m: float,
    history_warmup_steps: int,
    transitions_per_slot: int,
    stopped_speed_threshold_mps: float,
    diffusion_generators: tuple[torch.Generator, ...],
    policy_generators: tuple[torch.Generator, ...],
    noise_seeds: tuple[int, ...],
    policy_action_seeds: tuple[int, ...],
    timings: list[VectorRolloutRoundTiming] | None = None,
) -> tuple[tuple[RolloutEpisode, ...], ...]:
    """Collect one PPO batch with a temporary vector worker pool.

    Training should use :class:`VectorRolloutCollector` directly to reuse workers across
    PPO updates. This function preserves the one-shot collector API for callers that need it.
    """

    with VectorRolloutCollector(
        specs,
        runtime,
        env_config,
        mode=mode,
        map_query_radius_m=map_query_radius_m,
        history_warmup_steps=history_warmup_steps,
    ) as rollout_collector:
        return rollout_collector.collect(
            transitions_per_slot=transitions_per_slot,
            stopped_speed_threshold_mps=stopped_speed_threshold_mps,
            diffusion_generators=diffusion_generators,
            policy_generators=policy_generators,
            noise_seeds=noise_seeds,
            policy_action_seeds=policy_action_seeds,
            timings=timings,
        )


def _validate_vector_slots(
    slot_count: int,
    diffusion_generators: tuple[torch.Generator, ...],
    policy_generators: tuple[torch.Generator, ...],
    noise_seeds: tuple[int, ...],
    policy_action_seeds: tuple[int, ...],
) -> None:
    for name, values in (
        ("diffusion_generators", diffusion_generators),
        ("policy_generators", policy_generators),
        ("noise_seeds", noise_seeds),
        ("policy_action_seeds", policy_action_seeds),
    ):
        if len(values) != slot_count:
            raise ValueError(f"{name} must contain one value per vector slot")


def _transition_audit(
    execution: TrajectoryExecutionRecord,
    previous_route_completion: float,
    stopped_speed_threshold_mps: float,
) -> dict[str, float | bool]:
    state = execution.substep_states[0]
    distance_m = float(np.linalg.norm(state[:2] - execution.start_center))
    speed_mps = float(state[5])
    return {
        "route_completion_delta": float(execution.route_completion - previous_route_completion),
        "distance_m": distance_m,
        "speed_mps": speed_mps,
        "stopped": speed_mps < stopped_speed_threshold_mps,
        "position_error_m": float(execution.position_errors_m[0]),
        "heading_error_rad": float(execution.heading_errors_rad[0]),
        "arrive_dest": execution.arrive_dest,
        "out_of_road": execution.out_of_road,
        "crash_vehicle": execution.crash_vehicle,
        "crash_object": execution.crash_object,
        "crash_building": execution.crash_building,
        "crash_human": execution.crash_human,
    }


def _seed(value: int, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} seed must be a non-negative integer")
    return value
