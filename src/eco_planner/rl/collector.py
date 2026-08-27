"""Closed-loop rollout collectors for serial and fixed-slot vector execution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
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
    VectorEnvReset,
    VectorEnvScenario,
    VectorEnvStep,
    VectorMetaDriveEnv,
    collate_observations,
)
from eco_planner.envs.array_types import SingleObservation
from eco_planner.envs.metadrive.reward import (
    MetaDriveBuiltinRewardAudit,
    PlannerRFTEnergyRewardAudit,
    RewardAudit,
    RewardProfileConfig,
)
from eco_planner.evaluation.config import ScenarioConfig
from eco_planner.rl.rollout import (
    RolloutEpisode,
    TailKind,
    build_rollout_audit,
    build_training_transition,
    finalize_rollout_episode,
    set_training_transition_next_state_value,
)
from eco_planner.rl.runtime import FabricRolloutRuntime, RolloutDecision


@dataclass(frozen=True)
class VectorRolloutRoundTiming:
    """One fixed-slot collection round, exposed for throughput profiling only."""

    phase: Literal["decision", "bootstrap"]
    active_slots: int
    capacity: int
    planner_wall_s: float
    environment_wall_s: float
    worker_busy_s: float
    transport_sync_s: float
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
    reward_profile: RewardProfileConfig | None = None,
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
    if configured.get("execution_mode") not in {None, "rollout"}:
        raise ValueError("rollout requires env.execution_mode=rollout")
    configured["execution_mode"] = "rollout"
    env_slot = MetaDriveEnvSlot(
        configured,
        mode=mode,
        observation_spec=PlannerObservationSpec.from_planner_config(runtime.planner_config),
        map_query_radius_m=map_query_radius_m,
        history_warmup_steps=history_warmup_steps,
        reward_profile=reward_profile,
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
            dense_reward, terminal_override, reward_audit = _reward_audit_values(execution, reward)
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
                    dense_reward=dense_reward,
                    terminal_override=terminal_override,
                    terminated=terminated,
                    truncated=truncated,
                    map_seed=spec.seed,
                    noise_seed=resolved_noise_seed,
                    policy_action_seed=resolved_policy_seed,
                    planning_cycle_index=cycle,
                    reward_audit=reward_audit,
                    crash_sidewalk=execution.crash_sidewalk,
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


@dataclass
class _SlotCollectionState:
    spec: ScenarioConfig
    scenario: VectorEnvScenario
    diffusion_generator: torch.Generator
    policy_generator: torch.Generator
    noise_seed: int
    policy_action_seed: int
    observation: SingleObservation
    previous_route_completion: float
    collected: int = 0
    episode_cycle: int = 0
    training: list[TensorDictBase] = field(default_factory=list)
    audit: list[TensorDictBase] = field(default_factory=list)
    episodes: list[RolloutEpisode] = field(default_factory=list)

    def reset(self, result: VectorEnvReset) -> None:
        self.observation = result.observation
        self.previous_route_completion = result.route_completion
        self.episode_cycle = 0


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
        torch_threads_per_worker: int | None = None,
        reward_profile: RewardProfileConfig | None = None,
    ) -> None:
        if not specs:
            raise ValueError("vector rollout requires at least one scenario")
        configured = dict(env_config)
        if configured.get("execution_mode") not in {None, "rollout"}:
            raise ValueError("rollout requires env.execution_mode=rollout")
        configured["execution_mode"] = "rollout"
        if physical_slot_count is None:
            physical_slot_count = len(specs)
        if type(physical_slot_count) is not int or physical_slot_count <= 0:
            raise ValueError("physical_slot_count must be a positive integer")
        self._specs = specs
        self._runtime = runtime
        self._physical_slot_count = min(physical_slot_count, len(specs))
        self._scenarios = tuple(VectorEnvScenario(spec.name, spec.map, spec.seed) for spec in specs)
        configured_envs = tuple(
            {**configured, "map": spec.map} for spec in specs[: self._physical_slot_count]
        )
        self._envs = VectorMetaDriveEnv(
            configured_envs,
            mode=mode,
            observation_spec=PlannerObservationSpec.from_planner_config(runtime.planner_config),
            map_query_radius_m=map_query_radius_m,
            history_warmup_steps=history_warmup_steps,
            scenarios=self._scenarios,
            torch_threads_per_worker=torch_threads_per_worker,
            reward_profile=reward_profile,
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

        if type(transitions_per_slot) is not int or transitions_per_slot <= 0:
            raise ValueError("transitions_per_slot must be a positive integer")
        _validate_vector_slots(
            len(self._specs),
            diffusion_generators,
            policy_generators,
            noise_seeds,
            policy_action_seeds,
        )
        collected: list[tuple[RolloutEpisode, ...]] = []
        for start in range(0, len(self._specs), self._physical_slot_count):
            stop = min(start + self._physical_slot_count, len(self._specs))
            states = self._initialize_group(
                self._specs[start:stop],
                self._scenarios[start:stop],
                diffusion_generators[start:stop],
                policy_generators[start:stop],
                noise_seeds[start:stop],
                policy_action_seeds[start:stop],
            )
            collected.extend(
                self._collect_group(
                    states,
                    transitions_per_slot=transitions_per_slot,
                    stopped_speed_threshold_mps=stopped_speed_threshold_mps,
                    timings=timings,
                )
            )
        return tuple(collected)

    def _initialize_group(
        self,
        specs: tuple[ScenarioConfig, ...],
        scenarios: tuple[VectorEnvScenario, ...],
        diffusion_generators: tuple[torch.Generator, ...],
        policy_generators: tuple[torch.Generator, ...],
        noise_seeds: tuple[int, ...],
        policy_action_seeds: tuple[int, ...],
    ) -> list[_SlotCollectionState]:
        if len(specs) == self._physical_slot_count:
            resets = self._envs.reset(scenarios)
        else:
            resets = tuple(
                self._envs.reset_at(slot, scenario) for slot, scenario in enumerate(scenarios)
            )
        return [
            _SlotCollectionState(
                spec=spec,
                scenario=scenario,
                diffusion_generator=diffusion_generator,
                policy_generator=policy_generator,
                noise_seed=noise_seed,
                policy_action_seed=policy_action_seed,
                observation=reset.observation,
                previous_route_completion=reset.route_completion,
            )
            for (
                spec,
                scenario,
                diffusion_generator,
                policy_generator,
                noise_seed,
                policy_action_seed,
                reset,
            ) in zip(
                specs,
                scenarios,
                diffusion_generators,
                policy_generators,
                noise_seeds,
                policy_action_seeds,
                resets,
                strict=True,
            )
        ]

    def _collect_group(
        self,
        states: list[_SlotCollectionState],
        *,
        transitions_per_slot: int,
        stopped_speed_threshold_mps: float,
        timings: list[VectorRolloutRoundTiming] | None,
    ) -> tuple[tuple[RolloutEpisode, ...], ...]:
        active_slots = tuple(range(len(states)))
        full_capacity = len(states) == self._physical_slot_count
        while states[0].collected < transitions_per_slot:
            if any(state.collected != states[0].collected for state in states):
                raise RuntimeError("vector rollout consumed unequal transition counts")
            profile = timings is not None
            planner_started = perf_counter() if profile else 0.0
            decision = self._runtime.decide_batch(
                collate_observations([state.observation for state in states]),
                tuple(state.diffusion_generator for state in states),
                tuple(state.policy_generator for state in states),
            )
            for slot, state in enumerate(states):
                if state.training:
                    set_training_transition_next_state_value(
                        state.training[-1], decision.slot(slot).training_decision["state_value"]
                    )
            planner_s = perf_counter() - planner_started if profile else 0.0
            environment_started = perf_counter() if profile else 0.0
            if full_capacity:
                steps = self._envs.step(decision.ego_trajectories)
            else:
                steps = self._envs.step_slots(active_slots, decision.ego_trajectories)
            environment_s = perf_counter() - environment_started if profile else 0.0
            _append_decision_timing(
                timings,
                steps,
                capacity=self._physical_slot_count,
                planner_s=planner_s,
                environment_s=environment_s,
            )

            tails: list[tuple[int, TailKind]] = []
            for slot, (state, step) in enumerate(zip(states, steps, strict=True)):
                tail = _append_slot_transition(
                    state,
                    decision.slot(slot),
                    step,
                    transitions_per_slot=transitions_per_slot,
                    stopped_speed_threshold_mps=stopped_speed_threshold_mps,
                )
                if tail is not None:
                    tails.append((slot, tail))

            bootstrap_slots = [slot for slot, kind in tails if kind != "terminated"]
            bootstrap_values: dict[int, torch.Tensor] = {}
            if bootstrap_slots:
                bootstrap_started = perf_counter() if timings is not None else 0.0
                values = self._runtime.bootstrap_value_batch(
                    collate_observations([steps[slot].observation for slot in bootstrap_slots]),
                    tuple(states[slot].diffusion_generator for slot in bootstrap_slots),
                )
                _append_bootstrap_timing(
                    timings,
                    active_slots=len(bootstrap_slots),
                    capacity=self._physical_slot_count,
                    planner_s=(perf_counter() - bootstrap_started if timings is not None else 0.0),
                )
                for index, slot in enumerate(bootstrap_slots):
                    bootstrap_values[slot] = values[index : index + 1]

            tail_slots = {slot for slot, _ in tails}
            for slot, kind in tails:
                state = states[slot]
                state.episodes.append(
                    finalize_rollout_episode(
                        state.training,
                        state.audit,
                        kind,
                        (torch.zeros(1) if kind == "terminated" else bootstrap_values[slot]),
                    )
                )
                state.training = []
                state.audit = []
                if state.collected < transitions_per_slot:
                    state.reset(self._envs.reset_at(slot, state.scenario))
            for slot, (state, step) in enumerate(zip(states, steps, strict=True)):
                if state.collected < transitions_per_slot and slot not in tail_slots:
                    state.observation = step.observation

        if any(state.training or state.audit for state in states):
            raise RuntimeError("vector rollout ended with an unfinished episode")
        return tuple(tuple(state.episodes) for state in states)


def _append_slot_transition(
    state: _SlotCollectionState,
    decision: RolloutDecision,
    step: VectorEnvStep,
    *,
    transitions_per_slot: int,
    stopped_speed_threshold_mps: float,
) -> TailKind | None:
    execution = step.execution
    if execution.substep_states.shape[0] != 1:
        raise RuntimeError("rollout transition must execute exactly one substep")
    reward = float(execution.substep_rewards.sum())
    dense_reward, terminal_override, reward_audit = _reward_audit_values(execution, reward)
    state.training.append(
        build_training_transition(
            decision.training_decision,
            reward=reward,
            terminated=step.terminated,
            truncated=step.truncated,
        )
    )
    audit_result = decision.audit_result()
    state.audit.append(
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
            dense_reward=dense_reward,
            terminal_override=terminal_override,
            terminated=step.terminated,
            truncated=step.truncated,
            map_seed=state.spec.seed,
            noise_seed=state.noise_seed,
            policy_action_seed=state.policy_action_seed,
            planning_cycle_index=state.episode_cycle,
            reward_audit=reward_audit,
            crash_sidewalk=execution.crash_sidewalk,
            **_transition_audit(
                execution,
                state.previous_route_completion,
                stopped_speed_threshold_mps,
            ),
        )
    )
    state.collected += 1
    state.episode_cycle += 1
    state.previous_route_completion = execution.route_completion
    if step.terminated:
        return "terminated"
    if step.truncated:
        return "truncated"
    if state.collected == transitions_per_slot:
        return "rollout_limit"
    return None


def _append_decision_timing(
    timings: list[VectorRolloutRoundTiming] | None,
    steps: tuple[VectorEnvStep, ...],
    *,
    capacity: int,
    planner_s: float,
    environment_s: float,
) -> None:
    if timings is None:
        return
    worker_busy = tuple(step.timing.environment_s + step.timing.observation_s for step in steps)
    slowest = max(worker_busy)
    timings.append(
        VectorRolloutRoundTiming(
            phase="decision",
            active_slots=len(steps),
            capacity=capacity,
            planner_wall_s=planner_s,
            environment_wall_s=environment_s,
            worker_busy_s=sum(worker_busy),
            transport_sync_s=max(0.0, environment_s - slowest),
            worker_imbalance_s=sum(slowest - value for value in worker_busy),
        )
    )


def _append_bootstrap_timing(
    timings: list[VectorRolloutRoundTiming] | None,
    *,
    active_slots: int,
    capacity: int,
    planner_s: float,
) -> None:
    if timings is None:
        return
    timings.append(
        VectorRolloutRoundTiming(
            phase="bootstrap",
            active_slots=active_slots,
            capacity=capacity,
            planner_wall_s=planner_s,
            environment_wall_s=0.0,
            worker_busy_s=0.0,
            transport_sync_s=0.0,
            worker_imbalance_s=0.0,
        )
    )


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
    reward_profile: RewardProfileConfig | None = None,
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
        reward_profile=reward_profile,
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


def _reward_audit_values(
    execution: TrajectoryExecutionRecord, reward: float
) -> tuple[float, float, RewardAudit]:
    if len(execution.substep_reward_audits) != 1:
        raise RuntimeError("rollout transition must expose exactly one reward audit")
    audit = execution.substep_reward_audits[0]
    if not np.isclose(reward, audit.reward_total, rtol=0.0, atol=1e-12):
        raise RuntimeError("environment reward disagrees with its typed reward audit")
    if isinstance(audit, PlannerRFTEnergyRewardAudit):
        return audit.reward_ungated, audit.reward_total - audit.reward_ungated, audit
    if isinstance(audit, MetaDriveBuiltinRewardAudit):
        return audit.dense_reward, audit.terminal_override, audit
    raise TypeError("rollout transition returned an unsupported reward audit type")


def _seed(value: int, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} seed must be a non-negative integer")
    return value
