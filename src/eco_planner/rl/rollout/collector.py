"""Closed-loop rollout collectors for serial and fixed-slot vector execution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from time import perf_counter
from typing import Literal, cast
from weakref import finalize

import numpy as np
import torch
from tensordict import TensorDictBase

from eco_planner.configuration import ScenarioConfig
from eco_planner.contracts import ExecutionMode
from eco_planner.envs import (
    MetaDriveEnvSlot,
    TrajectoryExecutionResult,
)
from eco_planner.rl.reward import (
    RewardEvaluator,
    RewardProfileConfig,
    create_reward_evaluator,
)
from eco_planner.rl.rollout.contracts import (
    ExecutionTransitionAudit,
    RolloutEpisode,
    RolloutEpisodeBuilder,
    RolloutProvenance,
    TailKind,
)
from eco_planner.rl.rollout.decision import RolloutDecision
from eco_planner.rl.rollout.profiling import RolloutPlannerTiming
from eco_planner.rl.rollout.runtime import FabricRolloutRuntime
from eco_planner.runtime.envs import (
    VectorEnvScenario,
    VectorMetaDriveEnv,
    WorkerResetResult,
    WorkerStepResult,
    operation_results,
)


@dataclass(frozen=True)
class VectorRolloutRoundTiming:
    """One fixed-slot collection round, exposed for throughput profiling only."""

    phase: Literal["decision", "bootstrap"]
    active_slots: int
    capacity: int
    collate_wall_s: float
    planner_wall_s: float
    planner_timing: RolloutPlannerTiming
    environment_wall_s: float
    audit_resolve_wall_s: float
    audit_transfer_accelerator_s: float
    worker_busy_s: float
    transport_sync_s: float
    worker_imbalance_s: float


@dataclass
class _EpisodeLifecycle:
    previous_route_completion: float
    cycle: int = 0
    builder: RolloutEpisodeBuilder = field(default_factory=RolloutEpisodeBuilder)

    @property
    def empty(self) -> bool:
        return self.builder.empty

    def link_next_state_value(self, decision: RolloutDecision) -> None:
        if not self.builder.empty:
            self.builder.link_next_state_value(decision.training_decision["state_value"])

    def append(
        self,
        decision: RolloutDecision,
        step: TrajectoryExecutionResult,
        *,
        map_seed: int,
        noise_seed: int,
        policy_action_seed: int,
        stopped_speed_threshold_mps: float,
        reward_evaluator: RewardEvaluator,
        terminated: bool,
        truncated: bool,
        collection_limit: bool,
    ) -> TailKind | None:
        self.builder.append(
            decision.training_decision,
            decision.audit_result(),
            _execution_transition_audit(
                step,
                self.previous_route_completion,
                stopped_speed_threshold_mps,
                reward_evaluator=reward_evaluator,
                terminated=terminated,
                truncated=truncated,
            ),
            RolloutProvenance(
                map_seed=map_seed,
                noise_seed=noise_seed,
                policy_action_seed=policy_action_seed,
                planning_cycle_index=self.cycle,
            ),
        )
        self.cycle += 1
        self.previous_route_completion = step.execution.route_completion
        if terminated:
            return "terminated"
        if truncated:
            return "truncated"
        if collection_limit:
            return "rollout_limit"
        return None

    def finish(self, kind: TailKind, bootstrap_value: torch.Tensor) -> RolloutEpisode:
        episode = self.builder.finish(kind, bootstrap_value)
        self.builder = RolloutEpisodeBuilder()
        self.cycle = 0
        return episode

    def reset(self, route_completion: float) -> None:
        if not self.builder.empty:
            raise RuntimeError("cannot reset an unfinished rollout episode")
        self.previous_route_completion = route_completion
        self.cycle = 0


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
    reward_profile: RewardProfileConfig,
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
    env_slot = MetaDriveEnvSlot(
        configured,
        mode=mode,
        execution_mode=ExecutionMode.ROLLOUT,
        map_query_radius_m=map_query_radius_m,
        history_warmup_steps=history_warmup_steps,
    )
    reward_evaluator = create_reward_evaluator(reward_profile)
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
    try:
        reset = env_slot.reset(map_name=spec.map, seed=spec.seed)
        current_state = reset.state
        lifecycle = _EpisodeLifecycle(current_state.route_completion)

        for cycle in range(max_transitions):
            observation = cast(TensorDictBase, TensorDictBase.stack([current_state.observation]))
            decision = runtime.decide(observation, diffusion_generator, policy_generator)
            lifecycle.link_next_state_value(decision)
            slot_step = env_slot.step(decision.ego_trajectory)
            step = slot_step.execution
            terminated = step.terminated
            truncated = step.truncated
            tail = lifecycle.append(
                decision,
                step,
                map_seed=spec.seed,
                noise_seed=resolved_noise_seed,
                policy_action_seed=resolved_policy_seed,
                stopped_speed_threshold_mps=stopped_speed_threshold_mps,
                reward_evaluator=reward_evaluator,
                terminated=terminated,
                truncated=truncated,
                collection_limit=cycle + 1 == max_transitions,
            )
            if tail == "terminated":
                return lifecycle.finish(tail, torch.zeros(1))
            if tail in {"truncated", "rollout_limit"}:
                next_observation = cast(
                    TensorDictBase, TensorDictBase.stack([slot_step.state.observation])
                )
                return lifecycle.finish(
                    tail, runtime.bootstrap_value(next_observation, diffusion_generator)
                )
            current_state = slot_step.state
        raise RuntimeError("rollout lifecycle did not finish at its explicit transition limit")
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
    lifecycle: _EpisodeLifecycle
    collected: int = 0
    episodes: list[RolloutEpisode] = field(default_factory=list)

    def reset(self, result: WorkerResetResult) -> None:
        self.lifecycle.reset(result.route_completion)


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
        reward_profile: RewardProfileConfig,
    ) -> None:
        if not specs:
            raise ValueError("vector rollout requires at least one scenario")
        configured = dict(env_config)
        if physical_slot_count is None:
            physical_slot_count = len(specs)
        if type(physical_slot_count) is not int or physical_slot_count <= 0:
            raise ValueError("physical_slot_count must be a positive integer")
        self._specs = specs
        self._runtime = runtime
        self._reward_evaluator = create_reward_evaluator(reward_profile)
        self._physical_slot_count = min(physical_slot_count, len(specs))
        self._scenarios = tuple(VectorEnvScenario(spec.name, spec.map, spec.seed) for spec in specs)
        configured_envs = tuple(
            {**configured, "map": spec.map} for spec in specs[: self._physical_slot_count]
        )
        self._envs = VectorMetaDriveEnv(
            configured_envs,
            mode=mode,
            execution_mode=ExecutionMode.ROLLOUT,
            map_query_radius_m=map_query_radius_m,
            history_warmup_steps=history_warmup_steps,
            scenarios=self._scenarios,
            torch_threads_per_worker=torch_threads_per_worker,
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
        policy_sampling: Literal["sample", "mean"] = "sample",
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
        if policy_sampling not in {"sample", "mean"}:
            raise ValueError("policy_sampling must be 'sample' or 'mean'")
        collected: list[tuple[RolloutEpisode, ...]] = []
        for start in range(0, len(self._specs), self._physical_slot_count):
            stop = min(start + self._physical_slot_count, len(self._specs))
            states, observation = self._initialize_group(
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
                    observation,
                    transitions_per_slot=transitions_per_slot,
                    stopped_speed_threshold_mps=stopped_speed_threshold_mps,
                    policy_sampling=policy_sampling,
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
    ) -> tuple[list[_SlotCollectionState], TensorDictBase]:
        slots = tuple(range(len(specs)))
        resets = self._envs.reset(
            scenarios,
            slots=None if len(specs) == self._physical_slot_count else slots,
        )
        reset_results = operation_results(resets, WorkerResetResult)
        observation = cast(TensorDictBase, resets["observation"])
        states = [
            _SlotCollectionState(
                spec=spec,
                scenario=scenario,
                diffusion_generator=diffusion_generator,
                policy_generator=policy_generator,
                noise_seed=noise_seed,
                policy_action_seed=policy_action_seed,
                lifecycle=_EpisodeLifecycle(reset.route_completion),
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
                reset_results,
                strict=True,
            )
        ]
        return states, observation

    def _collect_group(
        self,
        states: list[_SlotCollectionState],
        observation: TensorDictBase,
        *,
        transitions_per_slot: int,
        stopped_speed_threshold_mps: float,
        policy_sampling: Literal["sample", "mean"],
        timings: list[VectorRolloutRoundTiming] | None,
    ) -> tuple[tuple[RolloutEpisode, ...], ...]:
        active_slots = tuple(range(len(states)))
        full_capacity = len(states) == self._physical_slot_count
        while states[0].collected < transitions_per_slot:
            if any(state.collected != states[0].collected for state in states):
                raise RuntimeError("vector rollout consumed unequal transition counts")
            profile = timings is not None
            collate_s = 0.0
            planner_timings: list[RolloutPlannerTiming] = []
            planner_started = perf_counter() if profile else 0.0
            diffusion_generators = tuple(state.diffusion_generator for state in states)
            if policy_sampling == "sample":
                decision = self._runtime.decide_batch(
                    observation,
                    diffusion_generators,
                    tuple(state.policy_generator for state in states),
                    timings=planner_timings if profile else None,
                )
            else:
                decision = self._runtime.decide_batch_mean(
                    observation,
                    diffusion_generators,
                    timings=planner_timings if profile else None,
                )
            for slot, state in enumerate(states):
                state.lifecycle.link_next_state_value(decision.slot(slot))
            planner_s = perf_counter() - planner_started if profile else 0.0
            environment_started = perf_counter() if profile else 0.0
            steps = self._envs.step(
                decision.ego_trajectories,
                slots=None if full_capacity else active_slots,
            )
            step_results = operation_results(steps, WorkerStepResult)
            environment_s = perf_counter() - environment_started if profile else 0.0
            audit_started = perf_counter() if profile else 0.0
            if profile:
                decision.audit_result()
            audit_resolve_s = perf_counter() - audit_started if profile else 0.0
            audit_transfer = decision.audit_transfer_timing
            if profile:
                _append_decision_timing(
                    timings,
                    step_results,
                    capacity=self._physical_slot_count,
                    collate_s=collate_s,
                    planner_s=planner_s,
                    planner_timing=_single_planner_timing(planner_timings, "decision"),
                    environment_s=environment_s,
                    audit_resolve_s=audit_resolve_s,
                    audit_transfer_accelerator_s=(
                        0.0 if audit_transfer is None else audit_transfer.accelerator_s
                    ),
                )

            tails: list[tuple[int, TailKind]] = []
            for slot, (state, step_result) in enumerate(zip(states, step_results, strict=True)):
                tail = _append_slot_transition(
                    state,
                    decision.slot(slot),
                    step_result,
                    terminated=bool(steps["terminated"][slot].item()),
                    truncated=bool(steps["truncated"][slot].item()),
                    transitions_per_slot=transitions_per_slot,
                    stopped_speed_threshold_mps=stopped_speed_threshold_mps,
                    reward_evaluator=self._reward_evaluator,
                )
                if tail is not None:
                    tails.append((slot, tail))

            bootstrap_slots = [slot for slot, kind in tails if kind != "terminated"]
            bootstrap_values: dict[int, torch.Tensor] = {}
            if bootstrap_slots:
                bootstrap_collate_started = perf_counter() if profile else 0.0
                bootstrap_observation = cast(
                    TensorDictBase,
                    cast(TensorDictBase, steps["observation"])[bootstrap_slots],
                )
                bootstrap_collate_s = perf_counter() - bootstrap_collate_started if profile else 0.0
                bootstrap_timings: list[RolloutPlannerTiming] = []
                bootstrap_started = perf_counter() if timings is not None else 0.0
                values = self._runtime.bootstrap_value_batch(
                    bootstrap_observation,
                    tuple(states[slot].diffusion_generator for slot in bootstrap_slots),
                    timings=bootstrap_timings if profile else None,
                )
                if profile:
                    _append_bootstrap_timing(
                        timings,
                        active_slots=len(bootstrap_slots),
                        capacity=self._physical_slot_count,
                        collate_s=bootstrap_collate_s,
                        planner_s=perf_counter() - bootstrap_started,
                        planner_timing=_single_planner_timing(bootstrap_timings, "bootstrap"),
                    )
                for index, slot in enumerate(bootstrap_slots):
                    bootstrap_values[slot] = values[index : index + 1]

            reset_slots: list[int] = []
            for slot, kind in tails:
                state = states[slot]
                state.episodes.append(
                    state.lifecycle.finish(
                        kind,
                        torch.zeros(1) if kind == "terminated" else bootstrap_values[slot],
                    )
                )
                if state.collected < transitions_per_slot:
                    reset_slots.append(slot)
            if reset_slots:
                reset_batch = self._envs.reset(
                    tuple(states[slot].scenario for slot in reset_slots),
                    slots=reset_slots,
                )
                reset_results = operation_results(reset_batch, WorkerResetResult)
                for slot, reset_result in zip(reset_slots, reset_results, strict=True):
                    states[slot].reset(reset_result)
                observation = cast(TensorDictBase, steps["observation"]).clone()
                observation[reset_slots] = cast(TensorDictBase, reset_batch["observation"])
            else:
                observation = cast(TensorDictBase, steps["observation"])

        if any(not state.lifecycle.empty for state in states):
            raise RuntimeError("vector rollout ended with an unfinished episode")
        return tuple(tuple(state.episodes) for state in states)


def _append_slot_transition(
    state: _SlotCollectionState,
    decision: RolloutDecision,
    step: WorkerStepResult,
    *,
    terminated: bool,
    truncated: bool,
    transitions_per_slot: int,
    stopped_speed_threshold_mps: float,
    reward_evaluator: RewardEvaluator,
) -> TailKind | None:
    env_step = step.step
    tail = state.lifecycle.append(
        decision,
        env_step,
        map_seed=state.spec.seed,
        noise_seed=state.noise_seed,
        policy_action_seed=state.policy_action_seed,
        stopped_speed_threshold_mps=stopped_speed_threshold_mps,
        reward_evaluator=reward_evaluator,
        terminated=terminated,
        truncated=truncated,
        collection_limit=state.collected + 1 == transitions_per_slot,
    )
    state.collected += 1
    return tail


def _append_decision_timing(
    timings: list[VectorRolloutRoundTiming] | None,
    steps: tuple[WorkerStepResult, ...],
    *,
    capacity: int,
    collate_s: float,
    planner_s: float,
    planner_timing: RolloutPlannerTiming,
    environment_s: float,
    audit_resolve_s: float,
    audit_transfer_accelerator_s: float,
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
            collate_wall_s=collate_s,
            planner_wall_s=planner_s,
            planner_timing=planner_timing,
            environment_wall_s=environment_s,
            audit_resolve_wall_s=audit_resolve_s,
            audit_transfer_accelerator_s=audit_transfer_accelerator_s,
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
    collate_s: float,
    planner_s: float,
    planner_timing: RolloutPlannerTiming,
) -> None:
    if timings is None:
        return
    timings.append(
        VectorRolloutRoundTiming(
            phase="bootstrap",
            active_slots=active_slots,
            capacity=capacity,
            collate_wall_s=collate_s,
            planner_wall_s=planner_s,
            planner_timing=planner_timing,
            environment_wall_s=0.0,
            audit_resolve_wall_s=0.0,
            audit_transfer_accelerator_s=0.0,
            worker_busy_s=0.0,
            transport_sync_s=0.0,
            worker_imbalance_s=0.0,
        )
    )


def _single_planner_timing(
    timings: list[RolloutPlannerTiming], phase: Literal["decision", "bootstrap"]
) -> RolloutPlannerTiming:
    if len(timings) != 1 or timings[0].phase != phase:
        raise RuntimeError(f"profiled {phase} batch must return exactly one matching timing")
    return timings[0]


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
    policy_sampling: Literal["sample", "mean"] = "sample",
    timings: list[VectorRolloutRoundTiming] | None = None,
    reward_profile: RewardProfileConfig,
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
            policy_sampling=policy_sampling,
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


def _execution_transition_audit(
    step: TrajectoryExecutionResult,
    previous_route_completion: float,
    stopped_speed_threshold_mps: float,
    *,
    terminated: bool,
    truncated: bool,
    reward_evaluator: RewardEvaluator,
) -> ExecutionTransitionAudit:
    execution = step.execution
    if execution.substep_states.shape[0] != 1:
        raise RuntimeError("rollout transition must execute exactly one substep")
    if len(step.metrics) != 1:
        raise RuntimeError("rollout transition must expose exactly one transition metric")
    reward_result = reward_evaluator(step.metrics[0])
    state = execution.substep_states[0]
    distance_m = float(np.linalg.norm(state[:2] - execution.start_center))
    speed_mps = float(state[5])
    return ExecutionTransitionAudit(
        reward_result=reward_result,
        route_completion_delta=float(execution.route_completion - previous_route_completion),
        distance_m=distance_m,
        speed_mps=speed_mps,
        stopped=speed_mps < stopped_speed_threshold_mps,
        position_error_m=step.metrics[0].position_error_m,
        heading_error_rad=step.metrics[0].heading_error_rad,
        arrive_dest=execution.arrive_dest,
        out_of_road=execution.out_of_road,
        crash_vehicle=execution.crash_vehicle,
        crash_object=execution.crash_object,
        crash_building=execution.crash_building,
        crash_human=execution.crash_human,
        crash_sidewalk=execution.crash_sidewalk,
        terminated=terminated,
        truncated=truncated,
    )


def _seed(value: int, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} seed must be a non-negative integer")
    return value
