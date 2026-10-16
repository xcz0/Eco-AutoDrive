"""Closed-loop rollout collectors for serial and fixed-slot vector execution."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from time import perf_counter
from typing import Literal, cast
from weakref import finalize

import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.collectors import Collector

from eco_planner.configuration import ScenarioConfig
from eco_planner.contracts import CLOSED_LOOP_EXECUTION_STEPS
from eco_planner.envs import (
    MetaDriveEnvSlot,
    TrajectoryExecutionResult,
)
from eco_planner.envs.parallel import (
    VectorEnvScenario,
    VectorMetaDriveEnv,
    WorkerResetResult,
    WorkerStepResult,
    operation_results,
)
from eco_planner.reward import (
    RewardEvaluator,
    RewardProfileConfig,
    aggregate_transition_reward,
    create_reward_evaluator,
)

from .assembly import (
    RolloutEpisodeBuilder,
    build_training_transition,
)
from .contracts import (
    TRAINING_KEYS,
    ExecutionTransitionAudit,
    RolloutBatch,
    RolloutEpisode,
    RolloutProvenance,
    TailKind,
)
from .decision import BatchRolloutDecision, RolloutDecision
from .profiling import (
    RolloutPlannerTiming,
    VectorRolloutRoundTiming,
    append_bootstrap_timing,
    append_decision_timing,
    single_planner_timing,
)
from .runtime import FabricRolloutRuntime
from .seeds import RolloutRandomStreams
from .torchrl import PlannerCollectionPolicy, TorchRLCollectionEnv


@dataclass
class _EpisodeLifecycle:
    previous_route_completion: float
    builder: RolloutEpisodeBuilder = field(default_factory=RolloutEpisodeBuilder)

    @property
    def empty(self) -> bool:
        return self.builder.empty

    def append(
        self,
        decision: RolloutDecision,
        step: TrajectoryExecutionResult,
        *,
        map_seed: int,
        noise_seed: int,
        policy_action_seed: int,
        reward_evaluator: RewardEvaluator,
        terminated: bool,
        truncated: bool,
        collection_limit: bool,
    ) -> TailKind | None:
        execution = _execution_transition_audit(
            step,
            self.previous_route_completion,
            reward_evaluator=reward_evaluator,
            terminated=terminated,
            truncated=truncated,
        )
        training_transition = build_training_transition(
            decision.training_decision,
            execution.reward_result,
            terminated=terminated,
            truncated=truncated,
        )
        self.builder.append(
            training_transition,
            decision.audit_result(),
            execution,
            RolloutProvenance(
                map_seed=map_seed,
                noise_seed=noise_seed,
                policy_action_seed=policy_action_seed,
                planning_cycle_index=self.builder.transition_count,
            ),
        )
        return self.advance(
            step, terminated=terminated, truncated=truncated, collection_limit=collection_limit
        )

    def advance(
        self,
        step: TrajectoryExecutionResult,
        *,
        terminated: bool,
        truncated: bool,
        collection_limit: bool,
    ) -> TailKind | None:
        """Advance domain state after the transition and its audit have been recorded."""

        self.previous_route_completion = step.execution.route_completion
        if terminated:
            return "terminated"
        if truncated:
            return "truncated"
        if collection_limit:
            return "rollout_limit"
        return None

    def finish(self, kind: TailKind, bootstrap_value: torch.Tensor) -> RolloutEpisode:
        return self.take_builder().finish(kind, bootstrap_value)

    def take_builder(self) -> RolloutEpisodeBuilder:
        builder = self.builder
        self.builder = RolloutEpisodeBuilder()
        return builder

    def reset(self, route_completion: float) -> None:
        if not self.builder.empty:
            raise RuntimeError("cannot reset an unfinished rollout episode")
        self.previous_route_completion = route_completion


def collect_rollout_episode(
    spec: ScenarioConfig,
    runtime: FabricRolloutRuntime,
    env_config: Mapping[str, object],
    *,
    mode: Literal["no_traffic", "traffic"],
    map_query_radius_m: float,
    history_warmup_steps: int,
    max_transitions: int,
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
    configured = dict(env_config)
    configured["map"] = spec.map
    env_slot = MetaDriveEnvSlot(
        configured,
        mode=mode,
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
        diffusion_generator = runtime.new_noise_generator(resolved_noise_seed)
    if policy_generator is None:
        policy_generator = runtime.new_policy_generator(resolved_policy_seed)
    try:
        reset = env_slot.reset(map_name=spec.map, seed=spec.seed)
        current_state = reset.state
        lifecycle = _EpisodeLifecycle(current_state.route_completion)

        for cycle in range(max_transitions):
            observation = cast(TensorDictBase, TensorDictBase.stack([current_state.observation]))
            decision = runtime.decide(observation, diffusion_generator, policy_generator)
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
    streams: RolloutRandomStreams
    lifecycle: _EpisodeLifecycle
    collected: int = 0
    pending: list[tuple[RolloutEpisodeBuilder, TailKind, torch.Tensor]] = field(
        default_factory=list
    )

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
        streams: tuple[RolloutRandomStreams, ...],
        policy_sampling: Literal["sample", "mean"] = "sample",
        timings: list[VectorRolloutRoundTiming] | None = None,
    ) -> RolloutBatch:
        """Collect one PPO batch while retaining workers for a subsequent call."""

        if type(transitions_per_slot) is not int or transitions_per_slot <= 0:
            raise ValueError("transitions_per_slot must be a positive integer")
        if len(streams) != len(self._specs):
            raise ValueError("streams must contain one value per logical scenario")
        if policy_sampling not in {"sample", "mean"}:
            raise ValueError("policy_sampling must be 'sample' or 'mean'")
        training_storage = TensorDict({}, batch_size=[len(self._specs) * transitions_per_slot])
        collected: list[tuple[RolloutEpisode, ...]] = []
        for start in range(0, len(self._specs), self._physical_slot_count):
            stop = min(start + self._physical_slot_count, len(self._specs))
            states = [
                _SlotCollectionState(spec, scenario, rng, _EpisodeLifecycle(0.0))
                for spec, scenario, rng in zip(
                    self._specs[start:stop],
                    self._scenarios[start:stop],
                    streams[start:stop],
                    strict=True,
                )
            ]
            group = _CollectionGroup(
                self._envs,
                self._runtime,
                self._reward_evaluator,
                self._physical_slot_count,
                states,
                transitions_per_slot,
                policy_sampling,
                timings,
            )
            collected.extend(group.collect(training_storage, start * transitions_per_slot))
        return RolloutBatch(training_storage, tuple(collected))


@dataclass
class _CollectionGroup:
    """Transient callbacks and episode state for one logical wave of workers."""

    envs: VectorMetaDriveEnv
    runtime: FabricRolloutRuntime
    reward_evaluator: RewardEvaluator
    capacity: int
    states: list[_SlotCollectionState]
    transitions_per_slot: int
    policy_sampling: Literal["sample", "mean"]
    timings: list[VectorRolloutRoundTiming] | None
    decision: BatchRolloutDecision = field(init=False)
    planner_timings: list[RolloutPlannerTiming] = field(default_factory=list, init=False)
    planner_s: float = field(default=0.0, init=False)
    environment_started: float = field(default=0.0, init=False)

    @property
    def profiling(self) -> bool:
        return self.timings is not None

    def plan(self, carrier: TensorDictBase) -> TensorDictBase:
        self.planner_timings = []
        started = perf_counter() if self.profiling else 0.0
        diffusion = tuple(state.streams.diffusion for state in self.states)
        observation = cast(TensorDictBase, carrier["observation"])
        if self.policy_sampling == "sample":
            self.decision = self.runtime.decide_batch(
                observation,
                diffusion,
                tuple(state.streams.policy for state in self.states),
                timings=self.planner_timings if self.profiling else None,
            )
        else:
            self.decision = self.runtime.decide_batch_mean(
                observation, diffusion, timings=self.planner_timings if self.profiling else None
            )
        self.planner_s = perf_counter() - started if self.profiling else 0.0
        self.environment_started = perf_counter() if self.profiling else 0.0
        return carrier.update(self.decision.training_decision).set(
            "action", torch.from_numpy(self.decision.ego_trajectories)
        )

    def record_step(self, steps: TensorDictBase) -> torch.Tensor:
        step_results = operation_results(steps, WorkerStepResult)
        environment_s = perf_counter() - self.environment_started if self.profiling else 0.0
        audit_started = perf_counter() if self.profiling else 0.0
        if self.profiling:
            self.decision.audit_result()
        audit_resolve_s = perf_counter() - audit_started if self.profiling else 0.0
        audit_transfer = self.decision.audit_transfer_timing
        if self.profiling:
            append_decision_timing(
                self.timings,
                step_results,
                capacity=self.capacity,
                collate_s=0.0,
                planner_s=self.planner_s,
                planner_timing=single_planner_timing(self.planner_timings, "decision"),
                environment_s=environment_s,
                audit_resolve_s=audit_resolve_s,
                audit_transfer_accelerator_s=(
                    0.0 if audit_transfer is None else audit_transfer.accelerator_s
                ),
            )
        executions, tails = _record_round_audits(
            self.states,
            self.decision,
            step_results,
            terminated=steps["terminated"].reshape(-1).tolist(),
            truncated=steps["truncated"].reshape(-1).tolist(),
            transitions_per_slot=self.transitions_per_slot,
            reward_evaluator=self.reward_evaluator,
        )
        bootstrap_slots = [slot for slot, kind in tails if kind != "terminated"]
        bootstrap_values: dict[int, torch.Tensor] = {}
        if bootstrap_slots:
            started = perf_counter() if self.profiling else 0.0
            observation = cast(
                TensorDictBase, cast(TensorDictBase, steps["observation"])[bootstrap_slots]
            )
            collate_s = perf_counter() - started if self.profiling else 0.0
            bootstrap_timings: list[RolloutPlannerTiming] = []
            started = perf_counter() if self.profiling else 0.0
            values = self.runtime.bootstrap_value_batch(
                observation,
                tuple(self.states[slot].streams.diffusion for slot in bootstrap_slots),
                timings=bootstrap_timings if self.profiling else None,
            )
            if self.profiling:
                append_bootstrap_timing(
                    self.timings,
                    active_slots=len(bootstrap_slots),
                    capacity=self.capacity,
                    collate_s=collate_s,
                    planner_s=perf_counter() - started,
                    planner_timing=single_planner_timing(bootstrap_timings, "bootstrap"),
                )
            bootstrap_values = {
                slot: values[index : index + 1] for index, slot in enumerate(bootstrap_slots)
            }
        for slot, kind in tails:
            state = self.states[slot]
            state.pending.append(
                (
                    state.lifecycle.take_builder(),
                    kind,
                    torch.zeros(1) if kind == "terminated" else bootstrap_values[slot],
                )
            )
        return torch.tensor(
            [execution.reward_result.total for execution in executions],
            dtype=torch.float32,
            device=self.runtime.device,
        ).reshape(len(self.states), 1)

    def record_reset(self, slots: tuple[int, ...], resets: TensorDictBase) -> None:
        results = operation_results(resets, WorkerResetResult)
        for slot, reset in zip(slots, results, strict=True):
            self.states[slot].reset(reset)

    def training_batch(self, batch: TensorDictBase) -> TensorDictBase:
        # TorchRL stacks [slot,time]; flattening preserves logical scenario order.
        return (
            batch.select(
                *TRAINING_KEYS[:-1],
                *(("next", key) for key in ("reward", "done", "terminated", "truncated")),
            )
            .to(self.runtime.device)
            .reshape(-1)
        )

    def collect(
        self,
        training_storage: TensorDictBase,
        training_start: int,
    ) -> tuple[tuple[RolloutEpisode, ...], ...]:
        env = TorchRLCollectionEnv(
            self.envs,
            tuple(state.scenario for state in self.states),
            transitions_per_slot=self.transitions_per_slot,
            reward_device=self.runtime.device,
            on_step=self.record_step,
            on_reset=self.record_reset,
        )
        collector = Collector(
            env,
            PlannerCollectionPolicy(self.plan),
            frames_per_batch=len(self.states) * self.transitions_per_slot,
            total_frames=len(self.states) * self.transitions_per_slot,
            init_random_frames=0,
            trust_policy=True,
            use_buffers=False,
            track_traj_ids=False,
            auto_register_policy_transforms=False,
            compact_obs=True,
            postproc=self.training_batch,
        )
        try:
            batch = next(iter(collector))
        finally:
            collector.shutdown()
        if any(
            state.collected != self.transitions_per_slot or not state.lifecycle.empty
            for state in self.states
        ):
            raise RuntimeError("vector rollout ended without its complete episode quota")
        value = batch["state_value"]
        batch["next", "state_value"] = torch.empty_like(
            value, dtype=torch.promote_types(value.dtype, torch.float32)
        )
        if not list(training_storage.keys()):
            training_storage.update(batch.new_empty(training_storage.batch_size))
        section = slice(training_start, training_start + batch.batch_size[0])
        training_storage.update_at_(batch, section)
        batch = cast(TensorDictBase, training_storage[section])
        groups = []
        for slot, state in enumerate(self.states):
            offset = slot * self.transitions_per_slot
            episodes = []
            for builder, kind, bootstrap in state.pending:
                stop = offset + builder.transition_count
                episodes.append(
                    builder.finish(
                        kind, bootstrap, training=cast(TensorDictBase, batch[offset:stop])
                    )
                )
                offset = stop
            groups.append(tuple(episodes))
        return tuple(groups)


def _record_round_audits(
    states: list[_SlotCollectionState],
    decision: BatchRolloutDecision,
    steps: tuple[WorkerStepResult, ...],
    *,
    terminated: list[bool],
    truncated: list[bool],
    transitions_per_slot: int,
    reward_evaluator: RewardEvaluator,
) -> tuple[tuple[ExecutionTransitionAudit, ...], list[tuple[int, TailKind]]]:
    rows = tuple(zip(states, steps, terminated, truncated, strict=True))
    executions = tuple(
        _execution_transition_audit(
            step.step,
            state.lifecycle.previous_route_completion,
            reward_evaluator=reward_evaluator,
            terminated=terminal,
            truncated=truncation,
        )
        for state, step, terminal, truncation in rows
    )
    RolloutEpisodeBuilder.append_audit_batch(
        tuple(state.lifecycle.builder for state in states),
        decision.audit_result(),
        executions,
        tuple(
            RolloutProvenance(
                map_seed=state.spec.seed,
                noise_seed=state.streams.noise_seed,
                policy_action_seed=state.streams.policy_action_seed,
                planning_cycle_index=state.lifecycle.builder.transition_count,
            )
            for state in states
        ),
    )
    tails = []
    for slot, (state, step, terminal, truncation) in enumerate(rows):
        tail = state.lifecycle.advance(
            step.step,
            terminated=terminal,
            truncated=truncation,
            collection_limit=state.collected + 1 == transitions_per_slot,
        )
        state.collected += 1
        if tail is not None:
            tails.append((slot, tail))
    return executions, tails


def _execution_transition_audit(
    step: TrajectoryExecutionResult,
    previous_route_completion: float,
    *,
    terminated: bool,
    truncated: bool,
    reward_evaluator: RewardEvaluator,
) -> ExecutionTransitionAudit:
    execution = step.execution
    substep_count = execution.substep_states.shape[0]
    if len(step.metrics) != substep_count:
        raise RuntimeError("rollout transition metrics must match executed substeps")
    if not 1 <= substep_count <= CLOSED_LOOP_EXECUTION_STEPS:
        raise RuntimeError(
            f"rollout transition must execute between 1 and {CLOSED_LOOP_EXECUTION_STEPS} substeps"
        )
    metrics = step.metrics
    substep_results = tuple(reward_evaluator(metric) for metric in metrics)
    reward_result = aggregate_transition_reward(substep_results)
    return ExecutionTransitionAudit(
        reward_result=reward_result,
        substep_results=substep_results,
        route_completion_delta=float(execution.route_completion - previous_route_completion),
        distance_m=sum(metric.step_distance_m for metric in metrics),
        speed_mps=_mean(metric.speed_mps for metric in metrics),
        stopped=any(metric.stopped for metric in metrics),
        collision=any(metric.collision for metric in metrics),
        wrong_direction=any(metric.wrong_direction for metric in metrics),
        position_error_m=_mean(metric.position_error_m for metric in metrics),
        heading_error_rad=_mean(metric.heading_error_rad for metric in metrics),
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


def _mean(values: Iterable[float]) -> float:
    materialized = tuple(values)
    return sum(materialized) / len(materialized)


def _seed(value: int, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} seed must be a non-negative integer")
    return value
