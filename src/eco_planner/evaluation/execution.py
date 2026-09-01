"""Serial and fixed-slot vector closed-loop evaluation execution."""

from __future__ import annotations

import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import numpy as np
from tensordict import TensorDictBase

from eco_planner.envs import (
    MetaDriveEnvSlot,
    PlannerObservationSpec,
    TrafficObservationAudit,
    TrajectoryExecutionRecord,
    VectorEnvReset,
    VectorEnvScenario,
    VectorMetaDriveEnv,
    collate_observations,
)
from eco_planner.envs.array_types import SingleObservation
from eco_planner.evaluation.agent import EvaluationAgent
from eco_planner.evaluation.artifacts import write_episode_artifacts
from eco_planner.evaluation.config import EvaluationJobConfig, ScenarioConfig
from eco_planner.evaluation.models import (
    CompletedEpisodeSummary,
    FailedEpisodeSummary,
    FailurePhase,
)
from eco_planner.evaluation.rendering import render_cycle_frame
from eco_planner.evaluation.summaries import build_episode_summary, build_failed_episode_summary
from eco_planner.evaluation.trace import EpisodeTraceRecorder
from eco_planner.execution_contracts import evaluation_plan_cycles


class EpisodeFailure(RuntimeError):
    """A classified episode failure that may be persisted before continuing the job."""

    def __init__(self, phase: FailurePhase, cause: Exception) -> None:
        self.phase = phase
        self.cause = cause
        super().__init__(f"{phase.value}: {cause}")


@dataclass
class _EpisodeState:
    spec: ScenarioConfig
    observation: SingleObservation | None
    traffic_audit: TrafficObservationAudit | None
    agent_episode: object
    trace: EpisodeTraceRecorder
    anchor: np.ndarray
    route_length_m: float
    environment_map_audit: dict[str, object]
    saw_traffic: bool = False
    total_reward: float = 0.0
    plan_index: int = 0

    def record_cycle(
        self,
        observation: SingleObservation,
        inference: TensorDictBase,
        execution: TrajectoryExecutionRecord,
        reward: float,
        traffic_audit: TrafficObservationAudit | None,
    ) -> int:
        cycle = self.plan_index
        self.trace.append_cycle(
            self.anchor,
            observation,
            inference,
            execution,
            cycle,
            traffic_audit,
        )
        self.saw_traffic = self.saw_traffic or _has_traffic(traffic_audit)
        self.total_reward += float(reward)
        self.plan_index += 1
        return cycle


def run_scenario(
    spec: ScenarioConfig,
    agent: EvaluationAgent,
    config: EvaluationJobConfig,
    output_root: Path,
    *,
    scenario_index: int = 0,
) -> CompletedEpisodeSummary | FailedEpisodeSummary:
    """Evaluate one scenario in one MetaDrive environment."""

    env_config = dict(config.env)
    env_config["map"] = spec.map
    env_slot: MetaDriveEnvSlot | None = None
    trace: EpisodeTraceRecorder | None = None
    finalized_trace_arrays: dict[str, np.ndarray] | None = None
    mode = config.evaluation.mode
    state: _EpisodeState | None = None
    frames: list[np.ndarray] = []
    try:
        env_slot = MetaDriveEnvSlot(
            env_config,
            mode=mode,
            observation_spec=PlannerObservationSpec.from_planner_config(agent.planner_config),
            map_query_radius_m=config.map_query_radius_m,
            history_warmup_steps=config.evaluation.history_warmup_steps,
        )
        reset = env_slot.reset(map_name=spec.map, seed=spec.seed)
        episode_route_length_m = reset.route_length_m
        if mode == "traffic" and not 2_000.0 <= episode_route_length_m <= 5_000.0:
            raise EpisodeFailure(
                phase=FailurePhase.RESET,
                cause=RuntimeError(
                    f"traffic evaluation route length {episode_route_length_m} m "
                    "is outside [2000, 5000]"
                ),
            )
        trace = EpisodeTraceRecorder.from_initial_state(
            reset.warmup_initial_state,
            max_plan_cycles=evaluation_plan_cycles(config.evaluation.evaluated_horizon_steps),
            max_warmup_steps=config.evaluation.history_warmup_steps,
            guided=agent.guided,
        )
        state = _EpisodeState(
            spec=spec,
            observation=None,
            traffic_audit=None,
            agent_episode=agent.new_episode(scenario_index),
            trace=trace,
            anchor=reset.warmup_initial_state.copy(),
            route_length_m=episode_route_length_m,
            environment_map_audit=dict(reset.programmatic_lane_speed_limit_audit),
        )
        if mode == "traffic":
            try:
                for execution in env_slot.warmup():
                    traffic_frames = execution.traffic_frames
                    trace.append_warmup(
                        execution,
                        np.asarray(
                            [len(frame.participants) for frame in traffic_frames], dtype=np.int64
                        ),
                        np.asarray(
                            [len(frame.static_objects) for frame in traffic_frames], dtype=np.int64
                        ),
                    )
            except Exception as error:
                raise EpisodeFailure(FailurePhase.WARMUP, error) from error
            trace.replace_initial_state(env_slot.vehicle_state)

        terminated = False
        truncated = False
        final_execution: TrajectoryExecutionRecord | None = None
        while not terminated and not truncated:
            slot_observation = env_slot.observe()
            raw_observation = slot_observation.observation
            traffic_audit = slot_observation.traffic_audit
            inference = agent.decide_batch(
                collate_observations([raw_observation]), (state.agent_episode,)
            )
            state.anchor = env_slot.vehicle_state
            step = env_slot.step(np.asarray(inference.ego_trajectories)[0])
            terminated = step.terminated
            truncated = step.truncated
            execution = step.execution
            cycle = state.record_cycle(
                raw_observation,
                _audit_slot(inference.audit_result(), 0),
                execution,
                step.reward,
                traffic_audit,
            )
            if config.video.enabled:
                frames.append(
                    render_cycle_frame(
                        env_slot.env, execution, state.anchor[:2], config.video, cycle
                    )
                )
            final_execution = execution
        if final_execution is None:
            raise EpisodeFailure(
                phase=FailurePhase.EXECUTION,
                cause=RuntimeError("closed-loop episode ended without a simulator result"),
            )
        trace_arrays = trace.finalize()
        finalized_trace_arrays = trace_arrays
        return finalize_completed_episode(
            spec,
            trace_arrays,
            final_execution,
            terminated,
            truncated,
            state.total_reward,
            state.environment_map_audit,
            state.route_length_m,
            state.saw_traffic,
            agent,
            config,
            output_root,
            frames,
            scenario_index=scenario_index,
        )
    except EpisodeFailure as failure:
        return persist_failed_episode(
            spec,
            trace,
            failure,
            agent,
            config,
            output_root,
            frames,
            finalized_trace_arrays,
            scenario_index=scenario_index,
        )
    finally:
        if env_slot is not None:
            env_slot.close()


def finalize_completed_episode(
    spec: ScenarioConfig,
    trace_arrays: dict[str, np.ndarray],
    final_execution: TrajectoryExecutionRecord,
    terminated: bool,
    truncated: bool,
    total_reward: float,
    environment_map_audit: dict[str, object],
    route_length_m: float,
    saw_traffic: bool,
    agent: EvaluationAgent,
    config: EvaluationJobConfig,
    output_root: Path,
    frames: list[np.ndarray],
    *,
    scenario_index: int,
) -> CompletedEpisodeSummary:
    """Build and persist a completed episode from its execution trace."""

    if config.evaluation.mode == "traffic" and not saw_traffic:
        raise EpisodeFailure(
            FailurePhase.OBSERVATION,
            RuntimeError("traffic evaluation never observed a participant within radius"),
        )
    summary = build_episode_summary(
        _scenario_payload(spec),
        trace_arrays,
        final_execution,
        terminated,
        truncated,
        total_reward,
        agent.noise_seed(scenario_index),
        environment_map_audit,
        config.evaluation.mode,
        float(config.env["traffic_density"]),
        route_length_m,
        asdict(agent.sampler_report),
        asdict(agent.guidance_config),
    )
    write_episode_artifacts(output_root / spec.name, trace_arrays, frames, summary, config.video)
    return summary


def persist_failed_episode(
    spec: ScenarioConfig,
    trace: EpisodeTraceRecorder | None,
    failure: EpisodeFailure,
    agent: EvaluationAgent,
    config: EvaluationJobConfig,
    output_root: Path,
    frames: list[np.ndarray],
    finalized_trace_arrays: dict[str, np.ndarray] | None = None,
    *,
    scenario_index: int,
) -> FailedEpisodeSummary:
    """Persist the available trace and failure metadata through one common path."""

    trace_status = "partial" if trace is not None and trace.has_recorded_steps else "empty"
    if finalized_trace_arrays is None:
        recorder = trace if trace is not None else EpisodeTraceRecorder.empty()
        trace_arrays = recorder.finalize(trace_status)
    else:
        trace_arrays = dict(finalized_trace_arrays)
        trace_arrays["trace_status"] = np.asarray(trace_status)
    summary = build_failed_episode_summary(
        _scenario_payload(spec),
        noise_seed=agent.noise_seed(scenario_index),
        evaluation_mode=config.evaluation.mode,
        traffic_density=float(config.env["traffic_density"]),
        sampler=asdict(agent.sampler_report),
        guidance=asdict(agent.guidance_config),
        trace_status=trace_status,
        phase=failure.phase,
        cause=failure.cause,
        traceback_text=traceback.format_exc(),
        trace_arrays=trace_arrays,
    )
    write_episode_artifacts(output_root / spec.name, trace_arrays, frames, summary, config.video)
    return summary


def _scenario_payload(spec: ScenarioConfig) -> dict[str, object]:
    return {"name": spec.name, "map_sequence": spec.map, "seed": spec.seed}


def run_vector_scenarios(
    specs: tuple[ScenarioConfig, ...],
    agent: EvaluationAgent,
    config: EvaluationJobConfig,
    output_root: Path,
    *,
    vector_env_slots: int,
    torch_threads_per_worker: int | None,
) -> tuple[CompletedEpisodeSummary | FailedEpisodeSummary, ...]:
    """Evaluate a scenario queue with persistent fixed slots and batched planning."""

    if not specs:
        raise ValueError("vector evaluation requires at least one scenario")
    if config.video.enabled:
        raise ValueError("vector evaluation requires video.enabled=false")
    slot_count = min(vector_env_slots, len(specs))
    initial_specs = specs[:slot_count]
    configured_envs = tuple({**config.env, "map": spec.map} for spec in initial_specs)
    scenarios = tuple(VectorEnvScenario(spec.name, spec.map, spec.seed) for spec in specs)
    with VectorMetaDriveEnv(
        configured_envs,
        mode=config.evaluation.mode,
        observation_spec=PlannerObservationSpec.from_planner_config(agent.planner_config),
        map_query_radius_m=config.map_query_radius_m,
        history_warmup_steps=config.evaluation.history_warmup_steps,
        scenarios=scenarios,
        torch_threads_per_worker=torch_threads_per_worker,
    ) as envs:
        resets = envs.reset(scenarios[:slot_count])
        slots: dict[int, _EpisodeState] = {}
        slot_scenario_indices: dict[int, int] = {}
        summaries: list[CompletedEpisodeSummary | FailedEpisodeSummary | None] = [None] * len(specs)
        next_scenario_index = slot_count

        def assign_next(slot_index: int) -> bool:
            nonlocal next_scenario_index
            while next_scenario_index < len(specs):
                scenario_index = next_scenario_index
                spec = specs[scenario_index]
                next_scenario_index += 1
                reset = envs.reset_at(slot_index, VectorEnvScenario(spec.name, spec.map, spec.seed))
                try:
                    slots[slot_index] = _initialize_vector_slot(
                        reset, agent, config, scenario_index
                    )
                except EpisodeFailure as failure:
                    summaries[scenario_index] = persist_failed_episode(
                        spec,
                        None,
                        failure,
                        agent,
                        config,
                        output_root,
                        [],
                        scenario_index=scenario_index,
                    )
                    continue
                slot_scenario_indices[slot_index] = scenario_index
                return True
            return False

        for slot_index, reset in enumerate(resets):
            try:
                slots[slot_index] = _initialize_vector_slot(reset, agent, config, slot_index)
            except EpisodeFailure as failure:
                summaries[slot_index] = persist_failed_episode(
                    initial_specs[slot_index],
                    None,
                    failure,
                    agent,
                    config,
                    output_root,
                    [],
                    scenario_index=slot_index,
                )
                assign_next(slot_index)
            else:
                slot_scenario_indices[slot_index] = slot_index
        active = list(slots)
        while active:
            observations = [_state_observation(slots[index]) for index in active]
            decision = agent.decide_batch(
                collate_observations(observations),
                tuple(slots[index].agent_episode for index in active),
            )
            steps = envs.step_slots(active, decision.ego_trajectories)
            audit = decision.audit_result()
            next_active: list[int] = []
            for batch_index, step in enumerate(steps):
                slot_index = active[batch_index]
                slot = slots[slot_index]
                slot.record_cycle(
                    _state_observation(slot),
                    _audit_slot(audit, batch_index),
                    step.execution,
                    step.reward,
                    slot.traffic_audit,
                )
                if step.terminated or step.truncated:
                    scenario_index = slot_scenario_indices.pop(slot_index)
                    trace_arrays = slot.trace.finalize()
                    try:
                        summaries[scenario_index] = finalize_completed_episode(
                            slot.spec,
                            trace_arrays,
                            step.execution,
                            step.terminated,
                            step.truncated,
                            slot.total_reward,
                            slot.environment_map_audit,
                            slot.route_length_m,
                            slot.saw_traffic,
                            agent,
                            config,
                            output_root,
                            [],
                            scenario_index=scenario_index,
                        )
                    except EpisodeFailure as failure:
                        summaries[scenario_index] = persist_failed_episode(
                            slot.spec,
                            slot.trace,
                            failure,
                            agent,
                            config,
                            output_root,
                            [],
                            trace_arrays,
                            scenario_index=scenario_index,
                        )
                    if assign_next(slot_index):
                        next_active.append(slot_index)
                    continue
                slot.observation = step.observation
                slot.traffic_audit = step.traffic_audit
                slot.anchor = _state_from_execution(step.execution)
                next_active.append(slot_index)
            active = next_active
    return cast(
        tuple[CompletedEpisodeSummary | FailedEpisodeSummary, ...],
        tuple(summaries),
    )


def _initialize_vector_slot(
    reset: VectorEnvReset,
    agent: EvaluationAgent,
    config: EvaluationJobConfig,
    scenario_index: int,
) -> _EpisodeState:
    if config.evaluation.mode == "traffic" and not 2_000.0 <= reset.route_length_m <= 5_000.0:
        raise EpisodeFailure(
            FailurePhase.RESET,
            RuntimeError(
                f"traffic evaluation route length {reset.route_length_m} m is outside [2000, 5000]"
            ),
        )
    trace = EpisodeTraceRecorder.from_initial_state(
        reset.warmup_initial_state,
        max_plan_cycles=evaluation_plan_cycles(config.evaluation.evaluated_horizon_steps),
        max_warmup_steps=config.evaluation.history_warmup_steps,
        guided=agent.guided,
    )
    for execution in reset.warmup_executions:
        trace.append_warmup(
            execution,
            np.asarray(
                [len(frame.participants) for frame in execution.traffic_frames], dtype=np.int64
            ),
            np.asarray(
                [len(frame.static_objects) for frame in execution.traffic_frames], dtype=np.int64
            ),
        )
    trace.replace_initial_state(reset.initial_state)
    return _EpisodeState(
        spec=ScenarioConfig(
            name=reset.scenario.name,
            map=reset.scenario.map,
            seed=reset.scenario.seed,
        ),
        observation=reset.observation,
        traffic_audit=reset.traffic_audit,
        agent_episode=agent.new_episode(scenario_index),
        trace=trace,
        anchor=reset.initial_state,
        route_length_m=reset.route_length_m,
        environment_map_audit=dict(reset.programmatic_lane_speed_limit_audit),
        saw_traffic=_has_traffic(reset.traffic_audit),
    )


def _state_from_execution(execution: TrajectoryExecutionRecord) -> np.ndarray:
    state = execution.substep_states[-1]
    return np.asarray(state, dtype=np.float64).copy()


def _has_traffic(audit: TrafficObservationAudit | None) -> bool:
    return audit is not None and audit.participant_count_in_radius > 0


def _state_observation(state: _EpisodeState) -> SingleObservation:
    return cast(SingleObservation, state.observation)


def _audit_slot(audit: TensorDictBase, index: int) -> TensorDictBase:
    return cast(TensorDictBase, audit[index : index + 1])
