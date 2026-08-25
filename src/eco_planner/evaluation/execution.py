"""Serial and fixed-slot vector closed-loop evaluation execution."""

from __future__ import annotations

import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
from tensordict import TensorDictBase

from eco_planner.envs import (
    MetaDriveObservationAdapter,
    NoTrafficMetaDriveObservationAdapter,
    TrafficObservationAudit,
    TrajectoryExecutionRecord,
    TrajectoryMetaDriveEnv,
    VectorEnvReset,
    VectorEnvScenario,
    VectorMetaDriveEnv,
    collate_observations,
)
from eco_planner.evaluation.artifacts import (
    CompletedEpisodeSummary,
    FailedEpisodeSummary,
    FailurePhase,
    write_episode_artifacts,
)
from eco_planner.evaluation.config import EvaluationJobConfig, ScenarioConfig
from eco_planner.evaluation.rendering import render_cycle_frame
from eco_planner.evaluation.runtime import FabricInferenceRuntime
from eco_planner.evaluation.summaries import build_episode_summary, build_failed_episode_summary
from eco_planner.evaluation.trace import EpisodeTraceRecorder
from eco_planner.execution_contracts import (
    EVALUATION_EXECUTION_STEPS,
    PLANNER_FUTURE_STEPS,
    TRAFFIC_HISTORY_STEPS,
    evaluation_plan_cycles,
)
from eco_planner.models import NoGuidanceConfig


class EpisodeFailure(RuntimeError):
    """A classified episode failure that may be persisted before continuing the job."""

    def __init__(self, phase: FailurePhase, cause: Exception) -> None:
        if not isinstance(phase, FailurePhase):
            raise TypeError("episode failure phase must be a FailurePhase")
        if not isinstance(cause, Exception):
            raise TypeError("episode failure cause must be an Exception")
        self.phase = phase
        self.cause = cause
        super().__init__(f"{phase.value}: {cause}")


@dataclass
class _EpisodeState:
    spec: ScenarioConfig
    observation: dict[str, torch.Tensor] | None
    traffic_audit: TrafficObservationAudit | None
    generator: torch.Generator
    trace: EpisodeTraceRecorder
    anchor: np.ndarray
    route_length_m: float
    environment_map_audit: dict[str, object]
    saw_traffic: bool = False
    total_reward: float = 0.0
    plan_index: int = 0

    def record_cycle(
        self,
        observation: dict[str, torch.Tensor],
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
    runtime: FabricInferenceRuntime,
    config: EvaluationJobConfig,
    output_root: Path,
) -> CompletedEpisodeSummary | FailedEpisodeSummary:
    """Evaluate one scenario in one MetaDrive environment."""

    env_config = dict(config.env)
    env_config["map"] = spec.map
    env: TrajectoryMetaDriveEnv | None = None
    trace: EpisodeTraceRecorder | None = None
    finalized_trace_arrays: dict[str, np.ndarray] | None = None
    mode = config.evaluation.mode
    traffic_adapter = (
        MetaDriveObservationAdapter(runtime.planner_config, config.map_query_radius_m)
        if mode == "traffic"
        else None
    )
    no_traffic_adapter = (
        NoTrafficMetaDriveObservationAdapter(runtime.planner_config, config.map_query_radius_m)
        if mode == "no_traffic"
        else None
    )
    state: _EpisodeState | None = None
    frames: list[np.ndarray] = []
    try:
        env = TrajectoryMetaDriveEnv(env_config)
        env.reset(seed=spec.seed)
        episode_route_length_m = route_length_m(env)
        if mode == "traffic" and not 2_000.0 <= episode_route_length_m <= 5_000.0:
            raise EpisodeFailure(
                phase=FailurePhase.RESET,
                cause=RuntimeError(
                    f"traffic evaluation route length {episode_route_length_m} m "
                    "is outside [2000, 5000]"
                ),
            )
        trace = EpisodeTraceRecorder.from_initial_state(
            vehicle_state(env),
            max_plan_cycles=evaluation_plan_cycles(config.evaluation.evaluated_horizon_steps),
            max_warmup_steps=config.evaluation.history_warmup_steps,
            guided=not isinstance(runtime.guidance_config, NoGuidanceConfig),
        )
        state = _EpisodeState(
            spec=spec,
            observation=None,
            traffic_audit=None,
            generator=runtime.new_noise_generator(),
            trace=trace,
            anchor=vehicle_state(env),
            route_length_m=episode_route_length_m,
            environment_map_audit=env.programmatic_lane_speed_limit_audit,
        )
        if traffic_adapter is not None:
            traffic_adapter.reset(env, env.initial_traffic_frame)
            run_traffic_warmup(
                env,
                traffic_adapter,
                trace,
                config.evaluation.history_warmup_steps,
            )
        elif no_traffic_adapter is not None:
            no_traffic_adapter.reset(env)

        terminated = False
        truncated = False
        final_execution: TrajectoryExecutionRecord | None = None
        while not terminated and not truncated:
            if traffic_adapter is not None:
                raw_observation, traffic_audit = traffic_adapter.build(env)
            elif no_traffic_adapter is not None:
                raw_observation = no_traffic_adapter.build(env)
                traffic_audit = None
            else:
                raise RuntimeError("evaluation mode did not create an observation adapter")
            inference = runtime.infer(collate_observations([raw_observation]), state.generator)
            state.anchor = vehicle_state(env)
            _, reward, terminated, truncated, info = env.step(inference.ego_trajectory)
            execution = info["trajectory_execution"]
            if traffic_adapter is not None:
                traffic_adapter.append_frames(execution.traffic_frames)
            cycle = state.record_cycle(
                raw_observation,
                inference.audit_result(),
                execution,
                float(reward),
                traffic_audit,
            )
            if config.video.enabled:
                frames.append(
                    render_cycle_frame(env, execution, state.anchor[:2], config.video, cycle)
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
            runtime,
            config,
            output_root,
            frames,
        )
    except EpisodeFailure as failure:
        return persist_failed_episode(
            spec,
            trace,
            failure,
            runtime,
            config,
            output_root,
            frames,
            finalized_trace_arrays,
        )
    finally:
        if env is not None:
            env.close()


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
    runtime: FabricInferenceRuntime,
    config: EvaluationJobConfig,
    output_root: Path,
    frames: list[np.ndarray],
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
        runtime.report.seed,
        environment_map_audit,
        config.evaluation.mode,
        float(config.env["traffic_density"]),
        route_length_m,
        asdict(runtime.sampler_report),
        asdict(runtime.guidance_config),
    )
    write_episode_artifacts(output_root / spec.name, trace_arrays, frames, summary, config.video)
    return summary


def persist_failed_episode(
    spec: ScenarioConfig,
    trace: EpisodeTraceRecorder | None,
    failure: EpisodeFailure,
    runtime: FabricInferenceRuntime,
    config: EvaluationJobConfig,
    output_root: Path,
    frames: list[np.ndarray],
    finalized_trace_arrays: dict[str, np.ndarray] | None = None,
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
        noise_seed=runtime.report.seed,
        evaluation_mode=config.evaluation.mode,
        traffic_density=float(config.env["traffic_density"]),
        sampler=asdict(runtime.sampler_report),
        guidance=asdict(runtime.guidance_config),
        trace_status=trace_status,
        phase=failure.phase,
        cause=failure.cause,
        traceback_text=traceback.format_exc(),
        trace_arrays=trace_arrays,
    )
    write_episode_artifacts(output_root / spec.name, trace_arrays, frames, summary, config.video)
    return summary


def run_traffic_warmup(
    env: TrajectoryMetaDriveEnv,
    adapter: MetaDriveObservationAdapter,
    trace: EpisodeTraceRecorder,
    warmup_steps: int,
) -> None:
    if warmup_steps % EVALUATION_EXECUTION_STEPS != 0:
        raise ValueError(f"history warmup steps must be divisible by {EVALUATION_EXECUTION_STEPS}")
    initial_position = trace.warmup_initial_state[:2].copy()
    for _ in range(warmup_steps // EVALUATION_EXECUTION_STEPS):
        _, _, terminated, truncated, info = env.step(stationary_trajectory())
        execution = info["trajectory_execution"]
        frames = execution.traffic_frames
        adapter.append_frames(frames)
        trace.append_warmup(
            execution,
            np.asarray([len(frame.participants) for frame in frames], dtype=np.int64),
            np.asarray([len(frame.static_objects) for frame in frames], dtype=np.int64),
        )
        if terminated or truncated:
            raise EpisodeFailure(
                FailurePhase.WARMUP,
                RuntimeError(
                    "traffic history warmup terminated before "
                    f"{TRAFFIC_HISTORY_STEPS} simulator steps"
                ),
            )
    states = np.concatenate(trace.warmup_state_arrays, axis=0)
    if states.shape != (warmup_steps, 7):
        raise EpisodeFailure(
            FailurePhase.WARMUP,
            RuntimeError("traffic warmup did not produce the required number of states"),
        )
    if float(np.linalg.norm(states[:, :2] - initial_position, axis=1).max()) >= 1e-3:
        raise EpisodeFailure(
            FailurePhase.WARMUP,
            RuntimeError("ego moved during stationary traffic history warmup"),
        )
    trace.replace_initial_state(vehicle_state(env))


def stationary_trajectory() -> np.ndarray:
    trajectory = np.zeros((PLANNER_FUTURE_STEPS, 4), dtype=np.float32)
    trajectory[:, 2] = 1.0
    return trajectory


def vehicle_state(env: TrajectoryMetaDriveEnv) -> np.ndarray:
    velocity = np.asarray(env.agent.velocity, dtype=np.float64)
    return np.array(
        [
            *np.asarray(env.agent.position, dtype=np.float64),
            float(env.agent.heading_theta),
            *velocity,
            float(env.agent.speed),
            0.0,
        ],
        dtype=np.float64,
    )


def route_length_m(env: TrajectoryMetaDriveEnv) -> float:
    checkpoints = list(env.agent.navigation.checkpoints)
    if len(checkpoints) < 2:
        raise RuntimeError("MetaDrive navigation did not expose a complete route")
    graph = env.current_map.road_network.graph
    edge_lengths: list[float] = []
    for start, end in zip(checkpoints[:-1], checkpoints[1:], strict=True):
        lanes = graph.get(start, {}).get(end, [])
        if not lanes:
            raise RuntimeError(f"route edge {(start, end)!r} has no lane")
        lane_length = getattr(lanes[0], "length", None)
        if isinstance(lane_length, (bool, np.bool_)) or not isinstance(
            lane_length, (int, float, np.integer, np.floating)
        ):
            raise RuntimeError(f"route edge {(start, end)!r} has an invalid length")
        if not np.isfinite(lane_length) or float(lane_length) <= 0.0:
            raise RuntimeError(f"route edge {(start, end)!r} has an invalid length")
        edge_lengths.append(float(lane_length))
    return float(sum(edge_lengths))


def _scenario_payload(spec: ScenarioConfig) -> dict[str, object]:
    return {"name": spec.name, "map_sequence": spec.map, "seed": spec.seed}


def run_vector_scenarios(
    specs: tuple[ScenarioConfig, ...],
    runtime: FabricInferenceRuntime,
    config: EvaluationJobConfig,
    output_root: Path,
) -> tuple[CompletedEpisodeSummary | FailedEpisodeSummary, ...]:
    """Evaluate a scenario queue with persistent fixed slots and batched planning."""

    if not specs:
        raise ValueError("vector evaluation requires at least one scenario")
    if config.video.enabled:
        raise ValueError("vector evaluation requires video.enabled=false")
    configured_slots = config.evaluation.execution.vector_env_slots or len(specs)
    slot_count = min(configured_slots, len(specs))
    initial_specs = specs[:slot_count]
    configured_envs = tuple({**config.env, "map": spec.map} for spec in initial_specs)
    initial_scenarios = tuple(
        VectorEnvScenario(spec.name, spec.map, spec.seed) for spec in initial_specs
    )
    with VectorMetaDriveEnv(
        configured_envs,
        mode=config.evaluation.mode,
        model_config=runtime.planner_config,
        map_query_radius_m=config.map_query_radius_m,
        history_warmup_steps=config.evaluation.history_warmup_steps,
    ) as envs:
        resets = envs.reset(initial_scenarios)
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
                    slots[slot_index] = _initialize_vector_slot(reset, runtime, config)
                except EpisodeFailure as failure:
                    summaries[scenario_index] = persist_failed_episode(
                        spec,
                        None,
                        failure,
                        runtime,
                        config,
                        output_root,
                        [],
                    )
                    continue
                slot_scenario_indices[slot_index] = scenario_index
                return True
            return False

        for slot_index, reset in enumerate(resets):
            try:
                slots[slot_index] = _initialize_vector_slot(reset, runtime, config)
            except EpisodeFailure as failure:
                summaries[slot_index] = persist_failed_episode(
                    initial_specs[slot_index],
                    None,
                    failure,
                    runtime,
                    config,
                    output_root,
                    [],
                )
                assign_next(slot_index)
            else:
                slot_scenario_indices[slot_index] = slot_index
        active = list(slots)
        while active:
            observations = [
                cast(dict[str, torch.Tensor], slots[index].observation) for index in active
            ]
            generators = tuple(slots[index].generator for index in active)
            noise = runtime.sample_noise(generators)
            decision = runtime.infer_batch(collate_observations(observations), noise, generators)
            steps = envs.step_slots(active, decision.ego_trajectories)
            audit = decision.audit_result()
            next_active: list[int] = []
            for batch_index, step in enumerate(steps):
                slot_index = active[batch_index]
                slot = slots[slot_index]
                slot.record_cycle(
                    cast(dict[str, torch.Tensor], slot.observation),
                    audit[batch_index : batch_index + 1],
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
                            runtime,
                            config,
                            output_root,
                            [],
                        )
                    except EpisodeFailure as failure:
                        summaries[scenario_index] = persist_failed_episode(
                            slot.spec,
                            slot.trace,
                            failure,
                            runtime,
                            config,
                            output_root,
                            [],
                            trace_arrays,
                        )
                    if assign_next(slot_index):
                        next_active.append(slot_index)
                    continue
                slot.observation = step.observation
                slot.traffic_audit = step.traffic_audit
                slot.anchor = _state_from_execution(step.execution)
                next_active.append(slot_index)
            active = next_active
    if any(summary is None for summary in summaries):
        raise RuntimeError("vector evaluation ended with an unfinished scenario")
    return tuple(summary for summary in summaries if summary is not None)


def _initialize_vector_slot(
    reset: VectorEnvReset,
    runtime: FabricInferenceRuntime,
    config: EvaluationJobConfig,
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
        guided=not isinstance(runtime.guidance_config, NoGuidanceConfig),
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
        generator=runtime.new_noise_generator(),
        trace=trace,
        anchor=reset.initial_state,
        route_length_m=reset.route_length_m,
        environment_map_audit=dict(reset.programmatic_lane_speed_limit_audit),
        saw_traffic=_has_traffic(reset.traffic_audit),
    )


def _state_from_execution(execution: TrajectoryExecutionRecord) -> np.ndarray:
    state = execution.substep_states[-1]
    if state.shape != (7,):
        raise RuntimeError("trajectory execution did not return a [7] final state")
    return np.asarray(state, dtype=np.float64).copy()


def _has_traffic(audit: TrafficObservationAudit | None) -> bool:
    return audit is not None and audit.participant_count_in_radius > 0
