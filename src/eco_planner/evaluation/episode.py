"""One closed-loop evaluation episode and its explicit failure boundary."""

from __future__ import annotations

import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

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
from eco_planner.evaluation.artifacts.io import write_episode_artifacts
from eco_planner.evaluation.artifacts.models import (
    CompletedEpisodeSummary,
    FailedEpisodeSummary,
    FailurePhase,
)
from eco_planner.evaluation.artifacts.trace_recorder import EpisodeTraceRecorder
from eco_planner.evaluation.config import EvaluationJobConfig, ScenarioConfig
from eco_planner.evaluation.failures import EpisodeFailure
from eco_planner.evaluation.rendering import render_cycle_frame
from eco_planner.evaluation.runtime.contracts import HostGuidanceDiagnostics, HostInferenceResult
from eco_planner.evaluation.runtime.engine import FabricInferenceRuntime
from eco_planner.evaluation.summaries import build_episode_summary, build_failed_episode_summary
from eco_planner.models import NoGuidanceConfig


@dataclass
class _VectorEvaluationSlot:
    spec: ScenarioConfig
    observation: dict[str, torch.Tensor]
    traffic_audit: TrafficObservationAudit | None
    generator: torch.Generator
    trace: EpisodeTraceRecorder
    anchor: np.ndarray
    route_length_m: float
    environment_map_audit: dict[str, object]
    saw_traffic: bool
    total_reward: float = 0.0
    plan_index: int = 0


def run_scenario(
    spec: ScenarioConfig,
    runtime: FabricInferenceRuntime,
    config: EvaluationJobConfig,
    output_root: Path,
) -> CompletedEpisodeSummary | FailedEpisodeSummary:
    env_config = dict(config.env)
    env_config["map"] = spec.map
    env: TrajectoryMetaDriveEnv | None = None
    trace: EpisodeTraceRecorder | None = None
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
    generator = runtime.new_noise_generator()
    frames: list[np.ndarray] = []
    try:
        env = TrajectoryMetaDriveEnv(env_config)
        env.reset(seed=spec.seed)
        episode_route_length_m = route_length_m(env)
        if mode == "traffic" and not 2_000.0 <= episode_route_length_m <= 5_000.0:
            raise EpisodeFailure(
                FailurePhase.RESET,
                RuntimeError(
                    f"traffic evaluation route length {episode_route_length_m} m "
                    "is outside [2000, 5000]"
                ),
            )
        trace = EpisodeTraceRecorder.from_initial_state(
            vehicle_state(env),
            max_plan_cycles=(config.evaluation.evaluated_horizon_steps + 4) // 5,
            max_warmup_steps=config.evaluation.history_warmup_steps,
            guided=not isinstance(runtime.guidance_config, NoGuidanceConfig),
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
        total_reward = 0.0
        final_execution: TrajectoryExecutionRecord | None = None
        plan_index = 0
        while not terminated and not truncated:
            if traffic_adapter is not None:
                raw_observation, traffic_audit = traffic_adapter.build(env)
            elif no_traffic_adapter is not None:
                raw_observation = no_traffic_adapter.build(env)
                traffic_audit = None
            else:
                raise RuntimeError("evaluation mode did not create an observation adapter")
            inference = runtime.infer(collate_observations([raw_observation]), generator)
            ego_trajectory = inference.ego_trajectory
            anchor = vehicle_state(env)
            _, reward, terminated, truncated, info = env.step(ego_trajectory)
            execution = info["trajectory_execution"]
            if traffic_adapter is not None:
                traffic_adapter.append_frames(execution.traffic_frames)
            total_reward += float(reward)
            audit_result = inference.audit_result()
            trace.append_cycle(
                anchor,
                raw_observation,
                audit_result,
                execution,
                plan_index,
                traffic_audit,
            )
            if config.video.enabled:
                frames.append(
                    render_cycle_frame(env, execution, anchor[:2], config.video, plan_index)
                )
            final_execution = execution
            plan_index += 1
        if final_execution is None:
            raise EpisodeFailure(
                FailurePhase.EXECUTION,
                RuntimeError("closed-loop episode ended without a simulator result"),
            )
        trace_arrays = trace.finalize()
        if mode == "traffic" and not np.any(trace_arrays["traffic_participant_counts"] > 0):
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
            env.programmatic_lane_speed_limit_audit,
            mode,
            float(config.env["traffic_density"]),
            episode_route_length_m,
            asdict(runtime.sampler_report),
            asdict(runtime.guidance_config),
        )
        write_episode_artifacts(
            output_root / spec.name, trace_arrays, frames, summary, config.video
        )
        return summary
    except EpisodeFailure as failure:
        trace_status = "partial" if trace is not None and trace.has_recorded_steps else "empty"
        recorder = trace if trace is not None else EpisodeTraceRecorder.empty()
        trace_arrays = recorder.finalize(trace_status)
        summary = build_failed_episode_summary(
            _scenario_payload(spec),
            noise_seed=runtime.report.seed,
            evaluation_mode=mode,
            traffic_density=float(config.env["traffic_density"]),
            sampler=asdict(runtime.sampler_report),
            guidance=asdict(runtime.guidance_config),
            trace_status=trace_status,
            phase=failure.phase,
            cause=failure.cause,
            traceback_text=traceback.format_exc(),
        )
        write_episode_artifacts(
            output_root / spec.name, trace_arrays, frames, summary, config.video
        )
        return summary
    finally:
        if env is not None:
            env.close()


def run_vector_scenarios(
    specs: tuple[ScenarioConfig, ...],
    runtime: FabricInferenceRuntime,
    config: EvaluationJobConfig,
    output_root: Path,
) -> tuple[CompletedEpisodeSummary | FailedEpisodeSummary, ...]:
    """Evaluate fixed scenario slots with one centralized batch planner runtime."""

    if not specs:
        raise ValueError("vector evaluation requires at least one scenario")
    if config.video.enabled:
        raise ValueError("vector evaluation requires video.enabled=false")
    mode = config.evaluation.mode
    configured_envs = tuple({**config.env, "map": spec.map} for spec in specs)
    scenarios = tuple(VectorEnvScenario(spec.name, spec.map, spec.seed) for spec in specs)
    with VectorMetaDriveEnv(
        configured_envs,
        mode=mode,
        model_config=runtime.planner_config,
        map_query_radius_m=config.map_query_radius_m,
        history_warmup_steps=config.evaluation.history_warmup_steps,
    ) as envs:
        resets = envs.reset(scenarios)
        slots = [_initialize_vector_slot(reset, runtime, config) for reset in resets]
        active = list(range(len(slots)))
        summaries: list[CompletedEpisodeSummary | FailedEpisodeSummary | None] = [None] * len(slots)
        while active:
            observations = [slots[index].observation for index in active]
            generators = tuple(slots[index].generator for index in active)
            noise = _batch_noise(runtime, generators)
            decision = runtime.infer_batch(collate_observations(observations), noise, generators)
            steps = envs.step_slots(active, decision.ego_trajectories)
            audit = decision.audit_result()
            next_active: list[int] = []
            for batch_index, step in enumerate(steps):
                slot_index = active[batch_index]
                slot = slots[slot_index]
                slot.trace.append_cycle(
                    slot.anchor,
                    slot.observation,
                    _slice_inference(audit, batch_index),
                    step.execution,
                    slot.plan_index,
                    slot.traffic_audit,
                )
                slot.saw_traffic = slot.saw_traffic or _has_traffic(slot.traffic_audit)
                slot.total_reward += step.reward
                slot.plan_index += 1
                if step.terminated or step.truncated:
                    try:
                        summaries[slot_index] = _complete_vector_slot(
                            slot,
                            step.execution,
                            step.terminated,
                            step.truncated,
                            runtime,
                            config,
                            output_root,
                        )
                    except EpisodeFailure as failure:
                        summaries[slot_index] = _fail_vector_slot(
                            slot, failure, runtime, config, output_root
                        )
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
) -> _VectorEvaluationSlot:
    if config.evaluation.mode == "traffic" and not 2_000.0 <= reset.route_length_m <= 5_000.0:
        raise EpisodeFailure(
            FailurePhase.RESET,
            RuntimeError(
                f"traffic evaluation route length {reset.route_length_m} m is outside [2000, 5000]"
            ),
        )
    trace = EpisodeTraceRecorder.from_initial_state(
        reset.warmup_initial_state,
        max_plan_cycles=(config.evaluation.evaluated_horizon_steps + 4) // 5,
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
    return _VectorEvaluationSlot(
        ScenarioConfig(
            name=reset.scenario.name,
            map=reset.scenario.map,
            seed=reset.scenario.seed,
        ),
        reset.observation,
        reset.traffic_audit,
        runtime.new_noise_generator(),
        trace,
        reset.initial_state,
        reset.route_length_m,
        dict(reset.programmatic_lane_speed_limit_audit),
        _has_traffic(reset.traffic_audit),
    )


def _batch_noise(
    runtime: FabricInferenceRuntime, generators: tuple[torch.Generator, ...]
) -> torch.Tensor:
    config = runtime.planner_config
    return torch.cat(
        [
            torch.randn(
                (1, 1 + config.predicted_neighbor_num, config.future_len, 4),
                dtype=torch.float32,
                device=runtime.device,
                generator=generator,
            )
            for generator in generators
        ]
    )


def _slice_inference(result: HostInferenceResult, index: int) -> HostInferenceResult:
    diagnostics = result.guidance_diagnostics
    sliced_diagnostics = (
        None
        if diagnostics is None
        else HostGuidanceDiagnostics(
            **{name: value[index : index + 1] for name, value in vars(diagnostics).items()}
        )
    )
    return HostInferenceResult(
        initial_noise=result.initial_noise[index : index + 1],
        prediction=result.prediction[index : index + 1],
        reference_prediction=(
            None
            if result.reference_prediction is None
            else result.reference_prediction[index : index + 1]
        ),
        guidance_action=(
            None if result.guidance_action is None else result.guidance_action[index : index + 1]
        ),
        guidance_diagnostics=sliced_diagnostics,
    )


def _state_from_execution(execution: TrajectoryExecutionRecord) -> np.ndarray:
    state = execution.substep_states[-1]
    if state.shape != (7,):
        raise RuntimeError("trajectory execution did not return a [7] final state")
    return np.asarray(state, dtype=np.float64).copy()


def _has_traffic(audit: TrafficObservationAudit | None) -> bool:
    return audit is not None and audit.participant_count_in_radius > 0


def _complete_vector_slot(
    slot: _VectorEvaluationSlot,
    final_execution: TrajectoryExecutionRecord,
    terminated: bool,
    truncated: bool,
    runtime: FabricInferenceRuntime,
    config: EvaluationJobConfig,
    output_root: Path,
) -> CompletedEpisodeSummary:
    if config.evaluation.mode == "traffic" and not slot.saw_traffic:
        raise EpisodeFailure(
            FailurePhase.OBSERVATION,
            RuntimeError("traffic evaluation never observed a participant within radius"),
        )
    trace_arrays = slot.trace.finalize()
    summary = build_episode_summary(
        _scenario_payload(slot.spec),
        trace_arrays,
        final_execution,
        terminated,
        truncated,
        slot.total_reward,
        runtime.report.seed,
        slot.environment_map_audit,
        config.evaluation.mode,
        float(config.env["traffic_density"]),
        slot.route_length_m,
        asdict(runtime.sampler_report),
        asdict(runtime.guidance_config),
    )
    write_episode_artifacts(output_root / slot.spec.name, trace_arrays, [], summary, config.video)
    return summary


def _fail_vector_slot(
    slot: _VectorEvaluationSlot,
    failure: EpisodeFailure,
    runtime: FabricInferenceRuntime,
    config: EvaluationJobConfig,
    output_root: Path,
) -> FailedEpisodeSummary:
    trace_status = "partial" if slot.trace.has_recorded_steps else "empty"
    trace_arrays = slot.trace.finalize(trace_status)
    summary = build_failed_episode_summary(
        _scenario_payload(slot.spec),
        noise_seed=runtime.report.seed,
        evaluation_mode=config.evaluation.mode,
        traffic_density=float(config.env["traffic_density"]),
        sampler=asdict(runtime.sampler_report),
        guidance=asdict(runtime.guidance_config),
        trace_status=trace_status,
        phase=failure.phase,
        cause=failure.cause,
        traceback_text=traceback.format_exc(),
    )
    write_episode_artifacts(output_root / slot.spec.name, trace_arrays, [], summary, config.video)
    return summary


def run_traffic_warmup(
    env: TrajectoryMetaDriveEnv,
    adapter: MetaDriveObservationAdapter,
    trace: EpisodeTraceRecorder,
    warmup_steps: int,
) -> None:
    if warmup_steps % 5 != 0:
        raise ValueError("history warmup steps must be divisible by five")
    initial_position = trace.warmup_initial_state[:2].copy()
    for _ in range(warmup_steps // 5):
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
                RuntimeError("traffic history warmup terminated before 20 simulator steps"),
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
    trajectory = np.zeros((80, 4), dtype=np.float32)
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
