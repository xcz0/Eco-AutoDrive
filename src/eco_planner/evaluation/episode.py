"""Shared closed-loop evaluation episode lifecycle and artifact finalization."""

from __future__ import annotations

import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np

from eco_planner.envs import (
    MetaDriveObservationAdapter,
    TrajectoryExecutionRecord,
    TrajectoryMetaDriveEnv,
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
from eco_planner.evaluation.runtime.engine import FabricInferenceRuntime
from eco_planner.evaluation.summaries import build_episode_summary, build_failed_episode_summary
from eco_planner.execution_contracts import (
    EVALUATION_EXECUTION_STEPS,
    PLANNER_FUTURE_STEPS,
    TRAFFIC_HISTORY_STEPS,
)


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
