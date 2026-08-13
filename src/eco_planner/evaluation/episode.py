"""One closed-loop evaluation episode and its explicit failure boundary."""

from __future__ import annotations

import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np

from eco_planner.envs import (
    MetaDriveObservationAdapter,
    NoTrafficMetaDriveObservationAdapter,
    TrajectoryExecutionRecord,
    TrajectoryMetaDriveEnv,
)
from eco_planner.evaluation.artifacts import (
    build_episode_summary,
    build_failed_episode_summary,
    write_episode_artifacts,
)
from eco_planner.evaluation.config import EvaluationJobConfig, ScenarioConfig
from eco_planner.evaluation.failures import EpisodeFailure
from eco_planner.evaluation.rendering import render_cycle_frame
from eco_planner.evaluation.runtime import FabricInferenceRuntime
from eco_planner.evaluation.schema import CompletedEpisodeSummary, FailedEpisodeSummary
from eco_planner.evaluation.trace import EpisodeTraceRecorder


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
                "reset",
                RuntimeError(
                    f"traffic evaluation route length {episode_route_length_m} m "
                    "is outside [2000, 5000]"
                ),
            )
        trace = EpisodeTraceRecorder.from_initial_state(vehicle_state(env))
        if traffic_adapter is not None:
            traffic_adapter.reset(env.initial_traffic_frame, env=env)
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
                raw_observation = traffic_adapter.build(env)
                traffic_audit = traffic_adapter.last_audit
            elif no_traffic_adapter is not None:
                raw_observation = no_traffic_adapter.build(env)
                traffic_audit = None
            else:
                raise RuntimeError("evaluation mode did not create an observation adapter")
            observation, noise, planner_result = runtime.infer(raw_observation, generator)
            ego_trajectory = (
                planner_result.prediction[0, 0].detach().cpu().numpy().astype(np.float32)
            )
            anchor = vehicle_state(env)
            _, reward, terminated, truncated, info = env.step(ego_trajectory)
            execution = TrajectoryExecutionRecord.from_info(info)
            if traffic_adapter is not None:
                traffic_adapter.append_frames(execution.traffic_frames)
            total_reward += float(reward)
            trace.append_cycle(
                anchor,
                observation,
                noise,
                planner_result,
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
                "execution", RuntimeError("closed-loop episode ended without a simulator result")
            )
        trace_arrays = trace.finalize()
        if mode == "traffic" and not np.any(trace_arrays["traffic_participant_counts"] > 0):
            raise EpisodeFailure(
                "observation",
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
            stage=failure.stage,
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
        execution = TrajectoryExecutionRecord.from_info(info)
        frames = execution.traffic_frames
        adapter.append_frames(frames)
        trace.append_warmup(
            execution,
            np.asarray([len(frame.participants) for frame in frames], dtype=np.int64),
            np.asarray([len(frame.static_objects) for frame in frames], dtype=np.int64),
        )
        if terminated or truncated:
            raise EpisodeFailure(
                "warmup",
                RuntimeError("traffic history warmup terminated before 20 simulator steps"),
            )
    states = np.concatenate(trace.warmup_state_arrays, axis=0)
    if states.shape != (warmup_steps, 7):
        raise EpisodeFailure(
            "warmup", RuntimeError("traffic warmup did not produce the required number of states")
        )
    if float(np.linalg.norm(states[:, :2] - initial_position, axis=1).max()) >= 1e-3:
        raise EpisodeFailure(
            "warmup", RuntimeError("ego moved during stationary traffic history warmup")
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
