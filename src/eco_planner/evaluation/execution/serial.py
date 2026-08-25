"""Single-environment closed-loop evaluation execution."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from eco_planner.envs import (
    MetaDriveObservationAdapter,
    NoTrafficMetaDriveObservationAdapter,
    TrajectoryExecutionRecord,
    TrajectoryMetaDriveEnv,
    collate_observations,
)
from eco_planner.evaluation.artifacts.models import (
    CompletedEpisodeSummary,
    FailedEpisodeSummary,
    FailurePhase,
)
from eco_planner.evaluation.artifacts.trace_recorder import EpisodeTraceRecorder
from eco_planner.evaluation.config import EvaluationJobConfig, ScenarioConfig
from eco_planner.evaluation.episode import (
    finalize_completed_episode,
    persist_failed_episode,
    route_length_m,
    run_traffic_warmup,
    vehicle_state,
)
from eco_planner.evaluation.failures import EpisodeFailure
from eco_planner.evaluation.rendering import render_cycle_frame
from eco_planner.evaluation.runtime.engine import FabricInferenceRuntime
from eco_planner.models import NoGuidanceConfig


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
    generator = runtime.new_noise_generator()
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
            anchor = vehicle_state(env)
            _, reward, terminated, truncated, info = env.step(inference.ego_trajectory)
            execution = info["trajectory_execution"]
            if traffic_adapter is not None:
                traffic_adapter.append_frames(execution.traffic_frames)
            total_reward += float(reward)
            trace.append_cycle(
                anchor,
                raw_observation,
                inference.audit_result(),
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
            total_reward,
            env.programmatic_lane_speed_limit_audit,
            episode_route_length_m,
            bool(np.any(trace_arrays["traffic_participant_counts"] > 0)),
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
