"""Serial closed-loop evaluation episode execution."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from eco_planner.contracts import evaluation_plan_cycles
from eco_planner.envs import (
    MetaDriveEnvSlot,
    PlannerObservationSpec,
    TrajectoryExecutionRecord,
    collate_observations,
)

from ..artifacts import CompletedEpisodeSummary, FailedEpisodeSummary, FailurePhase
from ..config import EvaluationJobConfig, ScenarioConfig
from ..inference import EvaluationAgent
from .lifecycle import (
    EpisodeFailure,
    EpisodeState,
    audit_slot,
    finalize_completed_episode,
    persist_failed_episode,
)
from .recorder import EpisodeTraceRecorder
from .rendering import render_cycle_frame


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
    state: EpisodeState | None = None
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
        state = EpisodeState(
            spec=spec,
            observation=None,
            traffic_audit=None,
            noise_generator=agent.new_noise_generator(scenario_index),
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
                collate_observations([raw_observation]), (state.noise_generator,)
            )
            state.anchor = env_slot.vehicle_state
            step = env_slot.step(np.asarray(inference.ego_trajectories)[0])
            terminated = step.terminated
            truncated = step.truncated
            execution = step.execution
            cycle = state.record_cycle(
                raw_observation,
                audit_slot(inference.audit_result(), 0),
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
