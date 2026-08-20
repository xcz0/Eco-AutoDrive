"""Serial 10 Hz closed-loop rollout collection for a single scenario."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import numpy as np
import torch

from eco_planner.envs import (
    MetaDriveObservationAdapter,
    NoTrafficMetaDriveObservationAdapter,
    TrajectoryExecutionRecord,
    TrajectoryMetaDriveEnv,
)
from eco_planner.evaluation.config import ScenarioConfig
from eco_planner.rl.rollout import (
    RolloutEpisode,
    build_rollout_audit,
    build_training_transition,
    finalize_rollout_episode,
)
from eco_planner.rl.runtime import FabricRolloutRuntime


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
    if configured.get("trajectory_execution_steps") != 1:
        raise ValueError("rollout requires env.trajectory_execution_steps=1")
    env = TrajectoryMetaDriveEnv(configured)
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
    traffic_adapter = (
        MetaDriveObservationAdapter(runtime.planner_config, map_query_radius_m)
        if mode == "traffic"
        else None
    )
    no_traffic_adapter = (
        NoTrafficMetaDriveObservationAdapter(runtime.planner_config, map_query_radius_m)
        if mode == "no_traffic"
        else None
    )
    training_transitions = []
    audit_transitions = []
    try:
        env.reset(seed=spec.seed)
        if traffic_adapter is not None:
            traffic_adapter.reset(env.initial_traffic_frame, env=env)
            _warmup_traffic(env, traffic_adapter, history_warmup_steps)
        elif no_traffic_adapter is not None:
            if history_warmup_steps != 0:
                raise ValueError("no-traffic rollout requires zero history warmup steps")
            no_traffic_adapter.reset(env)
        previous_route_completion = env.route_completion

        for cycle in range(max_transitions):
            observation = _build_observation(traffic_adapter, no_traffic_adapter, env)
            decision = runtime.decide(observation, diffusion_generator, policy_generator)
            training_decision = decision.training_decision
            _, _, terminated, truncated, info = env.step(decision.ego_trajectory)
            audit_result = decision.audit_result()
            execution = info["trajectory_execution"]
            if execution.substep_states.shape[0] != 1:
                raise RuntimeError("rollout transition must execute exactly one substep")
            if traffic_adapter is not None:
                traffic_adapter.append_frames(execution.traffic_frames)
            audit = _transition_audit(
                execution,
                previous_route_completion,
                stopped_speed_threshold_mps,
            )
            reward = float(execution.substep_rewards.sum())
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
                    dense_reward=float(execution.substep_dense_rewards.sum()),
                    terminal_override=float(
                        (execution.substep_rewards - execution.substep_dense_rewards).sum()
                    ),
                    terminated=terminated,
                    truncated=truncated,
                    map_seed=spec.seed,
                    noise_seed=resolved_noise_seed,
                    policy_action_seed=resolved_policy_seed,
                    planning_cycle_index=cycle,
                    **audit,
                )
            )
            previous_route_completion = execution.route_completion
            if terminated:
                return finalize_rollout_episode(
                    training_transitions, audit_transitions, "terminated", torch.zeros(1)
                )
            if truncated:
                next_observation = _build_observation(traffic_adapter, no_traffic_adapter, env)
                return finalize_rollout_episode(
                    training_transitions,
                    audit_transitions,
                    "truncated",
                    runtime.bootstrap_value(next_observation, diffusion_generator),
                )
        next_observation = _build_observation(traffic_adapter, no_traffic_adapter, env)
        return finalize_rollout_episode(
            training_transitions,
            audit_transitions,
            "rollout_limit",
            runtime.bootstrap_value(next_observation, diffusion_generator),
        )
    finally:
        env.close()


def _build_observation(
    traffic_adapter: MetaDriveObservationAdapter | None,
    no_traffic_adapter: NoTrafficMetaDriveObservationAdapter | None,
    env: TrajectoryMetaDriveEnv,
) -> Mapping[str, torch.Tensor]:
    if traffic_adapter is not None:
        return traffic_adapter.build(env)
    if no_traffic_adapter is not None:
        return no_traffic_adapter.build(env)
    raise RuntimeError("rollout did not create an observation adapter")


def _warmup_traffic(
    env: TrajectoryMetaDriveEnv,
    adapter: MetaDriveObservationAdapter,
    required_steps: int,
) -> None:
    collected = 0
    while collected < required_steps:
        _, _, terminated, truncated, info = env.step(_stationary_trajectory())
        execution = info["trajectory_execution"]
        if terminated or truncated:
            raise RuntimeError("traffic history warmup ended before the required frame count")
        adapter.append_frames(execution.traffic_frames)
        collected += execution.substep_states.shape[0]
    if collected != required_steps:
        raise RuntimeError("traffic history warmup overshot the required frame count")


def _stationary_trajectory() -> np.ndarray:
    trajectory = np.zeros((80, 4), dtype=np.float32)
    trajectory[:, 2] = 1.0
    return trajectory


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


def _seed(value: int, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} seed must be a non-negative integer")
    return value
