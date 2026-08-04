"""Fixed-seed official Diffusion Planner evaluation in MetaDrive."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from metadrive.utils.doc_utils import generate_gif
from omegaconf import DictConfig, OmegaConf

from eco_planner.envs import (
    MetaDriveObservationAdapter,
    NoTrafficMetaDriveObservationAdapter,
    TrafficFrame,
    TrafficObservationAudit,
    TrajectoryMetaDriveEnv,
)
from eco_planner.models.pretrained import (
    CheckpointLoadReport,
    PretrainedDiffusionPlanner,
    load_official_diffusion_planner,
)


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    map_sequence: str
    seed: int


@dataclass
class EpisodeTrace:
    warmup_initial_state: np.ndarray
    warmup_states: list[np.ndarray]
    warmup_rewards: list[np.ndarray]
    warmup_terminated: list[np.ndarray]
    warmup_truncated: list[np.ndarray]
    warmup_participant_counts: list[np.ndarray]
    warmup_static_object_counts: list[np.ndarray]
    initial_state: np.ndarray
    planning_anchors: list[np.ndarray]
    noises: list[np.ndarray]
    predictions_local: list[np.ndarray]
    observation_ego_current_state: list[np.ndarray]
    observation_neighbor_agents_past: list[np.ndarray]
    observation_static_objects: list[np.ndarray]
    observation_lanes: list[np.ndarray]
    observation_lanes_speed_limit: list[np.ndarray]
    observation_lanes_has_speed_limit: list[np.ndarray]
    observation_route_lanes: list[np.ndarray]
    ego_world: list[np.ndarray]
    substep_states: list[np.ndarray]
    substep_rewards: list[np.ndarray]
    substep_terminated: list[np.ndarray]
    substep_truncated: list[np.ndarray]
    substep_plan_indices: list[np.ndarray]
    target_centers: list[np.ndarray]
    target_headings: list[np.ndarray]
    position_errors_m: list[np.ndarray]
    heading_errors_rad: list[np.ndarray]
    traffic_selected_ids: list[np.ndarray]
    traffic_participant_counts: list[int]
    traffic_static_object_counts: list[int]
    traffic_nearest_distance_m: list[float]
    traffic_has_nearest: list[bool]


def _parse_scenarios(config: DictConfig) -> list[ScenarioSpec]:
    scenarios: list[ScenarioSpec] = []
    names: set[str] = set()
    for raw in config.scenarios:
        name = raw.name
        map_sequence = raw.map
        seed = raw.seed
        if not isinstance(name, str) or not name:
            raise ValueError("every evaluation scenario must have a non-empty name")
        if name in names:
            raise ValueError(f"duplicate evaluation scenario name: {name!r}")
        if not isinstance(map_sequence, str) or not map_sequence:
            raise ValueError(f"scenario {name!r} must have a non-empty map sequence")
        if type(seed) is not int or seed < 0:
            raise ValueError(f"scenario {name!r} seed must be a non-negative integer")
        scenarios.append(ScenarioSpec(name, map_sequence, seed))
        names.add(name)
    if not scenarios:
        raise ValueError("at least one evaluation scenario is required")
    return scenarios


def _validate_evaluation_config(config: DictConfig) -> None:
    mode = config.evaluation.mode
    if mode not in {"no_traffic", "traffic"}:
        raise ValueError("evaluation.mode must be 'no_traffic' or 'traffic'")
    warmup_steps = config.evaluation.history_warmup_steps
    evaluated_steps = config.evaluation.evaluated_horizon_steps
    if type(warmup_steps) is not int or warmup_steps < 0:
        raise ValueError("evaluation.history_warmup_steps must be a non-negative integer")
    if type(evaluated_steps) is not int or evaluated_steps <= 0:
        raise ValueError("evaluation.evaluated_horizon_steps must be a positive integer")
    if config.env.horizon != warmup_steps + evaluated_steps:
        raise ValueError("env.horizon must equal history_warmup_steps + evaluated_horizon_steps")
    if mode == "no_traffic" and warmup_steps != 0:
        raise ValueError("no-traffic evaluation requires zero history warmup steps")
    if mode == "traffic":
        if warmup_steps != 20:
            raise ValueError("traffic evaluation requires exactly 20 history warmup steps")
        if config.env.traffic_mode != "trigger":
            raise ValueError("traffic evaluation requires traffic_mode='trigger'")
        if type(config.env.traffic_density) not in {int, float}:
            raise TypeError("traffic_density must be numeric")
        if not 0.0 < float(config.env.traffic_density) <= 1.0:
            raise ValueError("traffic evaluation requires traffic_density in (0, 1]")
        if config.env.random_traffic is not False:
            raise ValueError("traffic evaluation requires random_traffic=false")
        if config.env.accident_prob != 0.0:
            raise ValueError("traffic evaluation requires accident_prob=0")
    if type(config.model.seed) is not int or config.model.seed < 0:
        raise ValueError("model.seed must be a non-negative integer")
    if type(config.map_query_radius_m) not in {int, float} or config.map_query_radius_m <= 0:
        raise ValueError("map_query_radius_m must be positive")
    if type(config.video.enabled) is not bool:
        raise TypeError("video.enabled must be a boolean")
    for name in ("screen_width", "screen_height", "film_width", "film_height"):
        value = config.video[name]
        if type(value) is not int or value <= 0:
            raise ValueError(f"video.{name} must be a positive integer")
    if type(config.video.scaling) not in {int, float} or config.video.scaling <= 0:
        raise ValueError("video.scaling must be positive")
    if type(config.video.fps) is not int or config.video.fps <= 0:
        raise ValueError("video.fps must be a positive integer")


def _stationary_trajectory() -> np.ndarray:
    trajectory = np.zeros((80, 4), dtype=np.float32)
    trajectory[:, 2] = 1.0
    return trajectory


def _traffic_frames(info: dict[str, Any]) -> tuple[TrafficFrame, ...]:
    frames = info.get("traffic_substep_frames")
    if not isinstance(frames, tuple) or not frames:
        raise RuntimeError("environment did not return traffic substep frames")
    if not all(isinstance(frame, TrafficFrame) for frame in frames):
        raise RuntimeError("environment returned invalid traffic substep frame values")
    return frames


def _run_traffic_warmup(
    env: TrajectoryMetaDriveEnv,
    adapter: MetaDriveObservationAdapter,
    trace: EpisodeTrace,
    warmup_steps: int,
) -> None:
    if warmup_steps % 5 != 0:
        raise ValueError("history warmup steps must be divisible by five")
    initial_position = trace.warmup_initial_state[:2].copy()
    for _ in range(warmup_steps // 5):
        _, _, terminated, truncated, info = env.step(_stationary_trajectory())
        frames = _traffic_frames(info)
        adapter.append_frames(frames)
        trace.warmup_states.append(np.asarray(info["trajectory_substep_states"], dtype=np.float64))
        trace.warmup_rewards.append(
            np.asarray(info["trajectory_substep_rewards"], dtype=np.float64)
        )
        trace.warmup_terminated.append(
            np.asarray(info["trajectory_substep_terminated"], dtype=np.bool_)
        )
        trace.warmup_truncated.append(
            np.asarray(info["trajectory_substep_truncated"], dtype=np.bool_)
        )
        trace.warmup_participant_counts.append(
            np.asarray([len(frame.participants) for frame in frames], dtype=np.int64)
        )
        trace.warmup_static_object_counts.append(
            np.asarray([len(frame.static_objects) for frame in frames], dtype=np.int64)
        )
        if terminated or truncated:
            raise RuntimeError("traffic history warmup terminated before 20 simulator steps")
    states = np.concatenate(trace.warmup_states, axis=0)
    if states.shape != (warmup_steps, 7):
        raise RuntimeError("traffic warmup did not produce the required number of states")
    displacements = np.linalg.norm(states[:, :2] - initial_position, axis=1)
    if float(displacements.max()) >= 1e-3:
        raise RuntimeError("ego moved during stationary traffic history warmup")
    trace.initial_state = _initial_vehicle_state(env)


def _initial_vehicle_state(env: TrajectoryMetaDriveEnv) -> np.ndarray:
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


def _route_length_m(env: TrajectoryMetaDriveEnv) -> float:
    checkpoints = list(env.agent.navigation.checkpoints)
    if len(checkpoints) < 2:
        raise RuntimeError("MetaDrive navigation did not expose a complete route")
    graph = env.current_map.road_network.graph
    edge_lengths: list[float] = []
    for start, end in zip(checkpoints[:-1], checkpoints[1:]):
        lanes = graph.get(start, {}).get(end, [])
        if not lanes:
            raise RuntimeError(f"route edge {(start, end)!r} has no lane")
        lane_length = getattr(lanes[0], "length", None)
        if isinstance(lane_length, (bool, np.bool_)) or not isinstance(
            lane_length, (int, float, np.integer, np.floating)
        ):
            raise RuntimeError(f"route edge {(start, end)!r} has an invalid length")
        if not np.isfinite(lane_length):
            raise RuntimeError(f"route edge {(start, end)!r} has an invalid length")
        if float(lane_length) <= 0.0:
            raise RuntimeError(f"route edge {(start, end)!r} has a non-positive length")
        edge_lengths.append(float(lane_length))
    return float(sum(edge_lengths))


def _new_episode_trace(
    env: TrajectoryMetaDriveEnv,
    *,
    warmup_initial_state: np.ndarray | None = None,
) -> EpisodeTrace:
    initial = _initial_vehicle_state(env)
    return EpisodeTrace(
        warmup_initial_state=initial.copy()
        if warmup_initial_state is None
        else warmup_initial_state.copy(),
        warmup_states=[],
        warmup_rewards=[],
        warmup_terminated=[],
        warmup_truncated=[],
        warmup_participant_counts=[],
        warmup_static_object_counts=[],
        initial_state=initial,
        planning_anchors=[],
        noises=[],
        predictions_local=[],
        observation_ego_current_state=[],
        observation_neighbor_agents_past=[],
        observation_static_objects=[],
        observation_lanes=[],
        observation_lanes_speed_limit=[],
        observation_lanes_has_speed_limit=[],
        observation_route_lanes=[],
        ego_world=[],
        substep_states=[],
        substep_rewards=[],
        substep_terminated=[],
        substep_truncated=[],
        substep_plan_indices=[],
        target_centers=[],
        target_headings=[],
        position_errors_m=[],
        heading_errors_rad=[],
        traffic_selected_ids=[],
        traffic_participant_counts=[],
        traffic_static_object_counts=[],
        traffic_nearest_distance_m=[],
        traffic_has_nearest=[],
    )


def _noise_for_planner(
    planner: PretrainedDiffusionPlanner,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    config = planner.config
    return torch.randn(
        (
            1,
            1 + config.predicted_neighbor_num,
            config.future_len,
            4,
        ),
        dtype=torch.float32,
        device=device,
        generator=generator,
    )


def _world_prediction(info: dict[str, Any]) -> np.ndarray:
    centers = np.asarray(info["trajectory_world_centers"], dtype=np.float64)
    headings = np.asarray(info["trajectory_world_headings"], dtype=np.float64)
    if centers.shape != (80, 2) or headings.shape != (80,):
        raise RuntimeError("environment returned an invalid world trajectory")
    return np.column_stack((centers, np.cos(headings), np.sin(headings)))


def _append_cycle(
    trace: EpisodeTrace,
    anchor: np.ndarray,
    observation: dict[str, torch.Tensor],
    noise: torch.Tensor,
    prediction: torch.Tensor,
    info: dict[str, Any],
    plan_index: int,
    traffic_audit: TrafficObservationAudit | None,
) -> None:
    substep_states = np.asarray(info["trajectory_substep_states"], dtype=np.float64)
    substep_count = substep_states.shape[0]
    if substep_states.ndim != 2 or substep_states.shape[1] != 7:
        raise RuntimeError("environment returned invalid trajectory substep states")
    raw_observation = _raw_observation_for_trace(observation)
    target_centers = np.asarray(info["trajectory_target_centers"], dtype=np.float64)
    target_headings = np.asarray(info["trajectory_target_headings"], dtype=np.float64)
    position_errors_m = np.asarray(info["trajectory_position_errors_m"], dtype=np.float64)
    heading_errors_rad = np.asarray(info["trajectory_heading_errors_rad"], dtype=np.float64)
    expected_shape = (substep_count,)
    if target_centers.shape != (substep_count, 2):
        raise RuntimeError("environment returned invalid trajectory target centers")
    if target_headings.shape != expected_shape:
        raise RuntimeError("environment returned invalid trajectory target headings")
    if position_errors_m.shape != expected_shape or heading_errors_rad.shape != expected_shape:
        raise RuntimeError("environment returned invalid trajectory execution errors")
    trace.planning_anchors.append(anchor)
    trace.noises.append(noise.detach().cpu().numpy())
    trace.predictions_local.append(prediction.detach().cpu().numpy())
    trace.observation_ego_current_state.append(raw_observation["ego_current_state"])
    trace.observation_neighbor_agents_past.append(raw_observation["neighbor_agents_past"])
    trace.observation_static_objects.append(raw_observation["static_objects"])
    trace.observation_lanes.append(raw_observation["lanes"])
    trace.observation_lanes_speed_limit.append(raw_observation["lanes_speed_limit"])
    trace.observation_lanes_has_speed_limit.append(raw_observation["lanes_has_speed_limit"])
    trace.observation_route_lanes.append(raw_observation["route_lanes"])
    trace.ego_world.append(_world_prediction(info))
    trace.substep_states.append(substep_states)
    trace.substep_rewards.append(np.asarray(info["trajectory_substep_rewards"], dtype=np.float64))
    trace.substep_terminated.append(
        np.asarray(info["trajectory_substep_terminated"], dtype=np.bool_)
    )
    trace.substep_truncated.append(np.asarray(info["trajectory_substep_truncated"], dtype=np.bool_))
    trace.substep_plan_indices.append(np.full(substep_count, plan_index, dtype=np.int64))
    trace.target_centers.append(target_centers)
    trace.target_headings.append(target_headings)
    trace.position_errors_m.append(position_errors_m)
    trace.heading_errors_rad.append(heading_errors_rad)
    selected_ids = np.full(32, "", dtype="<U64")
    if traffic_audit is None:
        participant_count = 0
        static_count = 0
        nearest_distance = 0.0
        has_nearest = False
    else:
        ids = traffic_audit.selected_participant_ids
        if len(ids) > selected_ids.size:
            raise RuntimeError("traffic observation selected more than 32 participants")
        selected_ids[: len(ids)] = ids
        participant_count = traffic_audit.participant_count_in_radius
        static_count = traffic_audit.static_object_count_in_radius
        nearest = traffic_audit.nearest_participant_distance_m
        nearest_distance = 0.0 if nearest is None else nearest
        has_nearest = nearest is not None
    trace.traffic_selected_ids.append(selected_ids)
    trace.traffic_participant_counts.append(participant_count)
    trace.traffic_static_object_counts.append(static_count)
    trace.traffic_nearest_distance_m.append(nearest_distance)
    trace.traffic_has_nearest.append(has_nearest)


def _raw_observation_for_trace(observation: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    names = (
        "ego_current_state",
        "neighbor_agents_past",
        "static_objects",
        "lanes",
        "lanes_speed_limit",
        "lanes_has_speed_limit",
        "route_lanes",
    )
    raw: dict[str, np.ndarray] = {}
    for name in names:
        value = observation.get(name)
        if not isinstance(value, torch.Tensor) or value.ndim < 1 or value.shape[0] != 1:
            raise ValueError(f"raw observation {name} must be a batch-one torch tensor")
        raw[name] = value.detach().cpu().numpy()[0].copy()
    return raw


def _draw_segment(
    frame: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
    width: int,
) -> None:
    delta_x = end[0] - start[0]
    delta_y = end[1] - start[1]
    samples = max(abs(delta_x), abs(delta_y)) + 1
    xs = np.rint(np.linspace(start[0], end[0], samples)).astype(np.int64)
    ys = np.rint(np.linspace(start[1], end[1], samples)).astype(np.int64)
    height, frame_width = frame.shape[:2]
    for offset_x in range(-width, width + 1):
        for offset_y in range(-width, width + 1):
            draw_x = xs + offset_x
            draw_y = ys + offset_y
            valid = (draw_x >= 0) & (draw_x < frame_width) & (draw_y >= 0) & (draw_y < height)
            frame[draw_y[valid], draw_x[valid], :3] = color


def _draw_world_polyline(
    frame: np.ndarray,
    points: np.ndarray,
    camera_position: np.ndarray,
    scaling: float,
    color: tuple[int, int, int],
    width: int,
) -> None:
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("polyline points must have shape [N, 2]")
    if points.shape[0] < 2:
        return
    height, frame_width = frame.shape[:2]
    pixels = np.empty((points.shape[0], 2), dtype=np.int64)
    pixels[:, 0] = np.rint(frame_width / 2 + (points[:, 0] - camera_position[0]) * scaling)
    pixels[:, 1] = np.rint(height / 2 - (points[:, 1] - camera_position[1]) * scaling)
    for start, end in zip(pixels[:-1], pixels[1:]):
        _draw_segment(frame, tuple(start), tuple(end), color, width)


def _render_cycle_frame(
    env: TrajectoryMetaDriveEnv,
    info: dict[str, Any],
    anchor_position: np.ndarray,
    video_config: DictConfig,
    plan_index: int,
) -> np.ndarray:
    frame = env.render(
        text={"plan_cycle": plan_index, "route_completion": info["route_completion"]},
        mode="top_down",
        screen_size=(video_config.screen_width, video_config.screen_height),
        film_size=(video_config.film_width, video_config.film_height),
        scaling=float(video_config.scaling),
        num_stack=20,
        history_smooth=0,
        window=False,
        center_on_map=False,
    )
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] < 3:
        raise RuntimeError("MetaDrive top-down renderer did not return an RGB frame")
    rendered = frame.copy()
    camera_position = np.asarray(env.agent.position, dtype=np.float64)
    planned = np.vstack(
        (anchor_position, np.asarray(info["trajectory_world_centers"], dtype=np.float64))
    )
    executed = np.vstack(
        (
            anchor_position,
            np.asarray(info["trajectory_substep_states"], dtype=np.float64)[:, :2],
        )
    )
    _draw_world_polyline(
        rendered,
        planned,
        camera_position,
        float(video_config.scaling),
        (40, 90, 240),
        1,
    )
    _draw_world_polyline(
        rendered,
        executed,
        camera_position,
        float(video_config.scaling),
        (30, 210, 80),
        2,
    )
    return rendered


def _terminal_reason(info: dict[str, Any], terminated: bool, truncated: bool) -> str:
    ordered_flags = (
        ("arrive_dest", "arrive_dest"),
        ("out_of_road", "out_of_road"),
        ("crash_vehicle", "crash_vehicle"),
        ("crash_object", "crash_object"),
        ("crash_building", "crash_building"),
        ("crash_human", "crash_human"),
        ("max_step", "max_step"),
    )
    for key, reason in ordered_flags:
        if bool(info[key]):
            return reason
    if truncated:
        return "truncated"
    if terminated:
        return "terminated"
    raise RuntimeError("episode ended without a terminal reason")


def _stack_trace(trace: EpisodeTrace) -> dict[str, np.ndarray]:
    if not trace.noises or not trace.substep_states:
        raise RuntimeError("cannot save an empty closed-loop trace")
    warmup_states = (
        np.concatenate(trace.warmup_states, axis=0)
        if trace.warmup_states
        else np.empty((0, 7), dtype=np.float64)
    )
    warmup_rewards = (
        np.concatenate(trace.warmup_rewards)
        if trace.warmup_rewards
        else np.empty(0, dtype=np.float64)
    )
    warmup_terminated = (
        np.concatenate(trace.warmup_terminated)
        if trace.warmup_terminated
        else np.empty(0, dtype=np.bool_)
    )
    warmup_truncated = (
        np.concatenate(trace.warmup_truncated)
        if trace.warmup_truncated
        else np.empty(0, dtype=np.bool_)
    )
    warmup_participant_counts = (
        np.concatenate(trace.warmup_participant_counts)
        if trace.warmup_participant_counts
        else np.empty(0, dtype=np.int64)
    )
    warmup_static_counts = (
        np.concatenate(trace.warmup_static_object_counts)
        if trace.warmup_static_object_counts
        else np.empty(0, dtype=np.int64)
    )
    return {
        "warmup_initial_state": trace.warmup_initial_state,
        "warmup_states": warmup_states,
        "warmup_rewards": warmup_rewards,
        "warmup_terminated": warmup_terminated,
        "warmup_truncated": warmup_truncated,
        "warmup_participant_counts": warmup_participant_counts,
        "warmup_static_object_counts": warmup_static_counts,
        "initial_state": trace.initial_state,
        "planning_anchors": np.stack(trace.planning_anchors),
        "initial_noise": np.concatenate(trace.noises, axis=0),
        "predictions_local": np.concatenate(trace.predictions_local, axis=0),
        "observation_ego_current_state": np.stack(trace.observation_ego_current_state),
        "observation_neighbor_agents_past": np.stack(trace.observation_neighbor_agents_past),
        "observation_static_objects": np.stack(trace.observation_static_objects),
        "observation_lanes": np.stack(trace.observation_lanes),
        "observation_lanes_speed_limit": np.stack(trace.observation_lanes_speed_limit),
        "observation_lanes_has_speed_limit": np.stack(trace.observation_lanes_has_speed_limit),
        "observation_route_lanes": np.stack(trace.observation_route_lanes),
        "ego_predictions_world": np.stack(trace.ego_world),
        "executed_states": np.concatenate(trace.substep_states, axis=0),
        "executed_rewards": np.concatenate(trace.substep_rewards, axis=0),
        "executed_terminated": np.concatenate(trace.substep_terminated, axis=0),
        "executed_truncated": np.concatenate(trace.substep_truncated, axis=0),
        "executed_plan_indices": np.concatenate(trace.substep_plan_indices, axis=0),
        "trajectory_target_centers": np.concatenate(trace.target_centers, axis=0),
        "trajectory_target_headings": np.concatenate(trace.target_headings, axis=0),
        "trajectory_position_errors_m": np.concatenate(trace.position_errors_m, axis=0),
        "trajectory_heading_errors_rad": np.concatenate(trace.heading_errors_rad, axis=0),
        "traffic_selected_ids": np.stack(trace.traffic_selected_ids),
        "traffic_participant_counts": np.asarray(trace.traffic_participant_counts, dtype=np.int64),
        "traffic_static_object_counts": np.asarray(
            trace.traffic_static_object_counts, dtype=np.int64
        ),
        "traffic_nearest_distance_m": np.asarray(
            trace.traffic_nearest_distance_m, dtype=np.float64
        ),
        "traffic_has_nearest": np.asarray(trace.traffic_has_nearest, dtype=np.bool_),
    }


def _episode_summary(
    spec: ScenarioSpec,
    report: CheckpointLoadReport,
    trace_arrays: dict[str, np.ndarray],
    final_info: dict[str, Any],
    terminated: bool,
    truncated: bool,
    total_reward: float,
    noise_seed: int,
    environment_map_audit: dict[str, object],
    evaluation_mode: str,
    traffic_density: float,
    route_length_m: float,
) -> dict[str, Any]:
    positions = np.vstack(
        (
            trace_arrays["initial_state"][None, :2],
            trace_arrays["executed_states"][:, :2],
        )
    )
    distance_m = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
    speeds = trace_arrays["executed_states"][:, 5]
    position_errors = trace_arrays["trajectory_position_errors_m"]
    heading_errors = trace_arrays["trajectory_heading_errors_rad"]
    warmup_states = trace_arrays["warmup_states"]
    warmup_displacement = (
        np.linalg.norm(
            warmup_states[:, :2] - trace_arrays["warmup_initial_state"][None, :2], axis=1
        )
        if warmup_states.size
        else np.empty(0, dtype=np.float64)
    )
    traffic_counts = trace_arrays["traffic_participant_counts"]
    traffic_has_nearest = trace_arrays["traffic_has_nearest"]
    nearest_distances = trace_arrays["traffic_nearest_distance_m"][traffic_has_nearest]
    return {
        "scenario": asdict(spec),
        "evaluation_mode": evaluation_mode,
        "traffic_density": traffic_density,
        "route_length_m": route_length_m,
        "noise_seed": noise_seed,
        "checkpoint": asdict(report),
        "plan_cycles": int(trace_arrays["initial_noise"].shape[0]),
        "simulator_steps": int(trace_arrays["executed_states"].shape[0]),
        "simulated_seconds": float(trace_arrays["executed_states"].shape[0] * 0.1),
        "environment_steps_including_warmup": int(
            warmup_states.shape[0] + trace_arrays["executed_states"].shape[0]
        ),
        "total_reward": total_reward,
        "distance_m": distance_m,
        "speed_mps": {
            "minimum": float(speeds.min()),
            "mean": float(speeds.mean()),
            "maximum": float(speeds.max()),
        },
        "route_completion": float(final_info["route_completion"]),
        "arrive_dest": bool(final_info["arrive_dest"]),
        "out_of_road": bool(final_info["out_of_road"]),
        "crash_vehicle": bool(final_info["crash_vehicle"]),
        "crash_object": bool(final_info["crash_object"]),
        "crash_building": bool(final_info["crash_building"]),
        "crash_human": bool(final_info["crash_human"]),
        "terminated": terminated,
        "truncated": truncated,
        "terminal_reason": _terminal_reason(final_info, terminated, truncated),
        "map_input_audit": _map_input_audit(trace_arrays, environment_map_audit),
        "history_warmup": {
            "simulator_steps": int(warmup_states.shape[0]),
            "simulated_seconds": float(warmup_states.shape[0] * 0.1),
            "ego_displacement_m_maximum": float(warmup_displacement.max())
            if warmup_displacement.size
            else 0.0,
            "participant_count_minimum": int(trace_arrays["warmup_participant_counts"].min())
            if trace_arrays["warmup_participant_counts"].size
            else 0,
            "participant_count_maximum": int(trace_arrays["warmup_participant_counts"].max())
            if trace_arrays["warmup_participant_counts"].size
            else 0,
        },
        "traffic_observation": {
            "planning_frames": int(traffic_counts.size),
            "frames_with_participants": int(np.count_nonzero(traffic_counts)),
            "frames_with_participants_fraction": float(np.mean(traffic_counts > 0)),
            "participant_count_minimum": int(traffic_counts.min()),
            "participant_count_maximum": int(traffic_counts.max()),
            "nearest_participant_distance_m_minimum": float(nearest_distances.min())
            if nearest_distances.size
            else None,
        },
        "trajectory_execution_error": {
            "position_m": _error_summary(position_errors),
            "heading_rad": _error_summary(heading_errors),
        },
    }


def _map_input_audit(
    trace_arrays: dict[str, np.ndarray], environment_map_audit: dict[str, object]
) -> dict[str, object]:
    speed_limits = trace_arrays["observation_lanes_speed_limit"]
    has_speed_limit = trace_arrays["observation_lanes_has_speed_limit"]
    if speed_limits.shape != has_speed_limit.shape:
        raise RuntimeError("trace lane speed limits and validity mask have incompatible shapes")
    valid_counts = has_speed_limit.sum(axis=(1, 2), dtype=np.int64)
    valid_speed_limits = speed_limits[has_speed_limit]
    result = dict(environment_map_audit)
    result.update(
        {
            "valid_lane_count_min": int(valid_counts.min()),
            "valid_lane_count_max": int(valid_counts.max()),
            "speed_limit_valid_count_min": int(valid_counts.min()),
            "speed_limit_valid_count_max": int(valid_counts.max()),
            "speed_limit_mps_min": None,
            "speed_limit_mps_max": None,
            "speed_limit_mps_unique_values": [],
        }
    )
    if valid_speed_limits.size:
        result["speed_limit_mps_min"] = float(valid_speed_limits.min())
        result["speed_limit_mps_max"] = float(valid_speed_limits.max())
        result["speed_limit_mps_unique_values"] = [
            float(value) for value in np.unique(valid_speed_limits)
        ]
    return result


def _error_summary(errors: np.ndarray) -> dict[str, float]:
    if errors.ndim != 1 or not errors.size or not np.isfinite(errors).all():
        raise RuntimeError(
            "trajectory execution errors must be a non-empty finite one-dimensional array"
        )
    return {
        "maximum": float(errors.max()),
        "mean": float(errors.mean()),
        "final": float(errors[-1]),
    }


def _write_episode_artifacts(
    output_dir: Path,
    trace: EpisodeTrace,
    frames: list[np.ndarray],
    summary: dict[str, Any],
    video_config: DictConfig,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output_dir / "trace.npz", **_stack_trace(trace))
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if video_config.enabled:
        if not frames:
            raise RuntimeError("video output was enabled but no frames were rendered")
        duration_ms = round(1000 / video_config.fps)
        generate_gif(frames, str(output_dir / "closed_loop.gif"), duration=duration_ms)


def _git_output(repository_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout


def _write_runtime_metadata(output_dir: Path) -> None:
    repository_root = Path(to_absolute_path("."))
    uv_lock = repository_root / "uv.lock"
    if not uv_lock.is_file():
        raise FileNotFoundError(f"required lock file does not exist: {uv_lock}")
    import hashlib

    metadata = {
        "git_head": _git_output(repository_root, "rev-parse", "HEAD").strip(),
        "git_status_short": _git_output(repository_root, "status", "--short").splitlines(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "metadrive": version("metadrive-simulator"),
        "uv_lock_sha256": hashlib.sha256(uv_lock.read_bytes()).hexdigest(),
    }
    (output_dir / "runtime_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "tracked_diff.patch").write_text(
        _git_output(repository_root, "diff", "--binary", "--no-ext-diff"),
        encoding="utf-8",
    )


def _run_scenario(
    spec: ScenarioSpec,
    planner: PretrainedDiffusionPlanner,
    report: CheckpointLoadReport,
    config: DictConfig,
    output_root: Path,
    device: torch.device,
) -> dict[str, Any]:
    raw_env_config = OmegaConf.to_container(config.env, resolve=True)
    if not isinstance(raw_env_config, dict):
        raise TypeError("env configuration must resolve to a dictionary")
    env_config = dict(raw_env_config)
    env_config["map"] = spec.map_sequence
    env = TrajectoryMetaDriveEnv(env_config)
    mode = str(config.evaluation.mode)
    if mode == "traffic":
        adapter: MetaDriveObservationAdapter | NoTrafficMetaDriveObservationAdapter
        adapter = MetaDriveObservationAdapter(planner.config, float(config.map_query_radius_m))
    else:
        adapter = NoTrafficMetaDriveObservationAdapter(
            planner.config, float(config.map_query_radius_m)
        )
    generator = torch.Generator(device=device).manual_seed(config.model.seed)
    frames: list[np.ndarray] = []
    try:
        env.reset(seed=spec.seed)
        route_length_m = _route_length_m(env)
        if mode == "traffic" and not 2_000.0 <= route_length_m <= 5_000.0:
            raise RuntimeError(
                f"traffic evaluation route length {route_length_m} m is outside [2000, 5000]"
            )
        trace = _new_episode_trace(env)
        if isinstance(adapter, MetaDriveObservationAdapter):
            adapter.reset(env.initial_traffic_frame)
            _run_traffic_warmup(
                env,
                adapter,
                trace,
                int(config.evaluation.history_warmup_steps),
            )
        terminated = False
        truncated = False
        total_reward = 0.0
        final_info: dict[str, Any] | None = None
        plan_index = 0
        while not terminated and not truncated:
            observation = adapter.build(env, device)
            traffic_audit = (
                adapter.last_audit if isinstance(adapter, MetaDriveObservationAdapter) else None
            )
            noise = _noise_for_planner(planner, generator, device)
            prediction = planner.predict(observation, noise)
            ego_trajectory = prediction[0, 0].detach().cpu().numpy().astype(np.float32)
            anchor = _initial_vehicle_state(env)
            _, reward, terminated, truncated, info = env.step(ego_trajectory)
            if isinstance(adapter, MetaDriveObservationAdapter):
                adapter.append_frames(_traffic_frames(info))
            total_reward += float(reward)
            _append_cycle(
                trace,
                anchor,
                observation,
                noise,
                prediction,
                info,
                plan_index,
                traffic_audit,
            )
            if config.video.enabled:
                frames.append(
                    _render_cycle_frame(
                        env,
                        info,
                        anchor[:2],
                        config.video,
                        plan_index,
                    )
                )
            final_info = info
            plan_index += 1
        if final_info is None:
            raise RuntimeError("closed-loop episode ended without a simulator result")
        trace_arrays = _stack_trace(trace)
        if mode == "traffic" and not np.any(trace_arrays["traffic_participant_counts"] > 0):
            raise RuntimeError("traffic evaluation never observed a participant within radius")
        summary = _episode_summary(
            spec,
            report,
            trace_arrays,
            final_info,
            terminated,
            truncated,
            total_reward,
            config.model.seed,
            env.programmatic_lane_speed_limit_audit,
            mode,
            float(config.env.traffic_density),
            route_length_m,
        )
        _write_episode_artifacts(
            output_root / spec.name,
            trace,
            frames,
            summary,
            config.video,
        )
        return summary
    finally:
        env.close()


def run_evaluation(config: DictConfig, output_dir: Path) -> list[dict[str, Any]]:
    """Run all configured no-traffic scenarios and write reproducible artifacts."""

    _validate_evaluation_config(config)
    scenarios = _parse_scenarios(config)
    device = torch.device(config.model.device)
    args_path = Path(to_absolute_path(config.model.args_path))
    checkpoint_path = Path(to_absolute_path(config.model.checkpoint_path))
    planner, report = load_official_diffusion_planner(
        args_path,
        checkpoint_path,
        config.model.expected_args_sha256,
        config.model.expected_checkpoint_sha256,
        device,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, output_dir / "resolved_config.yaml", resolve=True)
    _write_runtime_metadata(output_dir)
    summaries = [
        _run_scenario(spec, planner, report, config, output_dir, device) for spec in scenarios
    ]
    summary_payload = {
        "config": OmegaConf.to_container(config, resolve=True),
        "episodes": summaries,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summaries


@hydra.main(
    version_base="1.3",
    config_path="../../configs",
    config_name="evaluation/no_traffic",
)
def main(config: DictConfig) -> None:
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    summaries = run_evaluation(config, output_dir)
    print(json.dumps(summaries, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
