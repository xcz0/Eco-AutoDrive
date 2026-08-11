"""Fixed-seed Diffusion Planner closed-loop evaluation orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from eco_planner.envs import (
    MetaDriveObservationAdapter,
    NoTrafficMetaDriveObservationAdapter,
    TrafficFrame,
    TrajectoryMetaDriveEnv,
)
from eco_planner.evaluation.artifacts import (
    build_episode_summary,
    write_episode_artifacts,
    write_json,
    write_runtime_metadata,
)
from eco_planner.evaluation.rendering import render_cycle_frame
from eco_planner.evaluation.runtime import (
    FabricInferenceRuntime,
    create_fabric_inference_runtime,
    resolve_runtime_settings,
)
from eco_planner.evaluation.trace import EpisodeTraceRecorder
from eco_planner.models.sampling_config import parse_sampler_config


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    map_sequence: str
    seed: int


def run_evaluation(config: DictConfig, output_dir: Path) -> dict[str, Any]:
    """Run all configured scenarios and write reproducible artifacts."""

    _validate_evaluation_config(config)
    scenarios = _parse_scenarios(config)
    args_path = Path(to_absolute_path(config.model.args_path))
    checkpoint_path = Path(to_absolute_path(config.model.checkpoint_path))
    runtime = create_fabric_inference_runtime(
        config.runtime,
        config.sampler,
        args_path,
        checkpoint_path,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, output_dir / "resolved_config.yaml", resolve=True)
    write_runtime_metadata(output_dir, runtime.report, runtime.sampler_report)
    summaries = [_run_scenario(spec, runtime, config, output_dir) for spec in scenarios]
    summary = {
        "runtime": asdict(runtime.report),
        "checkpoint": asdict(runtime.checkpoint_report),
        "sampler": asdict(runtime.sampler_report),
        "episodes": summaries,
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def _run_scenario(
    spec: ScenarioSpec,
    runtime: FabricInferenceRuntime,
    config: DictConfig,
    output_root: Path,
) -> dict[str, Any]:
    raw_env_config = OmegaConf.to_container(config.env, resolve=True)
    if not isinstance(raw_env_config, dict):
        raise TypeError("env configuration must resolve to a dictionary")
    env_config = dict(raw_env_config)
    env_config["map"] = spec.map_sequence
    env = TrajectoryMetaDriveEnv(env_config)
    mode = str(config.evaluation.mode)
    traffic_adapter = (
        MetaDriveObservationAdapter(runtime.planner_config, float(config.map_query_radius_m))
        if mode == "traffic"
        else None
    )
    no_traffic_adapter = (
        NoTrafficMetaDriveObservationAdapter(
            runtime.planner_config, float(config.map_query_radius_m)
        )
        if mode == "no_traffic"
        else None
    )
    generator = runtime.new_noise_generator()
    frames: list[np.ndarray] = []
    try:
        env.reset(seed=spec.seed)
        route_length_m = _route_length_m(env)
        if mode == "traffic" and not 2_000.0 <= route_length_m <= 5_000.0:
            raise RuntimeError(
                f"traffic evaluation route length {route_length_m} m is outside [2000, 5000]"
            )
        trace = EpisodeTraceRecorder.from_initial_state(_initial_vehicle_state(env))
        if traffic_adapter is not None:
            traffic_adapter.reset(env.initial_traffic_frame)
            _run_traffic_warmup(
                env,
                traffic_adapter,
                trace,
                int(config.evaluation.history_warmup_steps),
            )

        terminated = False
        truncated = False
        total_reward = 0.0
        final_info: dict[str, Any] | None = None
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
            observation, noise, prediction = runtime.infer(raw_observation, generator)
            ego_trajectory = prediction[0, 0].detach().cpu().numpy().astype(np.float32)
            anchor = _initial_vehicle_state(env)
            _, reward, terminated, truncated, info = env.step(ego_trajectory)
            if traffic_adapter is not None:
                traffic_adapter.append_frames(_traffic_frames(info))
            total_reward += float(reward)
            trace.append_cycle(
                anchor,
                observation,
                noise,
                prediction,
                info,
                plan_index,
                traffic_audit,
            )
            if config.video.enabled:
                frames.append(render_cycle_frame(env, info, anchor[:2], config.video, plan_index))
            final_info = info
            plan_index += 1
        if final_info is None:
            raise RuntimeError("closed-loop episode ended without a simulator result")
        trace_arrays = trace.finalize()
        if mode == "traffic" and not np.any(trace_arrays["traffic_participant_counts"] > 0):
            raise RuntimeError("traffic evaluation never observed a participant within radius")
        summary = build_episode_summary(
            asdict(spec),
            trace_arrays,
            final_info,
            terminated,
            truncated,
            total_reward,
            runtime.report.seed,
            env.programmatic_lane_speed_limit_audit,
            mode,
            float(config.env.traffic_density),
            route_length_m,
            asdict(runtime.sampler_report),
        )
        write_episode_artifacts(
            output_root / spec.name, trace_arrays, frames, summary, config.video
        )
        return summary
    finally:
        env.close()


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
    resolve_runtime_settings(config.runtime)
    if "sampler" not in config:
        raise ValueError("evaluation configuration must select a sampler")
    parse_sampler_config(config.sampler)
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


def _run_traffic_warmup(
    env: TrajectoryMetaDriveEnv,
    adapter: MetaDriveObservationAdapter,
    trace: EpisodeTraceRecorder,
    warmup_steps: int,
) -> None:
    if warmup_steps % 5 != 0:
        raise ValueError("history warmup steps must be divisible by five")
    initial_position = trace.warmup_initial_state[:2].copy()
    for _ in range(warmup_steps // 5):
        _, _, terminated, truncated, info = env.step(_stationary_trajectory())
        frames = _traffic_frames(info)
        adapter.append_frames(frames)
        trace.append_warmup(
            info,
            np.asarray([len(frame.participants) for frame in frames], dtype=np.int64),
            np.asarray([len(frame.static_objects) for frame in frames], dtype=np.int64),
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
        if not np.isfinite(lane_length) or float(lane_length) <= 0.0:
            raise RuntimeError(f"route edge {(start, end)!r} has an invalid length")
        edge_lengths.append(float(lane_length))
    return float(sum(edge_lengths))
