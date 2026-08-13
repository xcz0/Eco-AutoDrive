"""Measure the planner-facing MetaDrive environment cycle without model inference."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from eco_planner.envs import (
    MetaDriveObservationAdapter,
    NoTrafficMetaDriveObservationAdapter,
    TrajectoryMetaDriveEnv,
)
from eco_planner.models.config import OfficialDiffusionPlannerConfig

_LONG_MIXED_MAP = "SCSCSCSCSCSCSCSCSCSCSCSCSCSCSCSCSCSCSCSC"
_WARMUP_HISTORY_STEPS = 20
_TIMING_WARMUP_CYCLES = 10
_MEASURED_CYCLES = 100
_REPEATS = 5


def _stationary_trajectory() -> np.ndarray:
    trajectory = np.zeros((80, 4), dtype=np.float32)
    trajectory[:, 2] = 1.0
    return trajectory


def _environment_config(traffic_density: float) -> dict[str, object]:
    return {
        "use_render": False,
        "image_observation": False,
        "map": _LONG_MIXED_MAP,
        "num_scenarios": 1,
        "traffic_density": traffic_density,
        "traffic_mode": "trigger",
        "random_traffic": False,
        "random_spawn_lane_index": False,
        "accident_prob": 0.0,
        "physics_world_step_size": 0.02,
        "decision_repeat": 5,
        "trajectory_horizon": 80,
        "trajectory_execution_steps": 5,
        "programmatic_lane_speed_limit_kmh": 50.0,
        "horizon": 700,
    }


def _run_once(model_config: OfficialDiffusionPlannerConfig, *, traffic: bool) -> float:
    env = TrajectoryMetaDriveEnv(_environment_config(0.05 if traffic else 0.0))
    trajectory = _stationary_trajectory()
    try:
        env.reset(seed=0)
        if traffic:
            adapter = MetaDriveObservationAdapter(model_config, 100.0)
            adapter.reset(env.initial_traffic_frame, env=env)
            for _ in range(_WARMUP_HISTORY_STEPS // 5):
                _, _, terminated, truncated, info = env.step(trajectory)
                if terminated or truncated:
                    raise RuntimeError("traffic history warmup ended unexpectedly")
                adapter.append_frames(info["traffic_substep_frames"])
        else:
            adapter = NoTrafficMetaDriveObservationAdapter(model_config, 100.0)
            adapter.reset(env)

        for _ in range(_TIMING_WARMUP_CYCLES):
            adapter.build(env)
            _, _, terminated, truncated, info = env.step(trajectory)
            if terminated or truncated:
                raise RuntimeError("timing warmup ended unexpectedly")
            if traffic:
                adapter.append_frames(info["traffic_substep_frames"])

        start = time.perf_counter()
        for _ in range(_MEASURED_CYCLES):
            adapter.build(env)
            _, _, terminated, truncated, info = env.step(trajectory)
            if terminated or truncated:
                raise RuntimeError("measured cycle ended unexpectedly")
            if traffic:
                adapter.append_frames(info["traffic_substep_frames"])
        elapsed = time.perf_counter() - start
        return elapsed * 1_000.0 / _MEASURED_CYCLES
    finally:
        env.close()


def _measure(model_config: OfficialDiffusionPlannerConfig, *, traffic: bool) -> dict[str, object]:
    samples = [_run_once(model_config, traffic=traffic) for _ in range(_REPEATS)]
    return {
        "cycle_ms": samples,
        "median_cycle_ms": statistics.median(samples),
        "minimum_cycle_ms": min(samples),
        "maximum_cycle_ms": max(samples),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--args-path",
        type=Path,
        default=Path("checkpoints/DP-Origin/args.json"),
    )
    parser.add_argument("--traffic-baseline-ms", type=float)
    parser.add_argument("--no-traffic-baseline-ms", type=float)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    model_config = OfficialDiffusionPlannerConfig.from_json(args.args_path)
    result = {
        "scenario": {
            "map": _LONG_MIXED_MAP,
            "seed": 0,
            "traffic_density": 0.05,
            "history_warmup_steps": _WARMUP_HISTORY_STEPS,
            "timing_warmup_cycles": _TIMING_WARMUP_CYCLES,
            "measured_cycles": _MEASURED_CYCLES,
            "repeats": _REPEATS,
        },
        "traffic": _measure(model_config, traffic=True),
        "no_traffic": _measure(model_config, traffic=False),
    }
    print(json.dumps(result, indent=2))

    if args.traffic_baseline_ms is not None:
        traffic_limit = 0.8 * args.traffic_baseline_ms
        if result["traffic"]["median_cycle_ms"] > traffic_limit:
            raise SystemExit(
                "traffic median did not improve by 20%: "
                f"{result['traffic']['median_cycle_ms']:.3f} ms > {traffic_limit:.3f} ms"
            )
    if args.no_traffic_baseline_ms is not None:
        no_traffic_limit = 1.05 * args.no_traffic_baseline_ms
        if result["no_traffic"]["median_cycle_ms"] > no_traffic_limit:
            raise SystemExit(
                "no-traffic median regressed by more than 5%: "
                f"{result['no_traffic']['median_cycle_ms']:.3f} ms > {no_traffic_limit:.3f} ms"
            )


if __name__ == "__main__":
    main()
