"""MetaDrive environment-cycle benchmark without planner inference."""

from __future__ import annotations

import statistics
import time
from typing import TypedDict

import numpy as np

from eco_planner.envs import (
    MetaDriveObservationAdapter,
    NoTrafficMetaDriveObservationAdapter,
    TrajectoryMetaDriveEnv,
)
from eco_planner.models import OfficialDiffusionPlannerConfig

_LONG_MIXED_MAP = "SCSCSCSCSCSCSCSCSCSCSCSCSCSCSCSCSCSCSCSC"
_WARMUP_HISTORY_STEPS = 20
_TIMING_WARMUP_CYCLES = 10
_MEASURED_CYCLES = 100
_REPEATS = 5


class BenchmarkMeasurement(TypedDict):
    cycle_ms: list[float]
    median_cycle_ms: float
    minimum_cycle_ms: float
    maximum_cycle_ms: float


class BenchmarkReport(TypedDict):
    scenario: dict[str, object]
    traffic: BenchmarkMeasurement
    no_traffic: BenchmarkMeasurement


class BenchmarkAcceptanceError(ValueError):
    """The measured benchmark did not meet a caller-provided threshold."""


def benchmark_environment(
    model_config: OfficialDiffusionPlannerConfig,
    *,
    traffic_baseline_ms: float | None = None,
    no_traffic_baseline_ms: float | None = None,
) -> BenchmarkReport:
    """Measure traffic and no-traffic environment cycles and enforce optional baselines."""

    result: BenchmarkReport = {
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
    _validate_baselines(result, traffic_baseline_ms, no_traffic_baseline_ms)
    return result


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
                adapter.append_frames(info["trajectory_execution"].traffic_frames)
        else:
            adapter = NoTrafficMetaDriveObservationAdapter(model_config, 100.0)
            adapter.reset(env)

        for _ in range(_TIMING_WARMUP_CYCLES):
            adapter.build(env)
            _, _, terminated, truncated, info = env.step(trajectory)
            if terminated or truncated:
                raise RuntimeError("timing warmup ended unexpectedly")
            if traffic:
                adapter.append_frames(info["trajectory_execution"].traffic_frames)

        start = time.perf_counter()
        for _ in range(_MEASURED_CYCLES):
            adapter.build(env)
            _, _, terminated, truncated, info = env.step(trajectory)
            if terminated or truncated:
                raise RuntimeError("measured cycle ended unexpectedly")
            if traffic:
                adapter.append_frames(info["trajectory_execution"].traffic_frames)
        elapsed = time.perf_counter() - start
        return elapsed * 1_000.0 / _MEASURED_CYCLES
    finally:
        env.close()


def _measure(
    model_config: OfficialDiffusionPlannerConfig, *, traffic: bool
) -> BenchmarkMeasurement:
    samples = [_run_once(model_config, traffic=traffic) for _ in range(_REPEATS)]
    return {
        "cycle_ms": samples,
        "median_cycle_ms": statistics.median(samples),
        "minimum_cycle_ms": min(samples),
        "maximum_cycle_ms": max(samples),
    }


def _validate_baselines(
    result: BenchmarkReport,
    traffic_baseline_ms: float | None,
    no_traffic_baseline_ms: float | None,
) -> None:
    if traffic_baseline_ms is not None:
        traffic_limit = 0.8 * traffic_baseline_ms
        if result["traffic"]["median_cycle_ms"] > traffic_limit:
            raise BenchmarkAcceptanceError(
                "traffic median did not improve by 20%: "
                f"{result['traffic']['median_cycle_ms']:.3f} ms > {traffic_limit:.3f} ms"
            )
    if no_traffic_baseline_ms is not None:
        no_traffic_limit = 1.05 * no_traffic_baseline_ms
        if result["no_traffic"]["median_cycle_ms"] > no_traffic_limit:
            raise BenchmarkAcceptanceError(
                "no-traffic median regressed by more than 5%: "
                f"{result['no_traffic']['median_cycle_ms']:.3f} ms > {no_traffic_limit:.3f} ms"
            )
