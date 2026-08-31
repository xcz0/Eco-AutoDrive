"""Planner-facing MetaDrive environment-cycle benchmark without model inference."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from eco_planner.envs import (
    MetaDriveEnvSlot,
    PlannerObservationSpec,
)
from eco_planner.models import OfficialDiffusionPlannerConfig

from .config import (
    EnvironmentBenchmarkConfig,
    EnvironmentBenchmarkJobConfig,
    Measurement,
    benchmark_provenance,
    host_resource_provenance,
    measurement,
    parse_environment_job,
    write_benchmark_artifacts,
)


class BenchmarkAcceptanceError(ValueError):
    """The measured environment benchmark did not meet its configured threshold."""


def run(config: DictConfig) -> None:
    parsed = parse_environment_job(config)
    model_config = OfficialDiffusionPlannerConfig.from_json(
        Path(to_absolute_path(parsed.model.args_path))
    )
    report: dict[str, object] = {
        "provenance": {
            **benchmark_provenance(parsed.benchmark),
            **host_resource_provenance(),
        },
        "scenario": parsed.benchmark.model_dump(mode="json"),
        "traffic": _measure(parsed, model_config, traffic=True),
        "no_traffic": _measure(parsed, model_config, traffic=False),
    }
    _validate_baselines(report, parsed.benchmark)
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    write_benchmark_artifacts(output_dir, config, "environment.json", report)
    print(json.dumps(report, indent=2))


def _measure(
    job: EnvironmentBenchmarkJobConfig,
    model_config: OfficialDiffusionPlannerConfig,
    *,
    traffic: bool,
) -> Measurement:
    samples = [_run_once(job, model_config, traffic=traffic) for _ in range(job.benchmark.repeats)]
    return measurement(samples)


def _run_once(
    job: EnvironmentBenchmarkJobConfig,
    model_config: OfficialDiffusionPlannerConfig,
    *,
    traffic: bool,
) -> float:
    benchmark = job.benchmark
    env_config = {
        **job.env,
        "map": benchmark.map,
        "traffic_density": benchmark.traffic_density if traffic else 0.0,
    }
    env_slot = MetaDriveEnvSlot(
        env_config,
        mode="traffic" if traffic else "no_traffic",
        observation_spec=PlannerObservationSpec.from_planner_config(model_config),
        map_query_radius_m=benchmark.map_query_radius_m,
        history_warmup_steps=benchmark.history_warmup_steps if traffic else 0,
    )
    trajectory = _stationary_trajectory(model_config.future_len)
    try:
        env_slot.reset(map_name=benchmark.map, seed=benchmark.seed)
        tuple(env_slot.warmup())

        for _ in range(benchmark.timing_warmup_cycles):
            _run_cycle(env_slot, trajectory)

        start = perf_counter()
        for _ in range(benchmark.measured_cycles):
            _run_cycle(env_slot, trajectory)
        elapsed = perf_counter() - start
        return elapsed * 1_000.0 / benchmark.measured_cycles
    finally:
        env_slot.close()


def _run_cycle(
    env_slot: MetaDriveEnvSlot,
    trajectory: np.ndarray,
) -> None:
    env_slot.observe()
    step = env_slot.step(trajectory)
    if step.terminated or step.truncated:
        raise RuntimeError("environment benchmark ended before measurement completed")


def _stationary_trajectory(future_len: int) -> np.ndarray:
    trajectory = np.zeros((future_len, 4), dtype=np.float32)
    trajectory[:, 2] = 1.0
    return trajectory


def _validate_baselines(report: dict[str, object], benchmark: EnvironmentBenchmarkConfig) -> None:
    traffic = report["traffic"]
    no_traffic = report["no_traffic"]
    if not isinstance(traffic, dict) or not isinstance(no_traffic, dict):
        raise TypeError("environment benchmark report has invalid measurements")
    if benchmark.traffic_baseline_ms is not None:
        limit = benchmark.traffic_baseline_ms * (
            1.0 - benchmark.traffic_required_improvement_fraction
        )
        if traffic["median"] > limit:
            raise BenchmarkAcceptanceError(
                "traffic median did not meet required improvement: "
                f"{traffic['median']:.3f} ms > {limit:.3f} ms"
            )
    if benchmark.no_traffic_baseline_ms is not None:
        limit = benchmark.no_traffic_baseline_ms * (
            1.0 + benchmark.no_traffic_allowed_regression_fraction
        )
        if no_traffic["median"] > limit:
            raise BenchmarkAcceptanceError(
                "no-traffic median exceeded allowed regression: "
                f"{no_traffic['median']:.3f} ms > {limit:.3f} ms"
            )
