"""Validation and statistics for the fixed MetaDrive traffic matrix."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from eco_planner.evaluation.artifacts.io import (
    load_episode_summary,
    load_job_summary,
    load_runtime_metadata,
    load_trace_artifact,
)
from eco_planner.evaluation.artifacts.models import EpisodeSummary
from eco_planner.evaluation.artifacts.trace_schema import validate_trace_arrays

BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 0
POSITION_ERROR_LIMIT_M = 1e-3
HEADING_ERROR_LIMIT_RAD = 1e-4


@dataclass(frozen=True)
class _MatrixSpec:
    seeds: tuple[int, ...]
    densities: tuple[float, ...]
    scenario_names: tuple[str, ...]
    warmup_steps: int
    video_enabled: bool

    @property
    def expected_jobs(self) -> set[tuple[int, float]]:
        return {(seed, density) for seed in self.seeds for density in self.densities}


def summarize_matrix(matrix_root: Path, *, partial: bool = False) -> dict[str, Any]:
    """Validate, summarize, and persist a complete or explicitly partial matrix."""

    matrix_root = matrix_root.resolve()
    report_path = matrix_root / ("partial_matrix_report.json" if partial else "matrix_report.json")
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {report_path}")
    report = build_matrix_report(matrix_root, partial=partial)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def build_matrix_report(matrix_root: Path, *, partial: bool = False) -> dict[str, Any]:
    """Validate and build a report without writing or overwriting an artifact."""

    matrix_root = matrix_root.resolve()
    if not matrix_root.is_dir():
        raise NotADirectoryError(f"matrix root does not exist: {matrix_root}")
    jobs = sorted(
        (path for path in matrix_root.iterdir() if path.is_dir() and path.name.isdigit()),
        key=lambda path: int(path.name),
    )
    if partial:
        jobs = [job for job in jobs if (job / "summary.json").is_file()]
        if not jobs:
            raise ValueError("partial matrix has no complete Hydra jobs")

    observed_jobs: set[tuple[int, float]] = set()
    episodes: list[EpisodeSummary] = []
    matrix_spec: _MatrixSpec | None = None
    for job_dir in jobs:
        _require_nonempty(job_dir / "resolved_config.yaml")
        job_spec = _read_matrix_spec(job_dir / "resolved_config.yaml")
        if matrix_spec is None:
            matrix_spec = job_spec
        elif matrix_spec != job_spec:
            raise ValueError(f"job {job_dir} resolved matrix specification disagrees")
        _require_nonempty(job_dir / ".hydra" / "overrides.yaml")
        metadata = load_runtime_metadata(job_dir / "runtime_metadata.json")
        _require_file(job_dir / "tracked_diff.patch")
        job_summary = load_job_summary(job_dir / "summary.json")
        runtime = job_summary.runtime
        metadata_runtime = metadata.inference_runtime
        if metadata_runtime != runtime:
            raise ValueError(f"job {job_dir} runtime metadata disagrees with summary")
        seed = runtime.seed
        job_episodes = job_summary.episodes
        density = job_episodes[0].traffic_density
        job_key = (seed, density)
        if job_key in observed_jobs:
            raise ValueError(f"duplicate matrix job: seed={seed}, density={density}")
        if job_key not in matrix_spec.expected_jobs:
            raise ValueError(f"unexpected matrix job: seed={seed}, density={density}")
        observed_jobs.add(job_key)
        scenario_names = {episode.scenario.name for episode in job_episodes}
        expected_scenarios = set(matrix_spec.scenario_names)
        if scenario_names != expected_scenarios:
            raise ValueError(f"job {job_dir} has unexpected scenarios: {scenario_names}")
        for episode in job_episodes:
            _validate_episode_summary(episode, job_dir, seed, density)
            scenario_name = episode.scenario.name
            episode_dir = job_dir / scenario_name
            persisted_episode = load_episode_summary(episode_dir / "summary.json")
            if persisted_episode != episode:
                raise ValueError(f"episode summary copy disagrees with job summary: {episode_dir}")
            if matrix_spec.video_enabled:
                _require_nonempty(episode_dir / "closed_loop.gif")
            _validate_trace(
                episode_dir / "trace.npz",
                episode,
                warmup_steps=matrix_spec.warmup_steps,
            )
            episodes.append(episode)

    if matrix_spec is None:
        raise ValueError("matrix has no Hydra jobs")
    expected_jobs = matrix_spec.expected_jobs
    if not partial and observed_jobs != expected_jobs:
        raise ValueError(
            f"matrix job grid mismatch: missing={sorted(expected_jobs - observed_jobs)}, "
            f"unexpected={sorted(observed_jobs - expected_jobs)}"
        )
    scenario_count = len(matrix_spec.scenario_names)
    expected_episode_count = scenario_count * len(observed_jobs)
    if len(episodes) != expected_episode_count:
        raise RuntimeError("matrix episode count does not match validated job count")
    return _build_report(
        matrix_root,
        partial,
        observed_jobs,
        expected_jobs,
        scenario_count,
        episodes,
    )


def _validate_trace(path: Path, episode: EpisodeSummary, *, warmup_steps: int) -> None:
    _require_nonempty(path)
    loaded = load_trace_artifact(path)
    arrays = loaded.arrays
    completed = episode.status == "completed"
    validate_trace_arrays(
        arrays,
        expected_plan_cycles=(episode.plan_cycles if completed else None),
        expected_simulator_steps=(episode.simulator_steps if completed else None),
        expected_warmup_steps=warmup_steps if completed else None,
        require_traffic=completed,
        expected_trace_status=episode.trace_status,
    )
    if not completed:
        return
    position_errors = arrays["trajectory_position_errors_m"]
    heading_errors = arrays["trajectory_heading_errors_rad"]
    if float(position_errors.max()) >= POSITION_ERROR_LIMIT_M:
        raise ValueError(f"trace {path} exceeds the trajectory position error limit")
    if float(heading_errors.max()) >= HEADING_ERROR_LIMIT_RAD:
        raise ValueError(f"trace {path} exceeds the trajectory heading error limit")


def _validate_episode_summary(
    episode: EpisodeSummary, job_dir: Path, seed: int, density: float
) -> None:
    if episode.noise_seed != seed:
        raise ValueError(f"job {job_dir} does not use the configured noise seed")
    if episode.scenario.seed != seed:
        raise ValueError(f"job {job_dir} does not use paired map/noise seeds")
    if episode.traffic_density != density:
        raise ValueError(f"job {job_dir} summary density disagrees with config")
    if episode.status == "failed":
        return
    route_length_m = episode.route_length_m
    if not 2_000.0 <= route_length_m <= 5_000.0:
        raise ValueError(f"job {job_dir} route length is outside [2000, 5000] m")


def _build_report(
    matrix_root: Path,
    partial: bool,
    observed_jobs: set[tuple[int, float]],
    expected_jobs: set[tuple[int, float]],
    scenario_count: int,
    episodes: list[EpisodeSummary],
) -> dict[str, Any]:
    grouped: defaultdict[tuple[str, float], list[EpisodeSummary]] = defaultdict(list)
    for episode in episodes:
        if episode.status != "completed":
            continue
        grouped[(episode.scenario.name, episode.traffic_density)].append(episode)
    statistics: dict[str, Any] = {}
    for (scenario, density), group in sorted(grouped.items()):
        label = f"{scenario}/density_{density:.2f}"
        metrics = {
            "simulated_seconds": np.asarray(
                [episode.simulated_seconds for episode in group], dtype=np.float64
            ),
            "distance_m": np.asarray([episode.distance_m for episode in group], dtype=np.float64),
            "route_completion": np.asarray(
                [episode.route_completion for episode in group], dtype=np.float64
            ),
            "mean_speed_mps": np.asarray(
                [episode.speed_mps.mean for episode in group], dtype=np.float64
            ),
            "total_reward": np.asarray(
                [episode.total_reward for episode in group], dtype=np.float64
            ),
        }
        statistics[label] = {
            "episode_count": len(group),
            "metrics": {name: _bootstrap(values) for name, values in metrics.items()},
            "arrive_rate": float(np.mean([episode.arrive_dest for episode in group])),
            "collision_rate": float(
                np.mean(
                    [
                        episode.crash_vehicle
                        or episode.crash_object
                        or episode.crash_building
                        or episode.crash_human
                        for episode in group
                    ]
                )
            ),
            "out_of_road_rate": float(np.mean([episode.out_of_road for episode in group])),
        }
    episode_rows = [
        {
            "scenario": episode.scenario.name,
            "seed": episode.noise_seed,
            "traffic_density": episode.traffic_density,
            "terminal_reason": (episode.terminal_reason if episode.status == "completed" else None),
            "status": episode.status,
            "termination": episode.termination.model_dump(mode="json"),
            "simulated_seconds": (
                episode.simulated_seconds if episode.status == "completed" else None
            ),
            "distance_m": episode.distance_m if episode.status == "completed" else None,
            "route_completion": (
                episode.route_completion if episode.status == "completed" else None
            ),
            "mean_speed_mps": episode.speed_mps.mean if episode.status == "completed" else None,
            "total_reward": episode.total_reward if episode.status == "completed" else None,
        }
        for episode in sorted(
            episodes,
            key=lambda item: (
                item.scenario.name,
                item.traffic_density,
                item.noise_seed,
            ),
        )
    ]
    return {
        "matrix_root": str(matrix_root),
        "matrix_complete": not partial,
        "matrix_successful": all(episode.status == "completed" for episode in episodes),
        "observed_job_grid": [
            {"seed": seed, "traffic_density": density} for seed, density in sorted(observed_jobs)
        ],
        "expected_job_grid": [
            {"seed": seed, "traffic_density": density} for seed, density in sorted(expected_jobs)
        ],
        "expected_episode_count": scenario_count * len(expected_jobs),
        "validated_episode_count": len(episodes),
        "status_counts": {
            status: sum(episode.status == status for episode in episodes)
            for status in ("completed", "failed")
        },
        "termination_type_counts": {
            kind: sum(episode.termination.type == kind for episode in episodes)
            for kind in sorted({episode.termination.type for episode in episodes})
        },
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "interface_limits": {
            "trajectory_position_error_m": POSITION_ERROR_LIMIT_M,
            "trajectory_heading_error_rad": HEADING_ERROR_LIMIT_RAD,
        },
        "episodes": episode_rows,
        "statistics": statistics,
    }


def _bootstrap(values: np.ndarray) -> dict[str, float | list[float]]:
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("bootstrap values must be a non-empty finite vector")
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    indices = generator.integers(0, values.size, size=(BOOTSTRAP_SAMPLES, values.size))
    means = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "mean_bootstrap_95_percentile_interval": [
            float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)),
        ],
    }


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"required artifact does not exist: {path}")


def _require_nonempty(path: Path) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"required non-empty artifact does not exist: {path}")


def _read_matrix_spec(path: Path) -> _MatrixSpec:
    config = OmegaConf.load(path)
    raw = OmegaConf.select(config, "evaluation.matrix")
    if raw is None:
        raise ValueError(f"evaluation.matrix is missing: {path}")
    seeds = tuple(raw.seeds)
    densities = tuple(float(value) for value in raw.traffic_densities)
    scenarios = OmegaConf.select(config, "scenarios")
    if not seeds or not all(type(seed) is int and seed >= 0 for seed in seeds):
        raise ValueError(f"matrix seeds must be non-empty non-negative integers: {path}")
    if not densities or not all(np.isfinite(value) and 0.0 < value <= 1.0 for value in densities):
        raise ValueError(f"matrix traffic densities must be in (0, 1]: {path}")
    if len(set(seeds)) != len(seeds) or len(set(densities)) != len(densities):
        raise ValueError(f"matrix grid values must be unique: {path}")
    if scenarios is None:
        raise ValueError(f"matrix scenarios are missing: {path}")
    names = tuple(str(value.name) for value in scenarios)
    warmup_steps = OmegaConf.select(config, "evaluation.history_warmup_steps")
    video_enabled = OmegaConf.select(config, "video.enabled")
    if type(warmup_steps) is not int or warmup_steps < 0:
        raise ValueError(f"matrix warmup steps are invalid: {path}")
    if type(video_enabled) is not bool:
        raise ValueError(f"matrix video.enabled must be boolean: {path}")
    return _MatrixSpec(seeds, densities, names, warmup_steps, video_enabled)
