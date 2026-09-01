"""Offline evaluation artifact validation, statistics, reports, and comparison."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from eco_planner.artifacts import write_json
from eco_planner.evaluation.artifacts import (
    load_episode_summary,
    load_job_summary,
    load_runtime_metadata,
    load_trace_artifact,
)
from eco_planner.evaluation.models import (
    CompletedEpisodeSummary,
    EpisodeSummary,
    EvaluationWorkload,
    JobSummary,
)
from eco_planner.evaluation.validation import validate_episode_artifact, validate_matrix_episode

POSITION_ERROR_LIMIT_M = 1e-3
HEADING_ERROR_LIMIT_RAD = 1e-4
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 0


def summarize_matrix(matrix_root: Path, *, partial: bool = False) -> dict[str, Any]:
    """Validate, summarize, and persist a complete or explicitly partial matrix."""

    matrix_root = matrix_root.resolve()
    report_path = matrix_root / ("partial_matrix_report.json" if partial else "matrix_report.json")
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {report_path}")
    report = build_matrix_report(matrix_root, partial=partial)
    write_json(report_path, report)
    return report


def build_matrix_report(matrix_root: Path, *, partial: bool = False) -> dict[str, Any]:
    """Validate artifacts and build a report without writing or overwriting an artifact."""

    return _build_report(validate_matrix_artifacts(matrix_root, partial=partial))


def _build_report(validated: ValidatedMatrix) -> dict[str, Any]:
    episodes = validated.episodes
    episode_rows = []
    for episode in sorted(
        episodes,
        key=lambda item: (item.scenario.name, item.traffic_density, item.noise_seed),
    ):
        completed = isinstance(episode, CompletedEpisodeSummary)
        metrics = episode.metrics if completed else None
        episode_rows.append(
            {
                "scenario": episode.scenario.name,
                "seed": episode.noise_seed,
                "traffic_density": episode.traffic_density,
                "terminal_reason": episode.terminal_reason if completed else None,
                "status": episode.status,
                "termination": episode.termination.model_dump(mode="json"),
                "simulated_seconds": None if metrics is None else metrics.simulated_seconds,
                "distance_m": None if metrics is None else metrics.distance_m,
                "energy_total_ml": None if metrics is None else metrics.energy.total_ml,
                "energy_ml_per_km": None if metrics is None else metrics.energy.ml_per_km,
                "route_completion": None if metrics is None else metrics.route_completion,
                "mean_speed_mps": None if metrics is None else metrics.speed_mps.mean,
                "total_reward": None if metrics is None else metrics.total_reward,
            }
        )
    return {
        "matrix_root": str(validated.matrix_root),
        "matrix_complete": not validated.partial,
        "matrix_successful": all(episode.status == "completed" for episode in episodes),
        "observed_job_grid": [
            {"seed": seed, "traffic_density": density}
            for seed, density in sorted(validated.observed_jobs)
        ],
        "expected_job_grid": [
            {"seed": seed, "traffic_density": density}
            for seed, density in sorted(validated.expected_jobs)
        ],
        "expected_episode_count": validated.scenario_count * len(validated.expected_jobs),
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
        "statistics": build_matrix_statistics(episodes),
    }


@dataclass(frozen=True)
class MatrixSpec:
    workload: EvaluationWorkload

    @property
    def expected_jobs(self) -> set[tuple[int, float]]:
        if self.workload.matrix is None:
            raise ValueError("evaluation workload does not declare a matrix")
        return {
            (seed, density)
            for seed in self.workload.matrix.seeds
            for density in self.workload.matrix.traffic_densities
        }

    @property
    def scenario_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.workload.scenarios)


@dataclass(frozen=True)
class ValidatedMatrix:
    matrix_root: Path
    partial: bool
    observed_jobs: set[tuple[int, float]]
    expected_jobs: set[tuple[int, float]]
    scenario_count: int
    episodes: tuple[EpisodeSummary, ...]


def validate_matrix_artifacts(matrix_root: Path, *, partial: bool = False) -> ValidatedMatrix:
    """Validate all required matrix artifacts and return their typed episode data."""

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
    matrix_spec: MatrixSpec | None = None
    for job_dir in jobs:
        _require_nonempty(job_dir / "resolved_config.yaml")
        job_summary = load_job_summary(job_dir / "summary.json")
        job_spec = MatrixSpec(job_summary.workload)
        if job_spec.workload.matrix is None:
            raise ValueError(f"job {job_dir} does not declare an evaluation matrix")
        if matrix_spec is None:
            matrix_spec = job_spec
        elif matrix_spec != job_spec:
            raise ValueError(f"job {job_dir} resolved matrix specification disagrees")
        _require_nonempty(job_dir / ".hydra" / "overrides.yaml")
        metadata = load_runtime_metadata(job_dir / "runtime_metadata.json")
        _require_file(job_dir / "tracked_diff.patch")
        if metadata.inference_runtime != job_summary.runtime:
            raise ValueError(f"job {job_dir} runtime metadata disagrees with summary")
        seed = job_summary.runtime.seed
        job_episodes = job_summary.episodes
        density = job_episodes[0].traffic_density
        job_key = (seed, density)
        if job_key in observed_jobs:
            raise ValueError(f"duplicate matrix job: seed={seed}, density={density}")
        if job_key not in matrix_spec.expected_jobs:
            raise ValueError(f"unexpected matrix job: seed={seed}, density={density}")
        observed_jobs.add(job_key)
        scenario_names = {episode.scenario.name for episode in job_episodes}
        if scenario_names != set(matrix_spec.scenario_names):
            raise ValueError(f"job {job_dir} has unexpected scenarios: {scenario_names}")
        for episode in job_episodes:
            validate_matrix_episode(episode, job_dir, seed, density)
            episode_dir = job_dir / episode.scenario.name
            if load_episode_summary(episode_dir / "summary.json") != episode:
                raise ValueError(f"episode summary copy disagrees with job summary: {episode_dir}")
            if matrix_spec.workload.video_enabled:
                _require_nonempty(episode_dir / "closed_loop.gif")
            validate_episode_artifact(
                episode_dir / "trace.npz",
                episode,
                warmup_steps=matrix_spec.workload.history_warmup_steps,
                require_traffic=matrix_spec.workload.mode == "traffic",
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
    if len(episodes) != scenario_count * len(observed_jobs):
        raise RuntimeError("matrix episode count does not match validated job count")
    return ValidatedMatrix(
        matrix_root, partial, observed_jobs, expected_jobs, scenario_count, tuple(episodes)
    )


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"required artifact does not exist: {path}")


def _require_nonempty(path: Path) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"required non-empty artifact does not exist: {path}")




def build_matrix_statistics(episodes: Sequence[EpisodeSummary]) -> dict[str, Any]:
    """Build per-scenario statistics from already validated episode summaries."""

    grouped: defaultdict[tuple[str, float], list[CompletedEpisodeSummary]] = defaultdict(list)
    for episode in episodes:
        if isinstance(episode, CompletedEpisodeSummary):
            grouped[(episode.scenario.name, episode.traffic_density)].append(episode)
    statistics: dict[str, Any] = {}
    for (scenario, density), group in sorted(grouped.items()):
        metrics = [episode.metrics for episode in group]
        ml_per_km: list[float] = []
        for item in metrics:
            value = item.energy.ml_per_km
            if value is None:
                raise ValueError(
                    "matrix cannot bootstrap energy_ml_per_km for a completed zero-distance episode"
                )
            ml_per_km.append(value)
        statistics[f"{scenario}/density_{density:.2f}"] = {
            "aggregation_unit": "evaluation_episode",
            "episode_count": len(group),
            "metrics": {
                "simulated_seconds": bootstrap(
                    np.asarray([item.simulated_seconds for item in metrics], dtype=np.float64)
                ),
                "distance_m": bootstrap(
                    np.asarray([item.distance_m for item in metrics], dtype=np.float64)
                ),
                "energy_total_ml": bootstrap(
                    np.asarray([item.energy.total_ml for item in metrics], dtype=np.float64)
                ),
                "energy_ml_per_km": bootstrap(np.asarray(ml_per_km, dtype=np.float64)),
                "route_completion": bootstrap(
                    np.asarray([item.route_completion for item in metrics], dtype=np.float64)
                ),
                "mean_speed_mps": bootstrap(
                    np.asarray([item.speed_mps.mean for item in metrics], dtype=np.float64)
                ),
                "total_reward": bootstrap(
                    np.asarray([item.total_reward for item in metrics], dtype=np.float64)
                ),
            },
            "arrive_rate": float(np.mean([item.arrive_dest for item in metrics])),
            "collision_rate": float(np.mean([item.collision for item in metrics])),
            "out_of_road_rate": float(np.mean([item.out_of_road for item in metrics])),
        }
    return statistics


def bootstrap(values: np.ndarray) -> dict[str, float | list[float]]:
    """Return fixed-seed mean, median, and bootstrap interval for one metric."""

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


_IGNORED_METADATA_FIELDS = frozenset(
    {
        "elapsed_seconds",
        "cuda_memory",
        "process_id",
    }
)


def compare_artifact_trees(serial_root: Path, parallel_root: Path) -> dict[str, Any]:
    """Require exact job summaries and trace arrays outside run-specific metadata."""

    serial_jobs = _indexed_jobs(serial_root)
    parallel_jobs = _indexed_jobs(parallel_root)
    if set(serial_jobs) != set(parallel_jobs):
        raise ValueError("serial and parallel roots contain different matrix job grids")
    compared_arrays = 0
    compared_episodes = 0
    for key in sorted(serial_jobs):
        serial_job = serial_jobs[key]
        parallel_job = parallel_jobs[key]
        serial_summary = load_job_summary(serial_job / "summary.json")
        parallel_summary = load_job_summary(parallel_job / "summary.json")
        if _stable_json(serial_summary.model_dump(mode="json")) != _stable_json(
            parallel_summary.model_dump(mode="json")
        ):
            raise ValueError(f"job summary mismatch for seed={key[0]}, density={key[1]}")
        serial_episodes = _episodes_by_name(serial_summary)
        parallel_episodes = _episodes_by_name(parallel_summary)
        if set(serial_episodes) != set(parallel_episodes):
            raise ValueError(f"episode grid mismatch for seed={key[0]}, density={key[1]}")
        for name in sorted(serial_episodes):
            serial_trace = load_trace_artifact(serial_job / name / "trace.npz")
            parallel_trace = load_trace_artifact(parallel_job / name / "trace.npz")
            if set(serial_trace.arrays) != set(parallel_trace.arrays):
                raise ValueError(f"trace field mismatch for {key}/{name}")
            for field in sorted(serial_trace.arrays):
                if not np.array_equal(serial_trace.arrays[field], parallel_trace.arrays[field]):
                    raise ValueError(f"trace array mismatch for {key}/{name}/{field}")
                compared_arrays += 1
            compared_episodes += 1
    return {
        "job_count": len(serial_jobs),
        "episode_count": compared_episodes,
        "array_count": compared_arrays,
        "equal": True,
    }


def _indexed_jobs(root: Path) -> dict[tuple[int, float], Path]:
    result: dict[tuple[int, float], Path] = {}
    for job in sorted(
        (path for path in root.iterdir() if path.is_dir() and path.name.isdigit()),
        key=lambda path: int(path.name),
    ):
        summary = load_job_summary(job / "summary.json")
        key = (summary.runtime.seed, summary.episodes[0].traffic_density)
        if key in result:
            raise ValueError(f"duplicate matrix job key: {key}")
        result[key] = job
    if not result:
        raise ValueError(f"artifact root contains no numbered jobs: {root}")
    return result


def _episodes_by_name(summary: JobSummary) -> dict[str, EpisodeSummary]:
    result: dict[str, EpisodeSummary] = {}
    for episode in summary.episodes:
        name = episode.scenario.name
        if name in result:
            raise ValueError("job summary contains an invalid episode name")
        result[name] = episode
    return result


def _stable_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _stable_json(child)
            for key, child in value.items()
            if key not in _IGNORED_METADATA_FIELDS
        }
    if isinstance(value, list):
        return [_stable_json(child) for child in value]
    return value
