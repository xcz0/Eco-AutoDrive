"""Offline evaluation artifact validation, statistics, reports, and comparison."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from eco_planner.evaluation.artifacts import (
    CompletedEpisodeSummary,
    EpisodeSummary,
    JobSummary,
    load_episode_summary,
    load_job_summary,
    load_runtime_metadata,
    load_trace_artifact,
    validate_trace_arrays,
    write_json,
)

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
            "energy_total_ml": episode.energy.total_ml if episode.status == "completed" else None,
            "energy_ml_per_km": episode.energy.ml_per_km if episode.status == "completed" else None,
            "route_completion": (
                episode.route_completion if episode.status == "completed" else None
            ),
            "mean_speed_mps": episode.speed_mps.mean if episode.status == "completed" else None,
            "total_reward": episode.total_reward if episode.status == "completed" else None,
        }
        for episode in sorted(
            episodes,
            key=lambda item: (item.scenario.name, item.traffic_density, item.noise_seed),
        )
    ]
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
    seeds: tuple[int, ...]
    densities: tuple[float, ...]
    scenario_names: tuple[str, ...]
    warmup_steps: int
    video_enabled: bool

    @property
    def expected_jobs(self) -> set[tuple[int, float]]:
        return {(seed, density) for seed in self.seeds for density in self.densities}


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
        job_spec = _read_matrix_spec(job_dir / "resolved_config.yaml")
        if matrix_spec is None:
            matrix_spec = job_spec
        elif matrix_spec != job_spec:
            raise ValueError(f"job {job_dir} resolved matrix specification disagrees")
        _require_nonempty(job_dir / ".hydra" / "overrides.yaml")
        metadata = load_runtime_metadata(job_dir / "runtime_metadata.json")
        _require_file(job_dir / "tracked_diff.patch")
        job_summary = load_job_summary(job_dir / "summary.json")
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
            _validate_episode_summary(episode, job_dir, seed, density)
            episode_dir = job_dir / episode.scenario.name
            if load_episode_summary(episode_dir / "summary.json") != episode:
                raise ValueError(f"episode summary copy disagrees with job summary: {episode_dir}")
            if matrix_spec.video_enabled:
                _require_nonempty(episode_dir / "closed_loop.gif")
            _validate_trace(
                episode_dir / "trace.npz", episode, warmup_steps=matrix_spec.warmup_steps
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


def _validate_trace(path: Path, episode: EpisodeSummary, *, warmup_steps: int) -> None:
    _require_nonempty(path)
    arrays = load_trace_artifact(path).arrays
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
    if float(arrays["trajectory_position_errors_m"].max()) >= POSITION_ERROR_LIMIT_M:
        raise ValueError(f"trace {path} exceeds the trajectory position error limit")
    if float(arrays["trajectory_heading_errors_rad"].max()) >= HEADING_ERROR_LIMIT_RAD:
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
    if episode.status != "completed":
        return
    if not 2_000.0 <= episode.route_length_m <= 5_000.0:
        raise ValueError(f"job {job_dir} route length is outside [2000, 5000] m")


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"required artifact does not exist: {path}")


def _require_nonempty(path: Path) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"required non-empty artifact does not exist: {path}")


def _read_matrix_spec(path: Path) -> MatrixSpec:
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
    return MatrixSpec(seeds, densities, names, warmup_steps, video_enabled)


def build_matrix_statistics(episodes: Sequence[EpisodeSummary]) -> dict[str, Any]:
    """Build per-scenario statistics from already validated episode summaries."""

    grouped: defaultdict[tuple[str, float], list[CompletedEpisodeSummary]] = defaultdict(list)
    for episode in episodes:
        if isinstance(episode, CompletedEpisodeSummary):
            grouped[(episode.scenario.name, episode.traffic_density)].append(episode)
    statistics: dict[str, Any] = {}
    for (scenario, density), group in sorted(grouped.items()):
        ml_per_km: list[float] = []
        for episode in group:
            value = episode.energy.ml_per_km
            if value is None:
                raise ValueError(
                    "matrix cannot bootstrap energy_ml_per_km for a completed zero-distance episode"
                )
            ml_per_km.append(value)
        statistics[f"{scenario}/density_{density:.2f}"] = {
            "episode_count": len(group),
            "metrics": {
                "simulated_seconds": bootstrap(
                    np.asarray([episode.simulated_seconds for episode in group], dtype=np.float64)
                ),
                "distance_m": bootstrap(
                    np.asarray([episode.distance_m for episode in group], dtype=np.float64)
                ),
                "energy_total_ml": bootstrap(
                    np.asarray([episode.energy.total_ml for episode in group], dtype=np.float64)
                ),
                "energy_ml_per_km": bootstrap(np.asarray(ml_per_km, dtype=np.float64)),
                "route_completion": bootstrap(
                    np.asarray([episode.route_completion for episode in group], dtype=np.float64)
                ),
                "mean_speed_mps": bootstrap(
                    np.asarray([episode.speed_mps.mean for episode in group], dtype=np.float64)
                ),
                "total_reward": bootstrap(
                    np.asarray([episode.total_reward for episode in group], dtype=np.float64)
                ),
            },
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
