"""Artifact validation for the fixed MetaDrive traffic matrix."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from eco_planner.evaluation.artifacts.models import EpisodeSummary
from eco_planner.evaluation.artifacts.readers import (
    load_episode_summary,
    load_job_summary,
    load_runtime_metadata,
    load_trace_artifact,
)
from eco_planner.evaluation.artifacts.trace_schema import validate_trace_arrays

POSITION_ERROR_LIMIT_M = 1e-3
HEADING_ERROR_LIMIT_RAD = 1e-4


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
