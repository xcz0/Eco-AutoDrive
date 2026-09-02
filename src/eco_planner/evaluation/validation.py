"""Evaluation-semantic validation over already structurally valid artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .io import load_trace_artifact
from .models import CompletedEpisodeSummary, EpisodeSummary

POSITION_ERROR_LIMIT_M = 1e-3
HEADING_ERROR_LIMIT_RAD = 1e-4


def validate_episode_artifact(
    path: Path,
    episode: EpisodeSummary,
    *,
    warmup_steps: int,
    require_traffic: bool,
) -> None:
    """Check one trace against its typed episode result without recomputing metrics."""

    _require_nonempty(path)
    trace = load_trace_artifact(path)
    arrays = trace.arrays
    completed = isinstance(episode, CompletedEpisodeSummary)
    if trace.trace_status != episode.trace_status:
        raise ValueError("trace status disagrees with summary")
    if not completed:
        return
    if arrays["initial_noise"].shape[0] != episode.plan_cycles:
        raise ValueError("trace planning cycle count disagrees with summary")
    if arrays["executed_states"].shape[0] != episode.simulator_steps:
        raise ValueError("trace simulator step count disagrees with summary")
    if arrays["warmup_states"].shape[0] != warmup_steps:
        raise ValueError(f"trace must contain exactly {warmup_steps} warmup states")
    if require_traffic and not np.any(arrays["traffic_participant_counts"] > 0):
        raise ValueError("trace never observed traffic within the query radius")
    if float(arrays["trajectory_position_errors_m"].max()) >= POSITION_ERROR_LIMIT_M:
        raise ValueError(f"trace {path} exceeds the trajectory position error limit")
    if float(arrays["trajectory_heading_errors_rad"].max()) >= HEADING_ERROR_LIMIT_RAD:
        raise ValueError(f"trace {path} exceeds the trajectory heading error limit")


def validate_matrix_episode(
    episode: EpisodeSummary, job_dir: Path, seed: int, density: float
) -> None:
    """Check generic matrix pairing and traffic-route semantics from typed results."""

    if episode.noise_seed != seed:
        raise ValueError(f"job {job_dir} does not use the configured noise seed")
    if episode.scenario.seed != seed:
        raise ValueError(f"job {job_dir} does not use paired map/noise seeds")
    if episode.traffic_density != density:
        raise ValueError(f"job {job_dir} summary density disagrees with matrix workload")
    if (
        isinstance(episode, CompletedEpisodeSummary)
        and not 2_000.0 <= episode.route_length_m <= 5_000.0
    ):
        raise ValueError(f"job {job_dir} route length is outside [2000, 5000] m")


def _require_nonempty(path: Path) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"required non-empty artifact does not exist: {path}")
