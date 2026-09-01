"""Evaluation-semantic validation over already structurally valid artifacts."""

from __future__ import annotations

from pathlib import Path

from eco_planner.evaluation.artifacts import load_trace_artifact
from eco_planner.evaluation.models import CompletedEpisodeSummary, EpisodeSummary
from eco_planner.evaluation.trace import validate_trace_arrays

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
    arrays = load_trace_artifact(path).arrays
    completed = isinstance(episode, CompletedEpisodeSummary)
    validate_trace_arrays(
        arrays,
        expected_plan_cycles=episode.plan_cycles if completed else None,
        expected_simulator_steps=episode.simulator_steps if completed else None,
        expected_warmup_steps=warmup_steps if completed else None,
        require_traffic=completed and require_traffic,
        expected_trace_status=episode.trace_status,
    )
    if not completed:
        return
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
