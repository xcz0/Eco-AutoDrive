"""Typed evaluation artifact persistence and readers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

import numpy as np
from pydantic import TypeAdapter

from .models import (
    CompletedEpisodeSummary,
    EpisodeSummary,
    FailedEpisodeSummary,
    JobSummary,
    RuntimeMetadata,
)
from .trace import LoadedTraceArtifact, validate_trace_arrays

if TYPE_CHECKING:
    from ..config import VideoConfig


POSITION_ERROR_LIMIT_M = 1e-3
HEADING_ERROR_LIMIT_RAD = 1e-4


_EPISODE_ADAPTER = TypeAdapter(EpisodeSummary)
_Artifact = TypeVar("_Artifact", JobSummary, RuntimeMetadata)


def load_job_summary(path: Path) -> JobSummary:
    """Load a typed job summary."""

    return _load_json(path, JobSummary)


def load_episode_summary(path: Path) -> EpisodeSummary:
    """Load a typed episode summary."""

    return _load_episode_json(path)


def load_runtime_metadata(path: Path) -> RuntimeMetadata:
    """Load typed runtime metadata."""

    return _load_json(path, RuntimeMetadata)


def load_trace_artifact(path: Path) -> LoadedTraceArtifact:
    """Load and structurally validate one NPZ trace."""

    with np.load(path, allow_pickle=False) as trace:
        arrays = {name: trace[name] for name in trace.files}
    validate_trace_arrays(arrays)
    status = str(arrays["trace_status"].item())
    return LoadedTraceArtifact(status, arrays)


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


def _load_json(path: Path, model: type[_Artifact]) -> _Artifact:
    return model.model_validate_json(path.read_text(encoding="utf-8"))


def _load_episode_json(path: Path) -> EpisodeSummary:
    return _EPISODE_ADAPTER.validate_json(path.read_text(encoding="utf-8"))


def write_episode_artifacts(
    output_dir: Path,
    trace_arrays: dict[str, np.ndarray],
    frames: list[np.ndarray],
    summary: CompletedEpisodeSummary | FailedEpisodeSummary,
    video_config: VideoConfig,
) -> None:
    """Persist one finalized episode without recomputing trace arrays."""

    from eco_planner.artifacts import write_json, write_npz

    output_dir.mkdir(parents=True, exist_ok=False)
    write_npz(output_dir / "trace.npz", trace_arrays)
    write_json(output_dir / "summary.json", summary)
    if video_config.enabled:
        if summary.status == "completed" and not frames:
            raise RuntimeError("video output was enabled but no frames were rendered")
        if frames:
            from ..episodes.rendering import write_gif

            write_gif(frames, output_dir / "closed_loop.gif", video_config.fps)


def _require_nonempty(path: Path) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"required non-empty artifact does not exist: {path}")
