"""Evaluation artifact persistence and current-schema readers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
from pydantic import BaseModel, TypeAdapter

from eco_planner.artifacts import write_npz
from eco_planner.evaluation.models import (
    CompletedEpisodeSummary,
    EpisodeSummary,
    FailedEpisodeSummary,
    JobSummary,
    RuntimeMetadata,
)
from eco_planner.evaluation.trace import LoadedTraceArtifact, validate_trace_arrays

if TYPE_CHECKING:
    from eco_planner.evaluation.config import VideoConfig


_EPISODE_ADAPTER = TypeAdapter(EpisodeSummary)
_Artifact = TypeVar("_Artifact", JobSummary, RuntimeMetadata)


def load_job_summary(path: Path) -> JobSummary:
    """Load a typed current-schema job summary without compatibility conversion."""

    return _load_json(path, JobSummary)


def load_episode_summary(path: Path) -> EpisodeSummary:
    """Load a typed current-schema episode summary without compatibility conversion."""

    return _load_episode_json(path)


def load_runtime_metadata(path: Path) -> RuntimeMetadata:
    """Load typed current-schema runtime metadata without compatibility conversion."""

    return _load_json(path, RuntimeMetadata)


def load_trace_artifact(path: Path) -> LoadedTraceArtifact:
    """Load current-schema NPZ arrays without synthesizing missing fields."""

    with np.load(path, allow_pickle=False) as trace:
        arrays = {name: trace[name] for name in trace.files}
    status = str(arrays["trace_status"].item())
    validate_trace_arrays(arrays, expected_trace_status=status)
    return LoadedTraceArtifact(status, arrays)


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

    output_dir.mkdir(parents=True, exist_ok=False)
    write_npz(output_dir / "trace.npz", trace_arrays)
    write_json(output_dir / "summary.json", summary)
    if video_config.enabled:
        if summary.status == "completed" and not frames:
            raise RuntimeError("video output was enabled but no frames were rendered")
        if frames:
            from eco_planner.evaluation.rendering import write_gif

            write_gif(frames, output_dir / "closed_loop.gif", video_config.fps)


def write_json(path: Path, payload: Any) -> None:
    """Persist a Pydantic JSON artifact with stable formatting."""

    if isinstance(payload, BaseModel):
        payload = payload.model_dump(mode="json")
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
