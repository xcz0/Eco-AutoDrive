"""Read-only loading for the current evaluation artifact schema."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
from metadrive.utils.doc_utils import generate_gif
from pydantic import BaseModel, TypeAdapter

from eco_planner.evaluation.artifacts.models import (
    ARTIFACT_SCHEMA_VERSION,
    CompletedEpisodeSummary,
    EpisodeSummary,
    FailedEpisodeSummary,
    JobSummary,
    RuntimeMetadata,
)
from eco_planner.evaluation.artifacts.trace_schema import validate_trace_arrays
from eco_planner.evaluation.config import VideoConfig

_EPISODE_ADAPTER = TypeAdapter(EpisodeSummary)
_Artifact = TypeVar("_Artifact", JobSummary, RuntimeMetadata)


@dataclass(frozen=True)
class LoadedTraceArtifact:
    """Validated current-schema trace arrays."""

    trace_status: str
    arrays: dict[str, np.ndarray]


def load_job_summary(path: Path) -> JobSummary:
    """Load a typed current-schema job summary without compatibility conversion."""

    return _load_json(path, JobSummary)


def load_episode_summary(path: Path) -> EpisodeSummary:
    """Load a typed current-schema episode summary without compatibility conversion."""

    return _load_episode_json(path)


def load_runtime_metadata(path: Path) -> RuntimeMetadata:
    """Load typed current-schema runtime metadata without compatibility conversion."""

    return _load_json(path, RuntimeMetadata)


def _load_json(path: Path, model: type[_Artifact]) -> _Artifact:
    text = path.read_text(encoding="utf-8")
    _require_schema_version(path, text)
    return model.model_validate_json(text)


def _load_episode_json(path: Path) -> EpisodeSummary:
    text = path.read_text(encoding="utf-8")
    _require_schema_version(path, text)
    return _EPISODE_ADAPTER.validate_json(text)


def _require_schema_version(path: Path, text: str) -> None:
    """Reject unrecognized and stale JSON before discriminated Pydantic validation."""

    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    if payload.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(f"artifact must use schema version {ARTIFACT_SCHEMA_VERSION}: {path}")


def load_trace_artifact(path: Path) -> LoadedTraceArtifact:
    """Load current-schema NPZ arrays without synthesizing missing fields."""

    with np.load(path, allow_pickle=False) as trace:
        arrays = {name: trace[name] for name in trace.files}
    if "schema_version" not in arrays:
        raise ValueError(f"trace is missing schema version: {path}")
    version = int(arrays["schema_version"].item())
    if version != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(f"unsupported trace schema version {version}: {path}")
    status = str(arrays["trace_status"].item())
    validate_trace_arrays(arrays, expected_trace_status=status)
    return LoadedTraceArtifact(status, arrays)


def write_episode_artifacts(
    output_dir: Path,
    trace_arrays: dict[str, np.ndarray],
    frames: list[np.ndarray],
    summary: CompletedEpisodeSummary | FailedEpisodeSummary,
    video_config: VideoConfig,
) -> None:
    """Persist one finalized episode without recomputing trace arrays."""

    output_dir.mkdir(parents=True, exist_ok=False)
    np.savez(output_dir / "trace.npz", **trace_arrays)
    write_json(output_dir / "summary.json", summary)
    if video_config.enabled:
        if summary.status == "completed" and not frames:
            raise RuntimeError("video output was enabled but no frames were rendered")
        if frames:
            duration_ms = round(1000 / video_config.fps)
            generate_gif(frames, str(output_dir / "closed_loop.gif"), duration=duration_ms)


def write_json(path: Path, payload: Any) -> None:
    """Persist a Pydantic JSON artifact with stable formatting."""

    if isinstance(payload, BaseModel):
        payload = payload.model_dump(mode="json")
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
