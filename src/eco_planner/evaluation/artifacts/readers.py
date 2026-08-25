"""Offline readers for the current evaluation artifact schema."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import numpy as np
from pydantic import TypeAdapter

from eco_planner.evaluation.artifacts.models import (
    EpisodeSummary,
    JobSummary,
    RuntimeMetadata,
)
from eco_planner.evaluation.artifacts.trace_schema import validate_trace_arrays

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
