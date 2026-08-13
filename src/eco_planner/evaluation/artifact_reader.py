"""Read-only loading for the current evaluation artifact schema."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import TypeAdapter

from eco_planner.evaluation.schema import (
    ARTIFACT_SCHEMA_VERSION,
    EpisodeSummary,
    JobSummary,
    RuntimeMetadata,
)
from eco_planner.evaluation.trace import validate_trace_arrays

_EPISODE_ADAPTER = TypeAdapter(EpisodeSummary)


@dataclass(frozen=True)
class LoadedTraceArtifact:
    """Validated current-schema trace arrays."""

    trace_status: str
    arrays: dict[str, np.ndarray]


def load_json_artifact(path: Path) -> dict[str, Any]:
    """Load a current-schema JSON artifact without compatibility conversion."""

    text = path.read_text(encoding="utf-8")
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    if payload.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(f"artifact must use schema version {ARTIFACT_SCHEMA_VERSION}: {path}")
    if "episodes" in payload:
        return JobSummary.model_validate_json(text).model_dump(mode="json")
    if "scenario" in payload and "status" in payload:
        return _EPISODE_ADAPTER.validate_json(text).model_dump(mode="json")
    if "inference_runtime" in payload:
        return RuntimeMetadata.model_validate_json(text).model_dump(mode="json")
    raise ValueError(f"unrecognized schema v{ARTIFACT_SCHEMA_VERSION} JSON artifact: {path}")


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
