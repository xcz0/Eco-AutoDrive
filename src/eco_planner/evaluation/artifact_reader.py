"""Read-only loading and normalization for evaluation artifact schemas."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from eco_planner.evaluation.artifacts import ARTIFACT_SCHEMA_VERSION

_V2_ROUTE_TRACE_FIELDS = frozenset(
    {
        "observation_route_lanes_speed_limit",
        "observation_route_lanes_has_speed_limit",
    }
)


@dataclass(frozen=True)
class LoadedTraceArtifact:
    """Trace arrays plus explicit schema fields unavailable in the source artifact."""

    source_schema_version: int
    trace_status: str
    arrays: dict[str, np.ndarray]
    unavailable_fields: frozenset[str]


def load_json_artifact(path: Path) -> dict[str, Any]:
    """Load a JSON artifact and normalize the v1 status/termination envelope."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    if "schema_version" in payload:
        if payload["schema_version"] != ARTIFACT_SCHEMA_VERSION:
            raise ValueError(f"unsupported artifact schema version: {path}")
        return payload
    normalized = dict(payload)
    normalized["schema_version"] = 1
    normalized["status"] = "completed"
    if "episodes" in normalized:
        episodes = normalized["episodes"]
        if not isinstance(episodes, list):
            raise TypeError(f"v1 job episodes must be a list: {path}")
        normalized["episodes"] = [_normalize_v1_episode(value, path) for value in episodes]
    elif "terminal_reason" in normalized:
        normalized = _normalize_v1_episode(normalized, path)
    return normalized


def load_trace_artifact(path: Path) -> LoadedTraceArtifact:
    """Load v1 or v2 NPZ arrays without synthesizing fields absent from v1."""

    with np.load(path, allow_pickle=False) as trace:
        arrays = {name: trace[name] for name in trace.files}
    if "schema_version" not in arrays:
        return LoadedTraceArtifact(1, "complete", arrays, _V2_ROUTE_TRACE_FIELDS)
    version = int(arrays["schema_version"].item())
    if version != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(f"unsupported trace schema version {version}: {path}")
    status = str(arrays["trace_status"].item())
    return LoadedTraceArtifact(version, status, arrays, frozenset())


def _normalize_v1_episode(value: object, path: Path) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"v1 episode summary must be an object: {path}")
    episode = dict(value)
    episode["schema_version"] = 1
    episode["status"] = "completed"
    episode["trace_status"] = "complete"
    reason = episode.get("terminal_reason")
    if not isinstance(reason, str) or not reason:
        raise ValueError(f"v1 episode is missing terminal_reason: {path}")
    if reason == "arrive_dest":
        kind = "arrive_dest"
    elif reason == "out_of_road":
        kind = "out_of_road"
    elif reason.startswith("crash_"):
        kind = "collision"
    elif reason in {"max_step", "truncated"}:
        kind = "time_truncation"
    else:
        kind = "runtime_error"
    episode["termination"] = {"type": kind, "detail": reason}
    return episode
