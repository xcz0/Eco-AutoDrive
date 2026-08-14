"""Exact comparison of serial and parallel evaluation artifact trees."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from eco_planner.evaluation.artifacts.io import load_job_summary, load_trace_artifact
from eco_planner.evaluation.artifacts.models import EpisodeSummary, JobSummary

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
