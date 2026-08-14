"""Artifact v3 episode persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from metadrive.utils.doc_utils import generate_gif
from pydantic import BaseModel

from eco_planner.evaluation.config import VideoConfig
from eco_planner.evaluation.schema import CompletedEpisodeSummary, FailedEpisodeSummary


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
    if isinstance(payload, BaseModel):
        payload = payload.model_dump(mode="json")
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
