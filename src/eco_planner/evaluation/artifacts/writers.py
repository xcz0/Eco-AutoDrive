"""Evaluation artifact persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel

from eco_planner.evaluation.artifacts.models import CompletedEpisodeSummary, FailedEpisodeSummary
from eco_planner.evaluation.config import VideoConfig


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
            from eco_planner.evaluation.artifacts.video import write_gif

            write_gif(frames, output_dir / "closed_loop.gif", video_config.fps)


def write_json(path: Path, payload: Any) -> None:
    """Persist a Pydantic JSON artifact with stable formatting."""

    if isinstance(payload, BaseModel):
        payload = payload.model_dump(mode="json")
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
