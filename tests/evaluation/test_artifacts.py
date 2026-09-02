"""Trace artifact structural-boundary regression tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eco_planner.evaluation import load_trace_artifact
from eco_planner.evaluation.episodes import EpisodeTraceRecorder


def _write_trace(path: Path, arrays: dict[str, np.ndarray]) -> None:
    np.savez(path, **arrays)  # pyright: ignore[reportArgumentType]


def test_trace_reader_rejects_wrong_shape(tmp_path: Path) -> None:
    arrays = EpisodeTraceRecorder.empty().finalize("empty")
    arrays["initial_state"] = np.zeros(8, dtype=np.float64)
    path = tmp_path / "trace.npz"
    _write_trace(path, arrays)

    with pytest.raises(ValueError, match="initial_state.*shape"):
        load_trace_artifact(path)


def test_trace_reader_rejects_wrong_dtype(tmp_path: Path) -> None:
    arrays = EpisodeTraceRecorder.empty().finalize("empty")
    arrays["initial_state"] = arrays["initial_state"].astype(np.float32)
    path = tmp_path / "trace.npz"
    _write_trace(path, arrays)

    with pytest.raises(TypeError, match="initial_state.*dtype"):
        load_trace_artifact(path)


def test_trace_reader_rejects_non_finite_values(tmp_path: Path) -> None:
    arrays = EpisodeTraceRecorder.empty().finalize("empty")
    arrays["initial_state"][0] = np.nan
    path = tmp_path / "trace.npz"
    _write_trace(path, arrays)

    with pytest.raises(ValueError, match="initial_state.*non-finite"):
        load_trace_artifact(path)
