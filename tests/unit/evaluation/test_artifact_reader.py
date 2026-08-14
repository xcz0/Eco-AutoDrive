from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from eco_planner.evaluation.artifacts.io import load_job_summary, load_trace_artifact


def test_reader_rejects_legacy_json_and_trace_artifacts(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    trace_path = tmp_path / "trace.npz"
    summary = {"terminal_reason": "crash_vehicle"}
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    np.savez_compressed(trace_path, initial_state=np.ones(7, dtype=np.float64))

    with pytest.raises(ValueError, match="schema version 4"):
        load_job_summary(summary_path)
    with pytest.raises(ValueError, match="missing schema version"):
        load_trace_artifact(trace_path)


def test_reader_rejects_unknown_v4_json_artifact(tmp_path: Path) -> None:
    path = tmp_path / "unknown.json"
    path.write_text('{"schema_version": 4, "unknown": true}', encoding="utf-8")

    with pytest.raises(ValidationError, match="Field required"):
        load_job_summary(path)
