from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from eco_planner.evaluation.artifact_reader import load_json_artifact, load_trace_artifact


def test_v1_reader_marks_route_speed_fields_unavailable_without_rewriting(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    trace_path = tmp_path / "trace.npz"
    summary = {"terminal_reason": "crash_vehicle"}
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    np.savez_compressed(trace_path, initial_state=np.ones(7, dtype=np.float64))

    loaded_summary = load_json_artifact(summary_path)
    loaded_trace = load_trace_artifact(trace_path)

    assert loaded_summary["schema_version"] == 1
    assert loaded_summary["termination"] == {
        "type": "collision",
        "detail": "crash_vehicle",
    }
    assert loaded_trace.source_schema_version == 1
    assert loaded_trace.unavailable_fields == {
        "observation_route_lanes_speed_limit",
        "observation_route_lanes_has_speed_limit",
    }
    assert set(loaded_trace.arrays) == {"initial_state"}
    assert json.loads(summary_path.read_text(encoding="utf-8")) == summary
