from __future__ import annotations

import json
from pathlib import Path

import pytest

from eco_planner.evaluation import EpisodeFailure
from eco_planner.evaluation.artifacts import ARTIFACT_SCHEMA_VERSION
from eco_planner.evaluation.trace import EpisodeTraceRecorder, validate_trace_arrays


def test_empty_v2_trace_has_explicit_state_validity() -> None:
    arrays = EpisodeTraceRecorder.empty().finalize("empty")

    assert arrays["schema_version"].item() == ARTIFACT_SCHEMA_VERSION
    assert arrays["trace_status"].item() == "empty"
    assert not arrays["warmup_initial_state_valid"].item()
    assert not arrays["initial_state_valid"].item()
    assert arrays["initial_noise"].shape == (0, 11, 80, 4)
    assert arrays["executed_states"].shape == (0, 7)
    assert arrays["observation_route_lanes_speed_limit"].shape == (0, 25, 1)
    assert arrays["observation_route_lanes_has_speed_limit"].shape == (0, 25, 1)
    validate_trace_arrays(arrays, expected_trace_status="empty")


def test_episode_failure_preserves_stage_and_original_error() -> None:
    cause = RuntimeError("planner produced a non-finite trajectory")

    failure = EpisodeFailure("inference", cause)

    assert failure.stage == "inference"
    assert failure.cause is cause
    assert str(failure) == "inference: planner produced a non-finite trajectory"


def test_v2_failure_summary_is_json_serializable(tmp_path: Path) -> None:
    payload = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": "failed",
        "termination": {"type": "runtime_error", "detail": "inference"},
        "failure": {
            "stage": "inference",
            "exception_type": "RuntimeError",
            "message": "boom",
            "traceback": "RuntimeError: boom",
        },
    }
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert json.loads(path.read_text(encoding="utf-8")) == payload


def test_empty_trace_rejects_completed_status() -> None:
    with pytest.raises(RuntimeError, match="complete trace"):
        EpisodeTraceRecorder.empty().finalize("complete")
