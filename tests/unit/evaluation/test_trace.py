from __future__ import annotations

import numpy as np
import pytest
import torch

from eco_planner.envs import TrafficFrame, TrajectoryExecutionRecord
from eco_planner.evaluation.artifacts.trace_recorder import EpisodeTraceRecorder
from eco_planner.evaluation.artifacts.trace_schema import validate_trace_arrays
from eco_planner.evaluation.runtime.contracts import HostGuidanceDiagnostics, HostInferenceResult


def _observation() -> dict[str, torch.Tensor]:
    return {
        "ego_current_state": torch.zeros((1, 10)),
        "neighbor_agents_past": torch.zeros((1, 32, 21, 11)),
        "static_objects": torch.zeros((1, 5, 10)),
        "lanes": torch.zeros((1, 70, 20, 12)),
        "lanes_speed_limit": torch.zeros((1, 70, 1)),
        "lanes_has_speed_limit": torch.zeros((1, 70, 1), dtype=torch.bool),
        "route_lanes": torch.zeros((1, 25, 20, 12)),
        "route_lanes_speed_limit": torch.full((1, 25, 1), 13.0),
        "route_lanes_has_speed_limit": torch.ones((1, 25, 1), dtype=torch.bool),
    }


def _step_info() -> dict[str, object]:
    states = np.zeros((5, 7), dtype=np.float64)
    return {
        "trajectory_start_center": np.zeros(2, dtype=np.float64),
        "trajectory_start_heading": 0.0,
        "trajectory_world_centers": np.zeros((80, 2), dtype=np.float64),
        "trajectory_world_headings": np.zeros(80),
        "trajectory_substep_states": states,
        "trajectory_substep_rewards": np.ones(5),
        "trajectory_substep_dense_rewards": np.ones(5),
        "trajectory_substep_terminated": np.array([False, False, False, False, True]),
        "trajectory_substep_truncated": np.zeros(5, dtype=np.bool_),
        "trajectory_target_centers": states[:, :2],
        "trajectory_target_headings": states[:, 2],
        "trajectory_position_errors_m": np.zeros(5),
        "trajectory_heading_errors_rad": np.zeros(5),
        "traffic_substep_frames": tuple(
            TrafficFrame(index, (0.0, 0.0), 0.0, 1.0, (), ()) for index in range(5)
        ),
        "route_completion": 1.0,
        "arrive_dest": True,
        "out_of_road": False,
        "crash_vehicle": False,
        "crash_object": False,
        "crash_building": False,
        "crash_human": False,
        "max_step": False,
    }


def _execution() -> TrajectoryExecutionRecord:
    return TrajectoryExecutionRecord.from_info(_step_info())


def test_trace_recorder_finalizes_stable_schema_once() -> None:
    recorder = EpisodeTraceRecorder.from_initial_state(
        np.zeros(7), max_plan_cycles=1, max_warmup_steps=0, guided=False
    )
    noise = np.zeros((1, 11, 80, 4), dtype=np.float32)
    recorder.append_cycle(
        np.zeros(7),
        _observation(),
        HostInferenceResult(initial_noise=noise, prediction=noise),
        _execution(),
        0,
        None,
    )

    arrays = recorder.finalize()

    assert arrays["initial_noise"].shape == (1, 11, 80, 4)
    assert arrays["observation_route_lanes_speed_limit"].shape == (1, 25, 1)
    assert arrays["observation_route_lanes_has_speed_limit"].dtype == np.bool_
    assert arrays["observation_neighbor_agents_past"].shape == (1, 32, 21, 11)
    assert arrays["executed_states"].shape == (5, 7)
    assert arrays["warmup_states"].shape == (0, 7)
    assert arrays["traffic_selected_ids"].shape == (1, 32)
    with pytest.raises(RuntimeError, match="already finalized"):
        recorder.finalize()


def test_empty_trace_records_explicit_invalid_initial_states() -> None:
    arrays = EpisodeTraceRecorder.empty().finalize("empty")

    assert arrays["trace_status"].item() == "empty"
    assert not arrays["warmup_initial_state_valid"].item()
    assert not arrays["initial_state_valid"].item()
    validate_trace_arrays(arrays, expected_trace_status="empty")

    with pytest.raises(RuntimeError, match="complete trace"):
        EpisodeTraceRecorder.empty().finalize("complete")


def test_trace_recorder_rejects_misaligned_step_arrays() -> None:
    info = _step_info()
    info["trajectory_target_centers"] = np.zeros((4, 2))

    with pytest.raises(ValueError, match="trajectory_target_centers"):
        TrajectoryExecutionRecord.from_info(info)


@pytest.mark.parametrize("value", [np.float32(0.5), np.float64(0.5), np.int64(1)])
def test_execution_record_accepts_finite_numpy_route_completion(value: np.generic) -> None:
    info = _step_info()
    info["route_completion"] = value

    execution = TrajectoryExecutionRecord.from_info(info)

    assert execution.route_completion == pytest.approx(float(value))


@pytest.mark.parametrize("value", [True, np.bool_(False), np.array(0.5), np.nan, np.inf])
def test_execution_record_rejects_invalid_route_completion(value: object) -> None:
    info = _step_info()
    info["route_completion"] = value

    with pytest.raises(TypeError, match="route_completion"):
        TrajectoryExecutionRecord.from_info(info)


def test_trace_validator_rejects_unexpected_arrays() -> None:
    arrays = EpisodeTraceRecorder.empty().finalize("empty")
    arrays["legacy_field"] = np.zeros(1)

    with pytest.raises(ValueError, match="unexpected arrays"):
        validate_trace_arrays(arrays)


def test_trace_validator_rejects_artifact_v3_dtype_change() -> None:
    arrays = EpisodeTraceRecorder.empty().finalize("empty")
    arrays["initial_noise"] = arrays["initial_noise"].astype(np.float64)

    with pytest.raises(TypeError, match="initial_noise.*float32"):
        validate_trace_arrays(arrays)


def test_guided_trace_requires_and_persists_reference_action_targets_and_step_diagnostics() -> None:
    recorder = EpisodeTraceRecorder.from_initial_state(
        np.zeros(7), max_plan_cycles=1, max_warmup_steps=0, guided=True
    )
    prediction = np.zeros((1, 11, 80, 4), dtype=np.float32)
    reference = np.ones_like(prediction)
    steps = np.arange(5, dtype=np.float32)[None]
    diagnostics = HostGuidanceDiagnostics(
        lateral_target_offset_m=np.array([2.5], dtype=np.float32),
        longitudinal_target_speed_fraction=np.array([0.0], dtype=np.float32),
        longitudinal_target_speed_delta_mps=np.zeros((1, 80), dtype=np.float32),
        lateral_objective_delta=steps,
        longitudinal_objective_delta=steps + 1.0,
        applied_gradient_l2=steps + 2.0,
        applied_gradient_max_abs=steps + 3.0,
        raw_neighbor_gradient_l2=steps + 4.0,
        zero_speed_count=np.arange(5, dtype=np.int64)[None],
    )
    result = HostInferenceResult(
        initial_noise=np.zeros_like(prediction),
        prediction=prediction,
        reference_prediction=reference,
        guidance_action=np.array([[1.0, 0.0]], dtype=np.float32),
        guidance_diagnostics=diagnostics,
    )

    recorder.append_cycle(
        np.zeros(7),
        _observation(),
        result,
        _execution(),
        0,
        None,
    )
    arrays = recorder.finalize()

    assert arrays["reference_predictions_local"].shape == (1, 11, 80, 4)
    assert arrays["guidance_actions"].shape == (1, 2)
    assert arrays["guidance_lateral_objective_delta"].shape == (1, 5)
    assert arrays["guidance_zero_speed_count"].dtype.kind in "iu"


def test_partial_trace_returns_only_recorded_warmup_steps() -> None:
    recorder = EpisodeTraceRecorder.from_initial_state(
        np.zeros(7), max_plan_cycles=2, max_warmup_steps=10, guided=False
    )
    recorder.append_warmup(
        _execution(),
        np.ones(5, dtype=np.int64),
        np.zeros(5, dtype=np.int64),
    )

    arrays = recorder.finalize("partial")

    assert arrays["warmup_states"].shape == (5, 7)
    assert arrays["initial_noise"].shape == (0, 11, 80, 4)
    assert arrays["executed_states"].shape == (0, 7)


def test_trace_recorder_rejects_planning_capacity_overflow() -> None:
    recorder = EpisodeTraceRecorder.from_initial_state(
        np.zeros(7), max_plan_cycles=0, max_warmup_steps=0, guided=False
    )
    prediction = np.zeros((1, 11, 80, 4), dtype=np.float32)

    with pytest.raises(RuntimeError, match="planning trace capacity"):
        recorder.append_cycle(
            np.zeros(7),
            _observation(),
            HostInferenceResult(initial_noise=prediction, prediction=prediction),
            _execution(),
            0,
            None,
        )


def test_trace_recorder_rejects_warmup_capacity_overflow() -> None:
    recorder = EpisodeTraceRecorder.from_initial_state(
        np.zeros(7), max_plan_cycles=0, max_warmup_steps=4, guided=False
    )

    with pytest.raises(RuntimeError, match="warmup trace capacity"):
        recorder.append_warmup(
            _execution(),
            np.ones(5, dtype=np.int64),
            np.zeros(5, dtype=np.int64),
        )
