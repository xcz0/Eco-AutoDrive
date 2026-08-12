from __future__ import annotations

import numpy as np
import pytest
import torch

from eco_planner.evaluation.trace import EpisodeTraceRecorder
from eco_planner.models.guidance import GuidanceDiagnostics
from eco_planner.models.pretrained import PlannerInferenceResult


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


def _step_info() -> dict[str, np.ndarray]:
    states = np.zeros((5, 7), dtype=np.float64)
    return {
        "trajectory_world_centers": np.zeros((80, 2), dtype=np.float64),
        "trajectory_world_headings": np.zeros(80),
        "trajectory_substep_states": states,
        "trajectory_substep_rewards": np.ones(5),
        "trajectory_substep_terminated": np.array([False, False, False, False, True]),
        "trajectory_substep_truncated": np.zeros(5, dtype=np.bool_),
        "trajectory_target_centers": states[:, :2],
        "trajectory_target_headings": states[:, 2],
        "trajectory_position_errors_m": np.zeros(5),
        "trajectory_heading_errors_rad": np.zeros(5),
    }


def test_trace_recorder_finalizes_stable_schema_once() -> None:
    recorder = EpisodeTraceRecorder.from_initial_state(np.zeros(7))
    noise = torch.zeros((1, 11, 80, 4))
    recorder.append_cycle(
        np.zeros(7),
        _observation(),
        noise,
        PlannerInferenceResult(prediction=noise),
        _step_info(),
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


def test_trace_recorder_rejects_misaligned_step_arrays() -> None:
    recorder = EpisodeTraceRecorder.from_initial_state(np.zeros(7))
    info = _step_info()
    info["trajectory_target_centers"] = np.zeros((4, 2))
    noise = torch.zeros((1, 11, 80, 4))

    with pytest.raises(RuntimeError, match="target centers"):
        recorder.append_cycle(
            np.zeros(7),
            _observation(),
            noise,
            PlannerInferenceResult(prediction=noise),
            info,
            0,
            None,
        )


def test_guided_trace_requires_and_persists_reference_action_targets_and_step_diagnostics() -> None:
    recorder = EpisodeTraceRecorder.from_initial_state(np.zeros(7))
    prediction = torch.zeros((1, 11, 80, 4))
    reference = torch.ones_like(prediction)
    steps = torch.arange(5, dtype=torch.float32)[None]
    diagnostics = GuidanceDiagnostics(
        lateral_target_offset_m=torch.tensor([2.5]),
        longitudinal_target_speed_fraction=torch.tensor([0.0]),
        longitudinal_target_speed_delta_mps=torch.zeros((1, 80)),
        lateral_objective_delta=steps,
        longitudinal_objective_delta=steps + 1.0,
        applied_gradient_l2=steps + 2.0,
        applied_gradient_max_abs=steps + 3.0,
        raw_neighbor_gradient_l2=steps + 4.0,
        zero_speed_count=torch.arange(5, dtype=torch.int64)[None],
    )
    result = PlannerInferenceResult(
        prediction=prediction,
        reference_prediction=reference,
        guidance_action=torch.tensor([[1.0, 0.0]]),
        guidance_diagnostics=diagnostics,
    )

    recorder.append_cycle(
        np.zeros(7),
        _observation(),
        torch.zeros_like(prediction),
        result,
        _step_info(),
        0,
        None,
    )
    arrays = recorder.finalize()

    assert arrays["reference_predictions_local"].shape == (1, 11, 80, 4)
    assert arrays["guidance_actions"].shape == (1, 2)
    assert arrays["guidance_lateral_objective_delta"].shape == (1, 5)
    assert arrays["guidance_zero_speed_count"].dtype.kind in "iu"
