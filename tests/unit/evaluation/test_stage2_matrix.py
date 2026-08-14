from __future__ import annotations

import numpy as np
import pytest

from eco_planner.evaluation.stage2_matrix import (
    _validate_expected_seeds,
    _validate_runtime,
    measure_first_cycle_guidance_trend,
)


def test_first_cycle_guidance_trend_uses_reference_left_normal_and_ten_hz_speed() -> None:
    reference = np.zeros((1, 11, 80, 4), dtype=np.float32)
    reference[0, 0, :, 0] = np.arange(1, 81, dtype=np.float32)
    reference[0, 0, :, 2] = 1.0
    prediction = reference.copy()
    prediction[0, 0, :, 1] += 0.75
    prediction[0, 0, :, 0] *= 1.25
    arrays = {
        "reference_predictions_local": reference,
        "predictions_local": prediction,
        "observation_ego_current_state": np.zeros((1, 10), dtype=np.float32),
    }

    trend = measure_first_cycle_guidance_trend(arrays)

    assert trend.mean_lateral_offset_m == 0.75
    assert trend.mean_longitudinal_speed_delta_mps == 2.5


def test_stage2_matrix_accepts_an_explicit_three_seed_cuda_bf16_contract() -> None:
    assert _validate_expected_seeds((0, 1, 2)) == frozenset({0, 1, 2})
    _validate_runtime(
        {
            "requested_accelerator": "cuda",
            "resolved_accelerator": "cuda",
            "requested_precision": "bf16-mixed",
            "resolved_precision": "bf16-mixed",
        },
        expected_accelerator="cuda",
        expected_precision="bf16-mixed",
    )


@pytest.mark.parametrize("seeds", [(), (0, 0), (-1, 0)])
def test_stage2_matrix_rejects_invalid_explicit_seeds(seeds: tuple[int, ...]) -> None:
    with pytest.raises((TypeError, ValueError), match="expected seeds"):
        _validate_expected_seeds(seeds)


def test_stage2_matrix_rejects_wrong_resolved_runtime() -> None:
    with pytest.raises(ValueError, match="resolved_precision"):
        _validate_runtime(
            {
                "requested_accelerator": "cuda",
                "resolved_accelerator": "cuda",
                "requested_precision": "bf16-mixed",
                "resolved_precision": "32-true",
            },
            expected_accelerator="cuda",
            expected_precision="bf16-mixed",
        )
