from __future__ import annotations

import numpy as np

from eco_planner.evaluation.stage2_matrix import measure_first_cycle_guidance_trend


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
