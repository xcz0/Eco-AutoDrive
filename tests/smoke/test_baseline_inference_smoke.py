from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from eco_planner.evaluation.runtime.engine import FabricInferenceRuntime


@pytest.mark.slow
def test_official_ema_checkpoint_cpu_smoke(
    baseline_runtime: FabricInferenceRuntime,
    baseline_observation: dict[str, torch.Tensor],
) -> None:
    generator = baseline_runtime.new_noise_generator()

    first_result = baseline_runtime.infer(baseline_observation, generator).audit_result()
    replay_generator = baseline_runtime.new_noise_generator()
    second_result = baseline_runtime.infer(baseline_observation, replay_generator).audit_result()
    first = first_result.prediction
    second = second_result.prediction

    report = baseline_runtime.checkpoint_report
    assert report.ema_tensor_count == 276
    assert report.parameter_count == 6_042_628
    assert tuple(first.shape) == (1, 11, 80, 4)
    assert np.isfinite(first).all()
    assert np.array_equal(first_result.initial_noise, second_result.initial_noise)
    assert np.array_equal(first, second)

    payload = {
        "ema_tensor_count": report.ema_tensor_count,
        "parameter_count": report.parameter_count,
        "prediction_shape": list(first.shape),
        "runtime_device": baseline_runtime.report.device,
        "precision": baseline_runtime.report.resolved_precision,
        "seed": 0,
        "status": "pass",
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
