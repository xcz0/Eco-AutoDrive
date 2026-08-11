from __future__ import annotations

import json

import pytest
import torch

from eco_planner.evaluation.runtime import FabricInferenceRuntime


@pytest.mark.slow
def test_official_ema_checkpoint_cpu_smoke(
    stage0_runtime: FabricInferenceRuntime,
    stage0_observation: dict[str, torch.Tensor],
) -> None:
    generator = stage0_runtime.new_noise_generator()

    _, noise, first_result = stage0_runtime.infer(stage0_observation, generator)
    replay_generator = stage0_runtime.new_noise_generator()
    _, replay_noise, second_result = stage0_runtime.infer(stage0_observation, replay_generator)
    first = first_result.prediction
    second = second_result.prediction

    report = stage0_runtime.checkpoint_report
    assert report.ema_tensor_count == 276
    assert report.parameter_count == 6_042_628
    assert tuple(first.shape) == (1, 11, 80, 4)
    assert torch.isfinite(first).all()
    assert torch.equal(noise, replay_noise)
    assert torch.equal(first, second)

    prediction_sum = float(first.sum().item())
    assert prediction_sum == 35847.875
    payload = {
        "ema_tensor_count": report.ema_tensor_count,
        "parameter_count": report.parameter_count,
        "prediction_shape": list(first.shape),
        "prediction_sum": prediction_sum,
        "runtime_device": stage0_runtime.report.device,
        "precision": stage0_runtime.report.resolved_precision,
        "seed": 0,
        "status": "pass",
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
