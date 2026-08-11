from __future__ import annotations

import json

import pytest
import torch

from eco_planner.models.pretrained import CheckpointLoadReport, PretrainedDiffusionPlanner


@pytest.mark.slow
def test_official_ema_checkpoint_cpu_smoke(
    stage0_planner: tuple[PretrainedDiffusionPlanner, CheckpointLoadReport],
    stage0_observation: dict[str, torch.Tensor],
) -> None:
    planner, report = stage0_planner
    device = torch.device("cpu")
    generator = torch.Generator(device=device).manual_seed(0)
    noise = torch.randn((1, 11, 80, 4), dtype=torch.float32, device=device, generator=generator)

    first = planner.predict(stage0_observation, noise)
    second = planner.predict(stage0_observation, noise)

    assert report.ema_tensor_count == 276
    assert report.parameter_count == 6_042_628
    assert tuple(first.shape) == (1, 11, 80, 4)
    assert torch.isfinite(first).all()
    assert torch.equal(first, second)

    prediction_sum = float(first.sum().item())
    assert prediction_sum == 35847.8671875
    payload = {
        "ema_tensor_count": report.ema_tensor_count,
        "parameter_count": report.parameter_count,
        "prediction_shape": list(first.shape),
        "prediction_sum": prediction_sum,
        "runtime_device": report.runtime_device,
        "seed": 0,
        "status": "pass",
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
