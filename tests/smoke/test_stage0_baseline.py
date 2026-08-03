from __future__ import annotations

import hashlib
import json

import pytest
import torch

from eco_planner.models.pretrained import CheckpointLoadReport, PretrainedDiffusionPlanner
from eco_planner.models.synthetic import make_stage0_observation


@pytest.mark.slow
def test_official_ema_checkpoint_cpu_smoke(
    stage0_planner: tuple[PretrainedDiffusionPlanner, CheckpointLoadReport],
) -> None:
    planner, report = stage0_planner
    device = torch.device("cpu")
    generator = torch.Generator(device=device).manual_seed(0)
    observation = make_stage0_observation(device)
    noise = torch.randn((1, 11, 80, 4), dtype=torch.float32, device=device, generator=generator)

    first = planner.predict(observation, noise)
    second = planner.predict(observation, noise)

    assert report.ema_tensor_count == 276
    assert report.parameter_count == 6_042_628
    assert tuple(first.shape) == (1, 11, 80, 4)
    assert torch.isfinite(first).all()
    assert torch.equal(first, second)

    output_bytes = first.detach().contiguous().cpu().numpy().tobytes()
    payload = {
        "args_sha256": report.args_sha256,
        "checkpoint_sha256": report.checkpoint_sha256,
        "ema_tensor_count": report.ema_tensor_count,
        "parameter_count": report.parameter_count,
        "prediction_sha256": hashlib.sha256(output_bytes).hexdigest(),
        "prediction_shape": list(first.shape),
        "prediction_sum": float(first.sum().item()),
        "runtime_device": report.runtime_device,
        "seed": 0,
        "status": "pass",
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
