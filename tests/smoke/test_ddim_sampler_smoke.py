from __future__ import annotations

import pytest
import torch

from eco_planner.evaluation.runtime.engine import FabricInferenceRuntime


@pytest.mark.slow
def test_ddim5_official_ema_cpu_is_finite_and_replayable(
    ddim_runtime: FabricInferenceRuntime,
    baseline_observation: dict[str, torch.Tensor],
) -> None:
    generator = ddim_runtime.new_noise_generator()
    result = ddim_runtime.infer(baseline_observation, generator)
    replay_generator = ddim_runtime.new_noise_generator()
    replay_result = ddim_runtime.infer(
        baseline_observation,
        replay_generator,
    )

    audit = result.audit_result()
    replay_audit = replay_result.audit_result()
    prediction = audit["prediction"]
    replay_prediction = replay_audit["prediction"]
    assert ddim_runtime.sampler_report.name == "ddim5"
    assert ddim_runtime.sampler_report.num_steps == 5
    assert ddim_runtime.sampler_report.initial_noise_scale == 1.0
    assert tuple(prediction.shape) == (1, 11, 80, 4)
    assert prediction.dtype == torch.float32
    assert torch.isfinite(prediction).all()
    assert torch.equal(audit["initial_noise"], replay_audit["initial_noise"])
    assert torch.equal(prediction, replay_prediction)
