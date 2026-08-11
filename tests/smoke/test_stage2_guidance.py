from __future__ import annotations

import pytest
import torch

from eco_planner.evaluation.runtime import FabricInferenceRuntime


@pytest.mark.slow
def test_stage2_guidance_reuses_real_ddim_reference_and_returns_finite_diagnostics(
    stage1_ddim_runtime: FabricInferenceRuntime,
    stage2_guided_runtime: FabricInferenceRuntime,
    stage0_observation: dict[str, torch.Tensor],
) -> None:
    baseline_generator = stage1_ddim_runtime.new_noise_generator()
    _, baseline_noise, baseline = stage1_ddim_runtime.infer(stage0_observation, baseline_generator)
    guided_generator = stage2_guided_runtime.new_noise_generator()
    _, guided_noise, guided = stage2_guided_runtime.infer(stage0_observation, guided_generator)

    assert torch.equal(guided_noise, baseline_noise)
    assert guided.reference_prediction is not None
    assert torch.equal(guided.reference_prediction, baseline.prediction)
    assert guided.guidance_action is not None
    assert torch.equal(guided.guidance_action, torch.tensor([[1.0, 0.0]]))
    assert guided.guidance_diagnostics is not None
    assert guided.guidance_diagnostics.applied_gradient_l2.shape == (1, 5)
    assert torch.isfinite(guided.guidance_diagnostics.applied_gradient_l2).all()
    assert torch.isfinite(guided.prediction).all()
    assert not torch.equal(guided.prediction, guided.reference_prediction)
