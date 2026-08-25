from __future__ import annotations

import pytest
import torch

from eco_planner.evaluation.runtime import FabricInferenceRuntime


@pytest.mark.slow
def test_guidance_reuses_real_ddim_reference_and_returns_finite_diagnostics(
    ddim_runtime: FabricInferenceRuntime,
    guided_runtime: FabricInferenceRuntime,
    baseline_observation: dict[str, torch.Tensor],
) -> None:
    baseline_generator = ddim_runtime.new_noise_generator()
    baseline = ddim_runtime.infer(baseline_observation, baseline_generator).audit_result()
    guided_generator = guided_runtime.new_noise_generator()
    guided = guided_runtime.infer(baseline_observation, guided_generator).audit_result()

    assert torch.equal(guided["initial_noise"], baseline["initial_noise"])
    assert torch.equal(guided["reference_prediction"], baseline["prediction"])
    torch.testing.assert_close(guided["guidance_action"], torch.tensor([[1.0, 0.0]]))
    assert guided["applied_gradient_l2"].shape == (1, 5)
    assert torch.isfinite(guided["applied_gradient_l2"]).all()
    assert torch.isfinite(guided["prediction"]).all()
    assert not torch.equal(guided["prediction"], guided["reference_prediction"])
