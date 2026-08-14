from __future__ import annotations

import numpy as np
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
    baseline = stage1_ddim_runtime.infer(stage0_observation, baseline_generator)
    guided_generator = stage2_guided_runtime.new_noise_generator()
    guided = stage2_guided_runtime.infer(stage0_observation, guided_generator)

    assert np.array_equal(guided.initial_noise, baseline.initial_noise)
    assert guided.reference_prediction is not None
    assert np.array_equal(guided.reference_prediction, baseline.prediction)
    assert guided.guidance_action is not None
    assert np.array_equal(guided.guidance_action, np.array([[1.0, 0.0]], dtype=np.float32))
    assert guided.guidance_diagnostics is not None
    assert guided.guidance_diagnostics.applied_gradient_l2.shape == (1, 5)
    assert np.isfinite(guided.guidance_diagnostics.applied_gradient_l2).all()
    assert np.isfinite(guided.prediction).all()
    assert not np.array_equal(guided.prediction, guided.reference_prediction)
