from __future__ import annotations

import numpy as np
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

    prediction = result.prediction
    replay_prediction = replay_result.prediction
    assert ddim_runtime.sampler_report.name == "ddim5"
    assert ddim_runtime.sampler_report.num_steps == 5
    assert ddim_runtime.sampler_report.initial_noise_scale == 1.0
    assert tuple(prediction.shape) == (1, 11, 80, 4)
    assert prediction.dtype == np.float32
    assert np.isfinite(prediction).all()
    assert np.array_equal(result.initial_noise, replay_result.initial_noise)
    assert np.array_equal(prediction, replay_prediction)
