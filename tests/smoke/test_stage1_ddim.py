from __future__ import annotations

import pytest
import torch

from eco_planner.evaluation.runtime import FabricInferenceRuntime


@pytest.mark.slow
def test_ddim5_official_ema_cpu_is_finite_and_replayable(
    stage1_ddim_runtime: FabricInferenceRuntime,
    stage0_observation: dict[str, torch.Tensor],
) -> None:
    generator = stage1_ddim_runtime.new_noise_generator()
    _, noise, prediction = stage1_ddim_runtime.infer(stage0_observation, generator)
    replay_generator = stage1_ddim_runtime.new_noise_generator()
    _, replay_noise, replay_prediction = stage1_ddim_runtime.infer(
        stage0_observation,
        replay_generator,
    )

    assert stage1_ddim_runtime.sampler_report.name == "ddim5"
    assert stage1_ddim_runtime.sampler_report.num_steps == 5
    assert stage1_ddim_runtime.sampler_report.initial_noise_scale == 1.0
    assert tuple(prediction.shape) == (1, 11, 80, 4)
    assert prediction.dtype == torch.float32
    assert torch.isfinite(prediction).all()
    assert torch.equal(noise, replay_noise)
    assert torch.equal(prediction, replay_prediction)
