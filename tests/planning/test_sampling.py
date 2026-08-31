"""Sampler checks that protect finite, reproducible planner inputs and outputs."""

from __future__ import annotations

import pytest
import torch

from eco_planner.models import Ddim5SamplerConfig
from eco_planner.models.sampling import DiffusionSampler, _DdimSampler


def _config() -> Ddim5SamplerConfig:
    return Ddim5SamplerConfig(
        name="ddim5",
        num_steps=5,
        timesteps=(1.0, 0.8, 0.6, 0.4, 0.2, 0.0),
        initial_noise_scale=1.0,
        ddim_stochasticity=0.7,
        parity_label="plannerrft_paper_text",
    )


@pytest.mark.smoke
def test_seeded_ddim_sampling_returns_a_finite_constrained_sample() -> None:
    initial = torch.randn((2, 3, 12), generator=torch.Generator().manual_seed(11))

    def model(sample: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        return sample * 0.2 + timestep[:, None, None]

    def constrain(sample: torch.Tensor) -> torch.Tensor:
        constrained = sample.clone()
        constrained[:, :, :4] = 0.0
        return constrained

    sampler = DiffusionSampler(_config())
    first = sampler.sample(initial, model, constrain, torch.Generator().manual_seed(13))
    second = sampler.sample(initial, model, constrain, torch.Generator().manual_seed(13))

    assert first.shape == initial.shape
    assert torch.isfinite(first).all()
    assert torch.equal(first, second)
    assert torch.count_nonzero(first[:, :, :4]) == 0


def test_deterministic_ddim_step_omits_unused_variance_noise_without_consuming_rng() -> None:
    sampler = _DdimSampler()
    sample = torch.randn((1, 2, 12), generator=torch.Generator().manual_seed(17))
    prediction = sample * 0.25
    scheduler = sampler._new_scheduler(sample.device, sample.dtype)
    generator = torch.Generator().manual_seed(19)
    rng_state = generator.get_state().clone()

    actual = sampler._step(scheduler, sample, prediction, 999, 0, 0.0, generator, None)
    expected = scheduler.step(
        model_output=prediction,
        timestep=999,
        sample=sample,
        eta=0.0,
        variance_noise=torch.zeros_like(sample),
    ).prev_sample

    assert torch.equal(actual, expected)
    assert torch.equal(generator.get_state(), rng_state)
