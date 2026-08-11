from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from eco_planner.models.ddim_sampler import DdimSampler
from eco_planner.models.vp_schedule import LinearVpSchedule


def _identity(value: torch.Tensor) -> torch.Tensor:
    return value.clone()


def test_ddim_matches_hand_calculated_stochastic_transition() -> None:
    sampler = DdimSampler()
    schedule = LinearVpSchedule()
    initial = torch.tensor([[[0.25, -0.75]]], dtype=torch.float64)
    timesteps = torch.tensor([1.0, 0.5, 0.0], dtype=torch.float64)
    x_start = torch.tensor([[[1.0, 2.0]]], dtype=torch.float64)

    def model(sample: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        return x_start if float(timestep[0]) == 1.0 else sample.clone()

    generator = torch.Generator().manual_seed(13)
    result = sampler.sample(initial, model, _identity, timesteps, 2, 1.0, generator)

    replay = torch.Generator().manual_seed(13)
    random_noise = torch.randn(initial.shape, dtype=initial.dtype, generator=replay)
    alpha_t = schedule.alpha(timesteps[0])
    sigma_t = schedule.sigma(timesteps[0])
    alpha_s = schedule.alpha(timesteps[1])
    sigma_s = schedule.sigma(timesteps[1])
    transition_sigma = sigma_s / sigma_t * torch.sqrt(1.0 - (alpha_t / alpha_s) ** 2)
    direction_scale = torch.sqrt(sigma_s**2 - transition_sigma**2)
    predicted_noise = (initial - alpha_t * x_start) / sigma_t
    expected = alpha_s * x_start + direction_scale * predicted_noise
    expected += transition_sigma * random_noise

    torch.testing.assert_close(result, expected, rtol=0.0, atol=1e-12)


def test_deterministic_ddim_does_not_consume_generator() -> None:
    sampler = DdimSampler()
    initial = torch.ones((1, 2, 4))
    timesteps = torch.tensor([1.0, 0.5, 0.0])
    generator = torch.Generator().manual_seed(7)
    state_before = generator.get_state().clone()

    first = sampler.sample(initial, _identity_model, _identity, timesteps, 2, 0.0, generator)
    second = sampler.sample(initial, _identity_model, _identity, timesteps, 2, 0.0, None)

    assert torch.equal(first, second)
    assert torch.equal(generator.get_state(), state_before)


def test_stochastic_ddim_replays_seed_and_changes_across_seeds() -> None:
    sampler = DdimSampler()
    initial = torch.ones((1, 2, 4))
    timesteps = torch.tensor([1.0, 0.5, 0.0])

    first = sampler.sample(
        initial,
        _identity_model,
        _identity,
        timesteps,
        2,
        1.0,
        torch.Generator().manual_seed(7),
    )
    replay = sampler.sample(
        initial,
        _identity_model,
        _identity,
        timesteps,
        2,
        1.0,
        torch.Generator().manual_seed(7),
    )
    different = sampler.sample(
        initial,
        _identity_model,
        _identity,
        timesteps,
        2,
        1.0,
        torch.Generator().manual_seed(8),
    )

    assert torch.equal(first, replay)
    assert not torch.equal(first, different)


def test_five_step_ddim_calls_denoiser_five_times_and_constraint_six_times() -> None:
    calls = {"model": 0, "constraint": 0}
    initial = torch.ones((2, 3, 4))
    timesteps = torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2, 0.0])

    def model(sample: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        calls["model"] += 1
        assert timestep.shape == (2,)
        return sample.clone()

    def constraint(sample: torch.Tensor) -> torch.Tensor:
        calls["constraint"] += 1
        result = sample.clone()
        result[:, :, 0] = 3.0
        return result

    result = DdimSampler().sample(initial, model, constraint, timesteps, 5, 0.0, None)

    assert calls == {"model": 5, "constraint": 6}
    assert torch.equal(result[:, :, 0], torch.full((2, 3), 3.0))


@pytest.mark.parametrize(
    ("mutate", "error", "message"),
    [
        (
            lambda initial, times: (initial.reshape(1, -1), times, 5, 0.0, None),
            ValueError,
            "shape",
        ),
        (lambda initial, times: (initial, times[:-1], 5, 0.0, None), ValueError, "shape"),
        (lambda initial, times: (initial, times, 0, 0.0, None), ValueError, "positive"),
        (
            lambda initial, times: (
                initial,
                times.clone().index_fill(0, torch.tensor([0]), 0.9),
                5,
                0.0,
                None,
            ),
            ValueError,
            "start",
        ),
        (lambda initial, times: (initial, times, 5, -0.1, None), ValueError, r"\[0, 1\]"),
        (lambda initial, times: (initial, times, 5, 1.0, None), ValueError, "Generator"),
    ],
)
def test_ddim_rejects_invalid_inputs(
    mutate: Callable[
        [torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor, int, float, torch.Generator | None],
    ],
    error: type[Exception],
    message: str,
) -> None:
    initial = torch.ones((1, 2, 4))
    times = torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2, 0.0])
    sample, schedule, num_steps, stochasticity, generator = mutate(initial, times)
    with pytest.raises(error, match=message):
        DdimSampler().sample(
            sample,
            _identity_model,
            _identity,
            schedule,
            num_steps,
            stochasticity,
            generator,
        )


def test_ddim_rejects_non_finite_and_callback_contract_violations() -> None:
    initial = torch.ones((1, 2, 4))
    timesteps = torch.tensor([1.0, 0.0])
    bad_initial = initial.clone()
    bad_initial[0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        DdimSampler().sample(bad_initial, _identity_model, _identity, timesteps, 1, 0.0, None)
    with pytest.raises(ValueError, match="preserve sample shape"):
        DdimSampler().sample(
            initial,
            lambda sample, timestep: sample[..., :1],
            _identity,
            timesteps,
            1,
            0.0,
            None,
        )
    with pytest.raises(TypeError, match="preserve sample dtype"):
        DdimSampler().sample(
            initial,
            lambda sample, timestep: sample.double(),
            _identity,
            timesteps,
            1,
            0.0,
            None,
        )


def test_ddim_rejects_timestep_dtype_and_device_mismatches() -> None:
    initial = torch.ones((1, 2, 4), dtype=torch.float32)
    with pytest.raises(TypeError, match="initial_sample dtype"):
        DdimSampler().sample(
            initial,
            _identity_model,
            _identity,
            torch.tensor([1.0, 0.0], dtype=torch.float64),
            1,
            0.0,
            None,
        )
    with pytest.raises(ValueError, match="initial_sample device"):
        DdimSampler().sample(
            initial,
            _identity_model,
            _identity,
            torch.empty((2,), device="meta"),
            1,
            0.0,
            None,
        )


@pytest.mark.gpu
def test_ddim_rejects_generator_on_different_device() -> None:
    initial = torch.ones((1, 2, 4), device="cuda")
    timesteps = torch.tensor([1.0, 0.0], device="cuda")

    with pytest.raises(ValueError, match="generator"):
        DdimSampler().sample(
            initial,
            _identity_model,
            _identity,
            timesteps,
            1,
            1.0,
            torch.Generator(device="cpu"),
        )


def _identity_model(sample: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
    return sample.clone()
