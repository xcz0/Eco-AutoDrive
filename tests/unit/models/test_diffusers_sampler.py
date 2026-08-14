from __future__ import annotations

import pytest
import torch

from eco_planner.models.guidance import GuidanceGradientResult
from eco_planner.models.sampling.backends.diffusers import DiffusersDdimSampler, DiffusersDpmSampler
from eco_planner.models.sampling.backends.vp_schedule import LinearVpSchedule
from eco_planner.models.sampling.config import Ddim5SamplerConfig
from eco_planner.models.sampling.planner import PlanningSampler


def _constraint(sample: torch.Tensor) -> torch.Tensor:
    result = sample.clone()
    result[:, :, 0] = 1.0
    return result


def _model(sample: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
    return sample * 0.2 + timestep[:, None, None]


def test_diffusers_trained_betas_match_the_pinned_continuous_vp_schedule() -> None:
    adapter = DiffusersDdimSampler()
    scheduler = adapter._new_scheduler(torch.device("cpu"), torch.float64)
    schedule = LinearVpSchedule()
    continuous_timesteps = torch.arange(1, 1001, dtype=torch.float64) / 1000
    expected = schedule.alpha(continuous_timesteps).square()

    torch.testing.assert_close(scheduler.alphas_cumprod, expected, rtol=0.0, atol=1e-12)


def test_diffusers_ddim_uses_the_certified_discrete_and_continuous_timesteps() -> None:
    adapter = DiffusersDdimSampler()
    scheduler = adapter._new_scheduler(torch.device("cpu"))
    model_timesteps: list[torch.Tensor] = []

    def record_model(sample: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        model_timesteps.append(timestep.clone())
        return sample.clone()

    result = adapter.sample(
        torch.ones((2, 3, 4)),
        record_model,
        _constraint,
        torch.tensor((1.0, 0.8, 0.6, 0.4, 0.2, 0.0)),
        5,
        0.0,
        None,
    )

    assert tuple(scheduler.timesteps.tolist()) == (999, 799, 599, 399, 199)
    torch.testing.assert_close(
        torch.stack(model_timesteps)[:, 0],
        torch.tensor((1.0, 0.8, 0.6, 0.4, 0.2)),
        rtol=0.0,
        atol=0.0,
    )
    assert torch.equal(result[:, :, 0], torch.ones((2, 3)))


def test_diffusers_guided_ddim_reuses_explicit_variance_noises_without_rng_side_effects() -> None:
    sampler = DiffusersDdimSampler()
    initial = torch.ones((1, 2, 8), dtype=torch.float64)
    timesteps = torch.tensor((1.0, 0.8, 0.6, 0.4, 0.2, 0.0), dtype=torch.float64)
    draw_generator = torch.Generator().manual_seed(31)
    variance_noises = (
        *(
            torch.randn(initial.shape, dtype=initial.dtype, generator=draw_generator)
            for _ in range(4)
        ),
        torch.zeros_like(initial),
    )
    reference_generator = torch.Generator().manual_seed(7)
    guided_generator = torch.Generator().manual_seed(11)
    reference_state = reference_generator.get_state().clone()
    guided_state = guided_generator.get_state().clone()

    def guidance(sample: torch.Tensor, prediction: torch.Tensor) -> GuidanceGradientResult:
        zeros = torch.zeros((sample.shape[0],), dtype=sample.dtype, device=sample.device)
        return GuidanceGradientResult(
            applied_gradient=torch.zeros_like(sample),
            lateral_objective_delta=zeros,
            longitudinal_objective_delta=zeros,
            applied_gradient_l2=zeros,
            applied_gradient_max_abs=zeros,
            raw_neighbor_gradient_l2=zeros,
            zero_speed_count=torch.zeros_like(zeros, dtype=torch.int64),
        )

    reference = sampler.sample(
        initial,
        _model,
        _constraint,
        timesteps,
        5,
        0.7,
        reference_generator,
        variance_noises=variance_noises,
    )
    guided = sampler.sample_guided(
        initial,
        _model,
        _constraint,
        timesteps,
        5,
        0.7,
        guided_generator,
        guidance,
        gradient_step_coefficient=1.0,
        variance_noises=variance_noises,
    )

    torch.testing.assert_close(guided.sample, reference, rtol=0.0, atol=1e-12)
    assert torch.equal(reference_generator.get_state(), reference_state)
    assert torch.equal(guided_generator.get_state(), guided_state)


def test_diffusers_ddim_retains_sampler_boundary_validation() -> None:
    sampler = DiffusersDdimSampler()
    initial = torch.ones((1, 2, 4))
    timesteps = torch.tensor((1.0, 0.8, 0.6, 0.4, 0.2, 0.0))

    with pytest.raises(ValueError, match="preserve sample shape"):
        sampler.sample(
            initial,
            lambda sample, timestep: sample[..., :1],
            _constraint,
            timesteps,
            5,
            0.0,
            None,
        )


def test_diffusers_dpm_uses_the_certified_profile_and_call_contract() -> None:
    adapter = DiffusersDpmSampler()
    scheduler = adapter._new_scheduler(torch.device("cpu"), torch.float64)
    schedule = LinearVpSchedule()
    continuous_timesteps = torch.arange(1, 1001, dtype=torch.float64) / 1000
    expected_alpha_bar = schedule.alpha(continuous_timesteps).square()
    model_timesteps: list[torch.Tensor] = []

    def record_model(sample: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        model_timesteps.append(timestep.clone())
        return torch.zeros_like(sample)

    constraint_calls = 0

    def counted_constraint(sample: torch.Tensor) -> torch.Tensor:
        nonlocal constraint_calls
        constraint_calls += 1
        return _constraint(sample)

    result = adapter.sample(
        torch.zeros((2, 3, 4), dtype=torch.float64), record_model, counted_constraint
    )

    torch.testing.assert_close(scheduler.alphas_cumprod, expected_alpha_bar, rtol=0.0, atol=1e-12)
    assert scheduler.config.prediction_type == "sample"
    assert scheduler.config.algorithm_type == "dpmsolver++"
    assert scheduler.config.solver_order == 2
    assert scheduler.config.solver_type == "midpoint"
    assert scheduler.config.use_lu_lambdas
    assert not scheduler.config.lower_order_final
    assert scheduler.config.final_sigmas_type == "sigma_min"
    assert len(model_timesteps) == 11
    lambda_start = schedule.lambda_(torch.tensor(1.0, dtype=torch.float64))
    lambda_end = schedule.lambda_(torch.tensor(0.001, dtype=torch.float64))
    expected_model_times = schedule.inverse_lambda(
        torch.linspace(lambda_start, lambda_end, 11, dtype=torch.float64)
    )
    torch.testing.assert_close(
        torch.stack(model_timesteps)[:, 0], expected_model_times, rtol=0.0, atol=1e-6
    )
    torch.testing.assert_close(
        model_timesteps[-1],
        torch.full((2,), 0.001, dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )
    assert constraint_calls == 12
    assert torch.equal(result[:, :, 0], torch.ones((2, 3), dtype=torch.float64))


def test_planning_sampler_rejects_invalid_static_ddim_profile_at_construction() -> None:
    config = Ddim5SamplerConfig(
        name="ddim5",
        num_steps=5,
        timesteps=(1.0, 0.7, 0.6, 0.4, 0.2, 0.0),
        initial_noise_scale=1.0,
        ddim_stochasticity=0.0,
        parity_label="plannerrft_paper_text",
    )

    with pytest.raises(ValueError, match="fixed five-step"):
        PlanningSampler(config)
