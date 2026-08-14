"""Diffusers-backed fixed diffusion sampling profiles."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
from diffusers import DDIMScheduler, DPMSolverMultistepScheduler

from eco_planner.models.config import Ddim5SamplerConfig, SamplerConfig
from eco_planner.models.guidance import GuidanceGradientResult


@dataclass(frozen=True)
class DdimGuidedSampleResult:
    """Final normalized DDIM sample and one diagnostic record per transition."""

    sample: torch.Tensor
    diagnostics: tuple[GuidanceGradientResult, ...]


class LinearVpSchedule:
    """The pinned beta schedule used by the official Diffusion Planner."""

    def __init__(self, beta_0: float = 0.1, beta_1: float = 20.0) -> None:
        self.total_n = 1000
        self.beta_0 = beta_0
        self.beta_1 = beta_1

    def log_alpha(self, timestep: torch.Tensor) -> torch.Tensor:
        beta_range = self.beta_1 - self.beta_0
        return -0.25 * timestep.square() * beta_range - 0.5 * timestep * self.beta_0

    def alpha(self, timestep: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.log_alpha(timestep))

    def sigma(self, timestep: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(1.0 - torch.exp(2.0 * self.log_alpha(timestep)))

    def lambda_(self, timestep: torch.Tensor) -> torch.Tensor:
        log_alpha = self.log_alpha(timestep)
        return log_alpha - 0.5 * torch.log(1.0 - torch.exp(2.0 * log_alpha))

    def inverse_lambda(self, value: torch.Tensor) -> torch.Tensor:
        temporary = (
            2.0
            * (self.beta_1 - self.beta_0)
            * torch.logaddexp(-2.0 * value, torch.zeros((1,), device=value.device))
        )
        delta = self.beta_0**2 + temporary
        return temporary / (torch.sqrt(delta) + self.beta_0) / (self.beta_1 - self.beta_0)


_NUM_TRAIN_TIMESTEPS = 1000
_DPM10_NUM_STEPS = 10
_DDIM5_TIMESTEPS = (1.0, 0.8, 0.6, 0.4, 0.2, 0.0)


def build_vp_trained_betas(schedule: LinearVpSchedule) -> np.ndarray:
    """Discretize the pinned VP-SDE as Diffusers ``trained_betas``."""

    timesteps = np.arange(1, schedule.total_n + 1, dtype=np.float64) / schedule.total_n
    log_alpha_bar = (
        -0.5 * (schedule.beta_1 - schedule.beta_0) * timesteps**2 - schedule.beta_0 * timesteps
    )
    alpha_bar = np.exp(log_alpha_bar)
    previous_alpha_bar = np.concatenate((np.ones(1, dtype=np.float64), alpha_bar[:-1]))
    return 1.0 - alpha_bar / previous_alpha_bar


class _DdimSampler:
    """Apply the validated DDIM-5 profile through ``diffusers.DDIMScheduler``."""

    def __init__(self) -> None:
        self._schedule = LinearVpSchedule()
        self._trained_betas = build_vp_trained_betas(self._schedule)

    def sample(
        self,
        initial_sample: torch.Tensor,
        model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        current_state_constraint: Callable[[torch.Tensor], torch.Tensor],
        timesteps: torch.Tensor,
        num_steps: int,
        ddim_stochasticity: float,
        generator: torch.Generator | None,
        *,
        variance_noises: tuple[torch.Tensor, ...] | None = None,
    ) -> torch.Tensor:
        """Sample the fixed DDIM-5 profile while retaining Planner constraints."""

        scheduler = self._new_scheduler(initial_sample.device, initial_sample.dtype)
        sample = current_state_constraint(initial_sample)
        for index in range(len(scheduler.timesteps)):
            discrete_timestep = int(_DDIM5_TIMESTEPS[index] * _NUM_TRAIN_TIMESTEPS) - 1
            prediction = self._prediction(model, sample, discrete_timestep)
            sample = self._step(
                scheduler,
                sample,
                prediction,
                discrete_timestep,
                index,
                ddim_stochasticity,
                generator,
                None if variance_noises is None else variance_noises[index],
            )
            sample = current_state_constraint(sample)
        return sample

    def sample_guided(
        self,
        initial_sample: torch.Tensor,
        model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        current_state_constraint: Callable[[torch.Tensor], torch.Tensor],
        timesteps: torch.Tensor,
        num_steps: int,
        ddim_stochasticity: float,
        generator: torch.Generator | None,
        guidance: Callable[[torch.Tensor, torch.Tensor], GuidanceGradientResult],
        *,
        gradient_step_coefficient: float,
        variance_noises: tuple[torch.Tensor, ...] | None = None,
    ) -> DdimGuidedSampleResult:
        """Sample DDIM-5 and apply the project's post-transition guidance policy."""

        scheduler = self._new_scheduler(initial_sample.device, initial_sample.dtype)
        sample = current_state_constraint(initial_sample)
        diagnostics: list[GuidanceGradientResult] = []
        for index in range(len(scheduler.timesteps)):
            discrete_timestep = int(_DDIM5_TIMESTEPS[index] * _NUM_TRAIN_TIMESTEPS) - 1
            sample_with_grad = sample.detach().requires_grad_(True)
            prediction = self._prediction(model, sample_with_grad, discrete_timestep)
            step = guidance(sample_with_grad, prediction)
            transitioned = self._step(
                scheduler,
                sample_with_grad.detach(),
                prediction.detach(),
                discrete_timestep,
                index,
                ddim_stochasticity,
                generator,
                None if variance_noises is None else variance_noises[index],
            )
            updated = transitioned - gradient_step_coefficient * step.applied_gradient
            sample = current_state_constraint(updated).detach()
            diagnostics.append(step)
        return DdimGuidedSampleResult(sample=sample, diagnostics=tuple(diagnostics))

    def _new_scheduler(
        self, device: torch.device, dtype: torch.dtype = torch.float32
    ) -> DDIMScheduler:
        scheduler = DDIMScheduler(
            num_train_timesteps=_NUM_TRAIN_TIMESTEPS,
            trained_betas=self._trained_betas,
            prediction_type="sample",
            clip_sample=False,
            set_alpha_to_one=True,
            timestep_spacing="trailing",
        )
        if dtype == torch.float64:
            scheduler.betas = torch.from_numpy(self._trained_betas).to(dtype=dtype)
            scheduler.alphas = 1.0 - scheduler.betas
            scheduler.alphas_cumprod = torch.cumprod(scheduler.alphas, dim=0)
            scheduler.final_alpha_cumprod = torch.tensor(1.0, dtype=dtype)
        scheduler.set_timesteps(num_inference_steps=5, device=device)
        return scheduler

    def _prediction(
        self,
        model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        sample: torch.Tensor,
        discrete_timestep: int,
    ) -> torch.Tensor:
        continuous_timestep = (discrete_timestep + 1) / _NUM_TRAIN_TIMESTEPS
        batch_timestep = torch.full(
            (sample.shape[0],),
            continuous_timestep,
            dtype=sample.dtype,
            device=sample.device,
        )
        return model(sample, batch_timestep)

    @staticmethod
    def _step(
        scheduler: DDIMScheduler,
        sample: torch.Tensor,
        prediction: torch.Tensor,
        discrete_timestep: int,
        index: int,
        stochasticity: float,
        generator: torch.Generator | None,
        variance_noise: torch.Tensor | None,
    ) -> torch.Tensor:
        if variance_noise is not None:
            random_noise = variance_noise
        elif stochasticity > 0.0 and index < scheduler.num_inference_steps - 1:
            random_noise = torch.randn(
                sample.shape,
                dtype=sample.dtype,
                device=sample.device,
                generator=generator,
            )
        else:
            random_noise = torch.zeros_like(sample)
        result = scheduler.step(
            model_output=prediction,
            timestep=discrete_timestep,
            sample=sample,
            eta=stochasticity,
            variance_noise=random_noise,
        ).prev_sample
        return result


class _DpmSampler:
    """Apply the fixed DPM-Solver++ ten-step profile through Diffusers."""

    def __init__(self) -> None:
        self._schedule = LinearVpSchedule()
        self._trained_betas = build_vp_trained_betas(self._schedule)

    def sample(
        self,
        initial_sample: torch.Tensor,
        model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        current_state_constraint: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Sample the certified DPM10 profile and retain its final ``x_start`` call."""

        scheduler = self._new_scheduler(initial_sample.device, initial_sample.dtype)
        with torch.no_grad():
            sample = initial_sample
            for index, scheduler_timestep in enumerate(scheduler.timesteps[:_DPM10_NUM_STEPS]):
                prediction = self._prediction(model, scheduler, sample, index)
                if index == 0:
                    sample = current_state_constraint(sample)
                sample = scheduler.step(
                    model_output=prediction,
                    timestep=scheduler_timestep,
                    sample=sample,
                ).prev_sample
                sample = current_state_constraint(sample)
            final_timestep = torch.full(
                (sample.shape[0],),
                1.0 / _NUM_TRAIN_TIMESTEPS,
                dtype=sample.dtype,
                device=sample.device,
            )
            final_prediction = model(sample, final_timestep)
            return current_state_constraint(final_prediction)

    def _new_scheduler(
        self, device: torch.device, dtype: torch.dtype = torch.float32
    ) -> DPMSolverMultistepScheduler:
        scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=_NUM_TRAIN_TIMESTEPS,
            trained_betas=self._trained_betas,
            prediction_type="sample",
            algorithm_type="dpmsolver++",
            solver_order=2,
            solver_type="midpoint",
            thresholding=False,
            use_lu_lambdas=True,
            lower_order_final=False,
            final_sigmas_type="sigma_min",
        )
        if dtype == torch.float64:
            scheduler.betas = torch.from_numpy(self._trained_betas).to(dtype=dtype)
            scheduler.alphas = 1.0 - scheduler.betas
            scheduler.alphas_cumprod = torch.cumprod(scheduler.alphas, dim=0)
            scheduler.alpha_t = torch.sqrt(scheduler.alphas_cumprod)
            scheduler.sigma_t = torch.sqrt(1.0 - scheduler.alphas_cumprod)
            scheduler.lambda_t = torch.log(scheduler.alpha_t) - torch.log(scheduler.sigma_t)
        # Diffusers includes both endpoints in its Lu schedule and then appends the final sigma.
        # Request one extra scheduler point so the ten executed transitions end at sigma_min rather
        # than duplicating it in the final transition.
        scheduler.set_timesteps(num_inference_steps=_DPM10_NUM_STEPS + 1, device=device)
        return scheduler

    def _prediction(
        self,
        model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        scheduler: DPMSolverMultistepScheduler,
        sample: torch.Tensor,
        index: int,
    ) -> torch.Tensor:
        sigma_ratio = scheduler.sigmas[index].to(device=sample.device, dtype=sample.dtype)
        lambda_timestep = -torch.log(sigma_ratio)
        continuous_timestep = self._schedule.inverse_lambda(lambda_timestep)
        batch_timestep = continuous_timestep.expand(sample.shape[0])
        model_input = scheduler.scale_model_input(sample)
        return model(model_input, batch_timestep)


@dataclass(frozen=True)
class GuidanceSamplingRandomness:
    """Transition randomness shared by the reference and guided DDIM passes."""

    variance_noises: tuple[torch.Tensor, ...] | None = None


class DiffusionSampler:
    """Select and invoke one configured sampler without exposing its backend API."""

    def __init__(self, config: SamplerConfig) -> None:
        self.config = config
        self._sampler = self._new_implementation(config)

    @property
    def initial_noise_scale(self) -> float:
        """Return the fixed initial future-noise scale for this sampler profile."""

        return self.config.initial_noise_scale

    @property
    def num_steps(self) -> int:
        """Return the number of denoising transitions represented by this profile."""

        return self.config.num_steps

    def prepare_guidance_randomness(
        self,
        initial_sample: torch.Tensor,
        generator: torch.Generator | None,
    ) -> GuidanceSamplingRandomness:
        """Capture one DDIM random stream for semantically identical paired passes."""

        config = self._ddim_config()
        if config.ddim_stochasticity == 0.0:
            return GuidanceSamplingRandomness()
        draws = tuple(
            torch.randn(
                initial_sample.shape,
                dtype=initial_sample.dtype,
                device=initial_sample.device,
                generator=generator,
            )
            for _ in range(config.num_steps - 1)
        )
        return GuidanceSamplingRandomness(
            variance_noises=(*draws, torch.zeros_like(initial_sample))
        )

    def sample(
        self,
        initial_sample: torch.Tensor,
        model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        current_state_constraint: Callable[[torch.Tensor], torch.Tensor],
        generator: torch.Generator | None = None,
        *,
        guidance_randomness: GuidanceSamplingRandomness | None = None,
    ) -> torch.Tensor:
        """Run this profile's unguided sampling pass."""

        if isinstance(self.config, Ddim5SamplerConfig):
            timesteps = torch.tensor(
                self.config.timesteps,
                dtype=initial_sample.dtype,
                device=initial_sample.device,
            )
            return self._sampler.sample(
                initial_sample,
                model,
                current_state_constraint,
                timesteps,
                self.config.num_steps,
                self.config.ddim_stochasticity,
                generator,
                variance_noises=(
                    None if guidance_randomness is None else guidance_randomness.variance_noises
                ),
            )
        return self._sampler.sample(initial_sample, model, current_state_constraint)

    def sample_guided(
        self,
        initial_sample: torch.Tensor,
        model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        current_state_constraint: Callable[[torch.Tensor], torch.Tensor],
        generator: torch.Generator | None,
        guidance: Callable[[torch.Tensor, torch.Tensor], GuidanceGradientResult],
        *,
        gradient_step_coefficient: float,
        guidance_randomness: GuidanceSamplingRandomness,
    ) -> DdimGuidedSampleResult:
        """Run the configured DDIM profile with the project's guidance policy."""

        config = self._ddim_config()
        timesteps = torch.tensor(
            config.timesteps,
            dtype=initial_sample.dtype,
            device=initial_sample.device,
        )
        return self._sampler.sample_guided(
            initial_sample,
            model,
            current_state_constraint,
            timesteps,
            config.num_steps,
            config.ddim_stochasticity,
            generator,
            guidance,
            gradient_step_coefficient=gradient_step_coefficient,
            variance_noises=guidance_randomness.variance_noises,
        )

    @staticmethod
    def _new_implementation(
        config: SamplerConfig,
    ) -> _DdimSampler | _DpmSampler:
        if isinstance(config, Ddim5SamplerConfig):
            return _DdimSampler()
        return _DpmSampler()

    def _ddim_config(self) -> Ddim5SamplerConfig:
        return cast(Ddim5SamplerConfig, self.config)
