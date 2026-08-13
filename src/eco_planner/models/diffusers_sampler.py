"""Diffusers-backed DDIM-5 sampler with the Planner's continuous-time contract."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from diffusers import DDIMScheduler

from eco_planner.models.ddim_sampler import DdimGuidedSampleResult, DdimSampler
from eco_planner.models.guidance import GuidanceGradientResult
from eco_planner.models.vp_schedule import LinearVpSchedule

_NUM_TRAIN_TIMESTEPS = 1000
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


class DiffusersDdimSampler:
    """Apply the validated DDIM-5 profile through ``diffusers.DDIMScheduler``."""

    def __init__(self) -> None:
        self._schedule = LinearVpSchedule()
        if self._schedule.total_n != _NUM_TRAIN_TIMESTEPS:
            raise RuntimeError("Diffusers DDIM-5 adapter requires exactly 1000 training timesteps")
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

        self._validate_inputs(initial_sample, timesteps, num_steps, ddim_stochasticity, generator)
        self._validate_variance_noises(
            variance_noises, initial_sample, num_steps, ddim_stochasticity
        )
        scheduler = self._new_scheduler(initial_sample.device, initial_sample.dtype)
        sample = DdimSampler._validate_callback_result(
            "current_state_constraint",
            current_state_constraint(initial_sample),
            initial_sample,
        )
        for index, scheduler_timestep in enumerate(scheduler.timesteps):
            prediction = self._prediction(model, sample, scheduler_timestep)
            sample = self._step(
                scheduler,
                sample,
                prediction,
                scheduler_timestep,
                index,
                ddim_stochasticity,
                generator,
                None if variance_noises is None else variance_noises[index],
            )
            sample = DdimSampler._validate_callback_result(
                "current_state_constraint",
                current_state_constraint(sample),
                sample,
            )
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

        self._validate_inputs(initial_sample, timesteps, num_steps, ddim_stochasticity, generator)
        self._validate_variance_noises(
            variance_noises, initial_sample, num_steps, ddim_stochasticity
        )
        if not callable(guidance):
            raise TypeError("guidance must be callable")
        if (
            type(gradient_step_coefficient) not in {int, float}
            or not np.isfinite(gradient_step_coefficient)
            or gradient_step_coefficient <= 0.0
        ):
            raise ValueError("gradient_step_coefficient must be finite and positive")
        scheduler = self._new_scheduler(initial_sample.device, initial_sample.dtype)
        sample = DdimSampler._validate_callback_result(
            "current_state_constraint",
            current_state_constraint(initial_sample),
            initial_sample,
        )
        diagnostics: list[GuidanceGradientResult] = []
        for index, scheduler_timestep in enumerate(scheduler.timesteps):
            sample_with_grad = sample.detach().requires_grad_(True)
            prediction = self._prediction(model, sample_with_grad, scheduler_timestep)
            step = DdimSampler._validate_guidance_result(
                guidance(sample_with_grad, prediction), sample_with_grad
            )
            transitioned = self._step(
                scheduler,
                sample_with_grad.detach(),
                prediction.detach(),
                scheduler_timestep,
                index,
                ddim_stochasticity,
                generator,
                None if variance_noises is None else variance_noises[index],
            )
            updated = transitioned - gradient_step_coefficient * step.applied_gradient
            sample = DdimSampler._validate_callback_result(
                "current_state_constraint",
                current_state_constraint(updated),
                updated,
            ).detach()
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
        scheduler_timestep: torch.Tensor,
    ) -> torch.Tensor:
        discrete_timestep = int(scheduler_timestep.item())
        continuous_timestep = (discrete_timestep + 1) / _NUM_TRAIN_TIMESTEPS
        batch_timestep = torch.full(
            (sample.shape[0],),
            continuous_timestep,
            dtype=sample.dtype,
            device=sample.device,
        )
        return DdimSampler._validate_callback_result(
            "denoise model", model(sample, batch_timestep), sample
        )

    @staticmethod
    def _step(
        scheduler: DDIMScheduler,
        sample: torch.Tensor,
        prediction: torch.Tensor,
        scheduler_timestep: torch.Tensor,
        index: int,
        stochasticity: float,
        generator: torch.Generator | None,
        variance_noise: torch.Tensor | None,
    ) -> torch.Tensor:
        if variance_noise is not None:
            random_noise = variance_noise
        elif stochasticity > 0.0 and index < scheduler.num_inference_steps - 1:
            if generator is None:
                raise RuntimeError("validated stochastic DDIM transition lost its generator")
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
            timestep=int(scheduler_timestep.item()),
            sample=sample,
            eta=stochasticity,
            variance_noise=random_noise,
        ).prev_sample
        if not torch.isfinite(result).all():
            raise RuntimeError("Diffusers DDIM transition produced non-finite state")
        return result

    @staticmethod
    def _validate_variance_noises(
        variance_noises: tuple[torch.Tensor, ...] | None,
        sample: torch.Tensor,
        num_steps: int,
        stochasticity: float,
    ) -> None:
        if variance_noises is None:
            return
        if stochasticity == 0.0:
            raise ValueError("variance_noises require non-zero ddim_stochasticity")
        if len(variance_noises) != num_steps:
            raise ValueError("variance_noises must have one tensor per DDIM transition")
        for index, noise in enumerate(variance_noises):
            if not isinstance(noise, torch.Tensor):
                raise TypeError(f"variance_noises[{index}] must be a torch.Tensor")
            if noise.shape != sample.shape:
                raise ValueError(f"variance_noises[{index}] must preserve sample shape")
            if noise.dtype != sample.dtype or noise.device != sample.device:
                raise ValueError(f"variance_noises[{index}] must preserve sample dtype and device")
            if not torch.isfinite(noise).all():
                raise ValueError(f"variance_noises[{index}] must be finite")

    @staticmethod
    def _validate_inputs(
        initial_sample: torch.Tensor,
        timesteps: torch.Tensor,
        num_steps: int,
        stochasticity: float,
        generator: torch.Generator | None,
    ) -> None:
        DdimSampler._validate_inputs(initial_sample, timesteps, num_steps, stochasticity, generator)
        if num_steps != 5:
            raise ValueError("Diffusers DDIM sampler only supports the five-step profile")
        expected = torch.tensor(
            _DDIM5_TIMESTEPS,
            dtype=timesteps.dtype,
            device=timesteps.device,
        )
        if not torch.equal(timesteps, expected):
            raise ValueError(f"Diffusers DDIM timesteps must equal {_DDIM5_TIMESTEPS}")
