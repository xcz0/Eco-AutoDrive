"""Validated DDIM transitions for the PlannerRFT sampler reproduction."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch

from eco_planner.models.guidance import GuidanceGradientResult
from eco_planner.models.vp_schedule import LinearVpSchedule


def _expand(value: torch.Tensor, dimensions: int) -> torch.Tensor:
    return value.reshape([-1] + [1] * (dimensions - 1))


class DdimSampler:
    """Apply explicit DDIM transitions to an ``x_start`` prediction model."""

    def __init__(self) -> None:
        self._schedule = LinearVpSchedule()

    def sample(
        self,
        initial_sample: torch.Tensor,
        model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        current_state_constraint: Callable[[torch.Tensor], torch.Tensor],
        timesteps: torch.Tensor,
        num_steps: int,
        ddim_stochasticity: float,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        """Sample with a validated, decreasing ``[num_steps + 1]`` time schedule."""

        self._validate_inputs(
            initial_sample,
            timesteps,
            num_steps,
            ddim_stochasticity,
            generator,
        )
        sample = self._validate_callback_result(
            "current_state_constraint",
            current_state_constraint(initial_sample),
            initial_sample,
        )
        for index in range(num_steps):
            current_time = timesteps[index]
            next_time = timesteps[index + 1]
            batch_time = current_time.expand(sample.shape[0])
            prediction = self._validate_callback_result(
                "denoise model",
                model(sample, batch_time),
                sample,
            )
            sample = self._transition(
                sample,
                prediction,
                current_time,
                next_time,
                ddim_stochasticity,
                generator,
            )
            sample = self._validate_callback_result(
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
    ) -> DdimGuidedSampleResult:
        """Sample DDIM and apply one audited noisy-sample gradient after each transition."""

        self._validate_inputs(
            initial_sample,
            timesteps,
            num_steps,
            ddim_stochasticity,
            generator,
        )
        if not callable(guidance):
            raise TypeError("guidance must be callable")
        if (
            type(gradient_step_coefficient) not in {int, float}
            or not math.isfinite(gradient_step_coefficient)
            or gradient_step_coefficient <= 0.0
        ):
            raise ValueError("gradient_step_coefficient must be finite and positive")
        sample = self._validate_callback_result(
            "current_state_constraint",
            current_state_constraint(initial_sample),
            initial_sample,
        )
        diagnostics: list[GuidanceGradientResult] = []
        for index in range(num_steps):
            sample_with_grad = sample.detach().requires_grad_(True)
            current_time = timesteps[index]
            next_time = timesteps[index + 1]
            batch_time = current_time.expand(sample.shape[0])
            prediction = self._validate_callback_result(
                "denoise model",
                model(sample_with_grad, batch_time),
                sample_with_grad,
            )
            step = self._validate_guidance_result(
                guidance(sample_with_grad, prediction), sample_with_grad
            )
            transitioned = self._transition(
                sample_with_grad.detach(),
                prediction.detach(),
                current_time,
                next_time,
                ddim_stochasticity,
                generator,
            )
            updated = transitioned - gradient_step_coefficient * step.applied_gradient
            sample = self._validate_callback_result(
                "current_state_constraint",
                current_state_constraint(updated),
                updated,
            ).detach()
            diagnostics.append(step)
        return DdimGuidedSampleResult(sample=sample, diagnostics=tuple(diagnostics))

    def _transition(
        self,
        sample: torch.Tensor,
        prediction: torch.Tensor,
        current_time: torch.Tensor,
        next_time: torch.Tensor,
        stochasticity: float,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        alpha_current = self._schedule.alpha(current_time)
        sigma_current = self._schedule.sigma(current_time)
        alpha_next = self._schedule.alpha(next_time)
        sigma_next = self._schedule.sigma(next_time)
        transition_variance_factor = 1.0 - (alpha_current / alpha_next).square()
        if bool(transition_variance_factor < 0.0):
            raise RuntimeError("DDIM transition variance factor must be non-negative")
        transition_sigma = (
            stochasticity * sigma_next / sigma_current * torch.sqrt(transition_variance_factor)
        )
        direction_variance = sigma_next.square() - transition_sigma.square()
        if bool(direction_variance < 0.0):
            raise RuntimeError("DDIM direction variance must be non-negative")

        alpha_current = _expand(alpha_current, sample.dim())
        sigma_current = _expand(sigma_current, sample.dim())
        alpha_next = _expand(alpha_next, sample.dim())
        transition_sigma = _expand(transition_sigma, sample.dim())
        direction_scale = _expand(torch.sqrt(direction_variance), sample.dim())
        predicted_noise = (sample - alpha_current * prediction) / sigma_current
        if stochasticity > 0.0 and float(next_time.item()) > 0.0:
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
        result = (
            alpha_next * prediction
            + direction_scale * predicted_noise
            + transition_sigma * random_noise
        )
        if not torch.isfinite(result).all():
            raise RuntimeError("DDIM transition produced non-finite state")
        return result

    @staticmethod
    def _validate_inputs(
        initial_sample: torch.Tensor,
        timesteps: torch.Tensor,
        num_steps: int,
        stochasticity: float,
        generator: torch.Generator | None,
    ) -> None:
        if not isinstance(initial_sample, torch.Tensor):
            raise TypeError("initial_sample must be a torch.Tensor")
        if initial_sample.ndim != 3 or any(size <= 0 for size in initial_sample.shape):
            raise ValueError("initial_sample must have a non-empty [B, A, D] shape")
        if not initial_sample.dtype.is_floating_point:
            raise TypeError("initial_sample must use a floating dtype")
        if not torch.isfinite(initial_sample).all():
            raise ValueError("initial_sample must be finite")
        if type(num_steps) is not int or num_steps <= 0:
            raise ValueError("num_steps must be a positive integer")
        if not isinstance(timesteps, torch.Tensor):
            raise TypeError("timesteps must be a torch.Tensor")
        if timesteps.device != initial_sample.device:
            raise ValueError("timesteps must be on the initial_sample device")
        if timesteps.dtype != initial_sample.dtype:
            raise TypeError("timesteps must use the initial_sample dtype")
        if timesteps.ndim != 1 or timesteps.shape[0] != num_steps + 1:
            raise ValueError("timesteps must have shape [num_steps + 1]")
        if not torch.isfinite(timesteps).all():
            raise ValueError("timesteps must be finite")
        if float(timesteps[0].item()) != 1.0 or float(timesteps[-1].item()) != 0.0:
            raise ValueError("timesteps must start at 1.0 and end at 0.0")
        if not torch.all(timesteps[:-1] > timesteps[1:]):
            raise ValueError("timesteps must be strictly decreasing")
        if type(stochasticity) not in {int, float} or not math.isfinite(stochasticity):
            raise TypeError("ddim_stochasticity must be a finite number")
        if not 0.0 <= stochasticity <= 1.0:
            raise ValueError("ddim_stochasticity must be in [0, 1]")
        if stochasticity > 0.0 and generator is None:
            raise ValueError("stochastic DDIM sampling requires an explicit torch.Generator")
        if generator is not None:
            if not isinstance(generator, torch.Generator):
                raise TypeError("generator must be a torch.Generator")
            if torch.device(generator.device) != initial_sample.device:
                raise ValueError("generator must be on the initial_sample device")

    @staticmethod
    def _validate_callback_result(
        name: str,
        result: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(result, torch.Tensor):
            raise TypeError(f"{name} must return a torch.Tensor")
        if result.shape != reference.shape:
            raise ValueError(f"{name} must preserve sample shape")
        if result.dtype != reference.dtype:
            raise TypeError(f"{name} must preserve sample dtype")
        if result.device != reference.device:
            raise ValueError(f"{name} must preserve sample device")
        if not torch.isfinite(result).all():
            raise ValueError(f"{name} must return finite values")
        return result

    @staticmethod
    def _validate_guidance_result(
        result: GuidanceGradientResult,
        reference: torch.Tensor,
    ) -> GuidanceGradientResult:
        if not isinstance(result, GuidanceGradientResult):
            raise TypeError("guidance must return GuidanceGradientResult")
        gradient = result.applied_gradient
        if gradient.shape != reference.shape:
            raise ValueError("guidance gradient must preserve sample shape")
        if gradient.dtype != reference.dtype or gradient.device != reference.device:
            raise ValueError("guidance gradient must preserve sample dtype and device")
        batch = reference.shape[0]
        for name in (
            "lateral_objective_delta",
            "longitudinal_objective_delta",
            "applied_gradient_l2",
            "applied_gradient_max_abs",
            "raw_neighbor_gradient_l2",
            "zero_speed_count",
        ):
            value = getattr(result, name)
            if not isinstance(value, torch.Tensor) or value.shape != (batch,):
                raise ValueError(f"guidance diagnostic {name} must have shape [B]")
            if value.device != reference.device:
                raise ValueError(f"guidance diagnostic {name} must be on the sample device")
            if value.dtype.is_floating_point and not torch.isfinite(value).all():
                raise ValueError(f"guidance diagnostic {name} must be finite")
        if not torch.isfinite(gradient).all():
            raise ValueError("guidance gradient must be finite")
        return result


@dataclass(frozen=True)
class DdimGuidedSampleResult:
    """Final normalized DDIM sample and one diagnostic record per transition."""

    sample: torch.Tensor
    diagnostics: tuple[GuidanceGradientResult, ...]
