"""Validation and result contracts shared by Diffusers-backed samplers."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from eco_planner.models.guidance import GuidanceGradientResult


@dataclass(frozen=True)
class DdimGuidedSampleResult:
    """Final normalized DDIM sample and one diagnostic record per transition."""

    sample: torch.Tensor
    diagnostics: tuple[GuidanceGradientResult, ...]


def validate_ddim_inputs(
    initial_sample: torch.Tensor,
    timesteps: torch.Tensor,
    num_steps: int,
    stochasticity: float,
    generator: torch.Generator | None,
) -> None:
    """Validate the DDIM profile and all values crossing the sampler boundary."""

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


def validate_callback_result(
    name: str,
    result: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Require callbacks to preserve the sample tensor contract."""

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


def validate_guidance_result(
    result: GuidanceGradientResult,
    reference: torch.Tensor,
) -> GuidanceGradientResult:
    """Require guidance to return auditable gradients and diagnostics."""

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
