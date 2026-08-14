"""Public tensor contracts for reference guidance."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GuidanceGradientResult:
    """One audited physical-energy gradient in normalized joint-sample space."""

    applied_gradient: torch.Tensor
    lateral_objective_delta: torch.Tensor
    longitudinal_objective_delta: torch.Tensor
    applied_gradient_l2: torch.Tensor
    applied_gradient_max_abs: torch.Tensor
    raw_neighbor_gradient_l2: torch.Tensor
    zero_speed_count: torch.Tensor


def validate_guidance_action(action: torch.Tensor, *, batch: int, device: torch.device) -> None:
    """Validate the signed lateral/longitudinal action without clipping."""

    if not isinstance(action, torch.Tensor):
        raise TypeError("guidance action must be a torch.Tensor")
    if tuple(action.shape) != (batch, 2):
        raise ValueError("guidance action must have shape [B, 2]")
    if action.dtype != torch.float32:
        raise TypeError("guidance action must use torch.float32")
    if action.device != device:
        raise ValueError("guidance action must be on the sample device")
    if not torch.isfinite(action).all():
        raise ValueError("guidance action must be finite")
    if torch.any((action < -1.0) | (action > 1.0)):
        raise ValueError("guidance action must be in [-1, 1]")
