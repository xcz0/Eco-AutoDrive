"""Diagnostics assembled from reference-guidance sampling passes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from eco_planner.models.guidance.config import OrthogonalReferenceGuidanceConfig


@dataclass
class GuidanceDiagnostics:
    """Planning-cycle guidance targets and one column per DDIM transition."""

    lateral_target_offset_m: torch.Tensor
    longitudinal_target_speed_fraction: torch.Tensor
    longitudinal_target_speed_delta_mps: torch.Tensor
    lateral_objective_delta: torch.Tensor
    longitudinal_objective_delta: torch.Tensor
    applied_gradient_l2: torch.Tensor
    applied_gradient_max_abs: torch.Tensor
    raw_neighbor_gradient_l2: torch.Tensor
    zero_speed_count: torch.Tensor


def zero_guidance_diagnostics(
    config: OrthogonalReferenceGuidanceConfig,
    action: torch.Tensor,
    *,
    future_len: int,
    num_steps: int,
) -> GuidanceDiagnostics:
    """Build auditable zero-gradient diagnostics for a zero action."""

    batch = action.shape[0]
    zeros = torch.zeros((batch, num_steps), dtype=torch.float32, device=action.device)
    return GuidanceDiagnostics(
        lateral_target_offset_m=config.lateral_max_offset_m * action[:, 0],
        longitudinal_target_speed_fraction=config.longitudinal_max_speed_fraction * action[:, 1],
        longitudinal_target_speed_delta_mps=torch.zeros(
            (batch, future_len), dtype=torch.float32, device=action.device
        ),
        lateral_objective_delta=zeros,
        longitudinal_objective_delta=zeros.clone(),
        applied_gradient_l2=zeros.clone(),
        applied_gradient_max_abs=zeros.clone(),
        raw_neighbor_gradient_l2=zeros.clone(),
        zero_speed_count=torch.zeros((batch, num_steps), dtype=torch.int64, device=action.device),
    )


def stack_guidance_diagnostics(
    config: OrthogonalReferenceGuidanceConfig,
    action: torch.Tensor,
    steps: tuple[Any, ...],
    longitudinal_target_speed_delta_mps: torch.Tensor,
) -> GuidanceDiagnostics:
    """Combine transition diagnostics with the planning-cycle physical targets."""

    if not steps:
        raise RuntimeError("guided DDIM returned no step diagnostics")

    def stack(name: str) -> torch.Tensor:
        return torch.stack([getattr(step, name) for step in steps], dim=1)

    return GuidanceDiagnostics(
        lateral_target_offset_m=config.lateral_max_offset_m * action[:, 0],
        longitudinal_target_speed_fraction=config.longitudinal_max_speed_fraction * action[:, 1],
        longitudinal_target_speed_delta_mps=longitudinal_target_speed_delta_mps,
        lateral_objective_delta=stack("lateral_objective_delta"),
        longitudinal_objective_delta=stack("longitudinal_objective_delta"),
        applied_gradient_l2=stack("applied_gradient_l2"),
        applied_gradient_max_abs=stack("applied_gradient_max_abs"),
        raw_neighbor_gradient_l2=stack("raw_neighbor_gradient_l2"),
        zero_speed_count=stack("zero_speed_count"),
    )
