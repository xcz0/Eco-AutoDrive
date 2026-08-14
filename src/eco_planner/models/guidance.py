"""Orthogonal guidance math and inference diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from eco_planner.models.config import (
    OrthogonalPolicyGuidanceConfig,
    OrthogonalReferenceGuidanceConfig,
    StateNormalizer,
)


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


_OrthogonalGuidanceConfig = OrthogonalReferenceGuidanceConfig | OrthogonalPolicyGuidanceConfig


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
    config: _OrthogonalGuidanceConfig,
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
    config: _OrthogonalGuidanceConfig,
    action: torch.Tensor,
    steps: tuple[Any, ...],
    longitudinal_target_speed_delta_mps: torch.Tensor,
) -> GuidanceDiagnostics:
    """Combine transition diagnostics with the planning-cycle physical targets."""

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


class OrthogonalGuidance:
    """Compute centered lateral/longitudinal guidance for one DDIM denoise step."""

    def __init__(
        self,
        config: OrthogonalReferenceGuidanceConfig | OrthogonalPolicyGuidanceConfig,
        state_normalizer: StateNormalizer,
    ) -> None:
        self.config = config
        self._state_normalizer = state_normalizer

    def gradient(
        self,
        sample: torch.Tensor,
        predicted_x_start: torch.Tensor,
        reference_prediction: torch.Tensor,
        current_states: torch.Tensor,
        action: torch.Tensor,
    ) -> GuidanceGradientResult:
        """Return the masked noisy-sample gradient and per-batch diagnostics."""

        batch, participants, flattened = sample.shape
        future_len = flattened // 4 - 1
        predicted = predicted_x_start.reshape(batch, participants, future_len + 1, 4)
        predicted_physical = self._state_normalizer.inverse(predicted)
        current_physical = self._state_normalizer.inverse(current_states[:, :, None])[:, :, 0]
        ego_reference = reference_prediction[:, 0]
        heading = ego_reference[..., 2:4]
        heading_norm = torch.linalg.vector_norm(heading, dim=-1)
        tangent = heading / heading_norm[..., None]
        normal = torch.stack((-tangent[..., 1], tangent[..., 0]), dim=-1)

        ego_predicted_positions = predicted_physical[:, 0, 1:, :2]
        ego_reference_positions = ego_reference[..., :2]
        lateral_displacement = torch.sum(
            normal * (ego_predicted_positions - ego_reference_positions), dim=-1
        )
        lateral_target = self.config.lateral_max_offset_m * action[:, 0, None]
        lateral_delta = torch.mean(
            lateral_target.square() - 2.0 * lateral_target * lateral_displacement,
            dim=-1,
        )

        predicted_points = torch.cat(
            [current_physical[:, 0, None, :2], ego_predicted_positions], dim=1
        )
        reference_points = torch.cat(
            [current_physical[:, 0, None, :2], ego_reference_positions], dim=1
        )
        predicted_velocity = torch.diff(predicted_points, dim=1) / self.config.trajectory_dt_s
        reference_velocity = torch.diff(reference_points, dim=1) / self.config.trajectory_dt_s
        reference_along_track_speed = torch.sum(tangent * reference_velocity, dim=-1)
        relative_along_track_speed = torch.sum(
            tangent * (predicted_velocity - reference_velocity), dim=-1
        )
        longitudinal_target = (
            self.config.longitudinal_max_speed_fraction
            * action[:, 1, None]
            * reference_along_track_speed
        )
        longitudinal_delta = torch.mean(
            longitudinal_target.square() - 2.0 * longitudinal_target * relative_along_track_speed,
            dim=-1,
        )
        raw_gradient = torch.autograd.grad((lateral_delta + longitudinal_delta).sum(), sample)[0]
        raw_gradient = raw_gradient.reshape(batch, participants, future_len + 1, 4)
        raw_neighbor_l2 = torch.linalg.vector_norm(raw_gradient[:, 1:].reshape(batch, -1), dim=-1)
        applied = raw_gradient.clone()
        applied[:, :, 0] = 0.0
        applied[:, 1:] = 0.0
        applied_flat = applied.reshape(batch, participants, -1)
        applied_values = applied_flat.reshape(batch, -1)
        zero_speed_count = torch.sum(
            torch.linalg.vector_norm(reference_velocity, dim=-1)
            <= self.config.zero_speed_tolerance_mps,
            dim=-1,
        )
        return GuidanceGradientResult(
            applied_gradient=applied_flat.detach(),
            lateral_objective_delta=lateral_delta.detach(),
            longitudinal_objective_delta=longitudinal_delta.detach(),
            applied_gradient_l2=torch.linalg.vector_norm(applied_values, dim=-1).detach(),
            applied_gradient_max_abs=torch.amax(torch.abs(applied_values), dim=-1).detach(),
            raw_neighbor_gradient_l2=raw_neighbor_l2.detach(),
            zero_speed_count=zero_speed_count.detach(),
        )

    def longitudinal_target_speed_delta_mps(
        self,
        reference_prediction: torch.Tensor,
        current_states: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Return the physical 10 Hz longitudinal speed target for artifact auditing."""

        batch, _, _, _ = reference_prediction.shape
        current_physical = self._state_normalizer.inverse(current_states[:, :, None])[:, 0, 0]
        ego_reference = reference_prediction[:, 0]
        heading = ego_reference[..., 2:4]
        heading_norm = torch.linalg.vector_norm(heading, dim=-1)
        tangent = heading / heading_norm[..., None]
        points = torch.cat([current_physical[:, None, :2], ego_reference[..., :2]], dim=1)
        velocity = torch.diff(points, dim=1) / self.config.trajectory_dt_s
        along_track_speed = torch.sum(tangent * velocity, dim=-1)
        return (
            self.config.longitudinal_max_speed_fraction * action[:, 1, None] * along_track_speed
        ).detach()
