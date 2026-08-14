"""Orthogonal reference-guidance gradient calculation."""

from __future__ import annotations

import torch

from eco_planner.models.checkpoint.normalization import StateNormalizer
from eco_planner.models.guidance.config import OrthogonalReferenceGuidanceConfig
from eco_planner.models.guidance.contracts import GuidanceGradientResult, validate_guidance_action


class OrthogonalGuidance:
    """Compute centered lateral/longitudinal guidance for one DDIM denoise step."""

    def __init__(
        self,
        config: OrthogonalReferenceGuidanceConfig,
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

        batch, participants, future_len = self._validate_inputs(
            sample,
            predicted_x_start,
            reference_prediction,
            current_states,
            action,
        )
        predicted = predicted_x_start.reshape(batch, participants, future_len + 1, 4)
        predicted_physical = self._state_normalizer.inverse(predicted)
        current_physical = self._state_normalizer.inverse(current_states[:, :, None])[:, :, 0]
        ego_reference = reference_prediction[:, 0]
        heading = ego_reference[..., 2:4]
        heading_norm = torch.linalg.vector_norm(heading, dim=-1)
        if torch.any(heading_norm <= self.config.heading_norm_epsilon):
            raise ValueError("reference heading is degenerate")
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

        if reference_prediction.ndim != 4 or reference_prediction.shape[-1] != 4:
            raise ValueError("reference_prediction must have shape [B, A, T, 4]")
        batch, participants, future_len, _ = reference_prediction.shape
        if tuple(current_states.shape) != (batch, participants, 4):
            raise ValueError("current_states must have shape [B, A, 4]")
        validate_guidance_action(action, batch=batch, device=reference_prediction.device)
        current_physical = self._state_normalizer.inverse(current_states[:, :, None])[:, 0, 0]
        ego_reference = reference_prediction[:, 0]
        heading = ego_reference[..., 2:4]
        heading_norm = torch.linalg.vector_norm(heading, dim=-1)
        if torch.any(heading_norm <= self.config.heading_norm_epsilon):
            raise ValueError("reference heading is degenerate")
        tangent = heading / heading_norm[..., None]
        points = torch.cat([current_physical[:, None, :2], ego_reference[..., :2]], dim=1)
        velocity = torch.diff(points, dim=1) / self.config.trajectory_dt_s
        along_track_speed = torch.sum(tangent * velocity, dim=-1)
        if along_track_speed.shape != (batch, future_len):
            raise RuntimeError("longitudinal target speed must preserve the reference horizon")
        return (
            self.config.longitudinal_max_speed_fraction * action[:, 1, None] * along_track_speed
        ).detach()

    @staticmethod
    def _validate_inputs(
        sample: torch.Tensor,
        predicted_x_start: torch.Tensor,
        reference_prediction: torch.Tensor,
        current_states: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[int, int, int]:
        if not isinstance(sample, torch.Tensor) or sample.ndim != 3:
            raise ValueError("sample must have shape [B, A, (T + 1) * 4]")
        if sample.shape[2] % 4 != 0 or sample.shape[2] <= 4:
            raise ValueError("sample must have shape [B, A, (T + 1) * 4]")
        if not sample.dtype.is_floating_point or not sample.requires_grad:
            raise ValueError("sample must be a floating tensor requiring gradients")
        if predicted_x_start.shape != sample.shape:
            raise ValueError("predicted_x_start must preserve sample shape")
        if predicted_x_start.dtype != sample.dtype or predicted_x_start.device != sample.device:
            raise ValueError("predicted_x_start must preserve sample dtype and device")
        batch, participants, flattened = sample.shape
        future_len = flattened // 4 - 1
        expected_reference = (batch, participants, future_len, 4)
        if tuple(reference_prediction.shape) != expected_reference:
            raise ValueError(f"reference_prediction must have shape {expected_reference}")
        if tuple(current_states.shape) != (batch, participants, 4):
            raise ValueError("current_states must have shape [B, A, 4]")
        for name, value in (
            ("reference_prediction", reference_prediction),
            ("current_states", current_states),
        ):
            if value.dtype != sample.dtype or value.device != sample.device:
                raise ValueError(f"{name} must preserve sample dtype and device")
        validate_guidance_action(action, batch=batch, device=sample.device)
        for name, value in (
            ("sample", sample),
            ("predicted_x_start", predicted_x_start),
            ("reference_prediction", reference_prediction),
            ("current_states", current_states),
            ("guidance action", action),
        ):
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} must be finite")
        return batch, participants, future_len
