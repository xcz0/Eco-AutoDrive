"""Strict stage-2 reference guidance configuration and public contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from omegaconf import DictConfig, OmegaConf

from eco_planner.models.normalization import StateNormalizer
from eco_planner.models.sampling_config import (
    Ddim5SamplerConfig,
    SamplerConfig,
)

_FORMULA_LABEL = "centered_energy_gradient_delta_v1"


@dataclass(frozen=True)
class NoGuidanceConfig:
    """Disable reference generation and orthogonal guidance."""

    name: Literal["none"] = "none"


@dataclass(frozen=True)
class OrthogonalReferenceGuidanceConfig:
    """Fully explicit fixed-action profile for stage-2 guidance evaluation."""

    name: Literal["orthogonal_reference"]
    formula_label: Literal["centered_energy_gradient_delta_v1"]
    lateral_scale: float
    longitudinal_scale: float
    lateral_max_offset_m: float
    longitudinal_max_speed_fraction: float
    trajectory_dt_s: float
    gradient_step_coefficient: float
    reference_refresh_cycles: int
    share_scene_encoding: bool
    share_initial_noise: bool
    share_transition_noise: bool
    heading_norm_epsilon: float
    zero_speed_tolerance_mps: float

    @property
    def fixed_action(self) -> tuple[float, float]:
        return (self.lateral_scale, self.longitudinal_scale)


GuidanceConfig = NoGuidanceConfig | OrthogonalReferenceGuidanceConfig


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
        total = lateral_delta + longitudinal_delta
        raw_gradient = torch.autograd.grad(total.sum(), sample, create_graph=False)[0]
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

        if reference_prediction.ndim != 4 or reference_prediction.shape[1:] != (11, 80, 4):
            raise ValueError("reference_prediction must have shape [B, 11, 80, 4]")
        batch = reference_prediction.shape[0]
        if tuple(current_states.shape) != (batch, 11, 4):
            raise ValueError("current_states must have shape [B, 11, 4]")
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
            raise ValueError("sample must have shape [B, 11, (T + 1) * 4]")
        if sample.shape[1] != 11 or sample.shape[2] % 4 != 0 or sample.shape[2] <= 4:
            raise ValueError("sample must have shape [B, 11, (T + 1) * 4]")
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
            raise ValueError("current_states must have shape [B, 11, 4]")
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


def parse_guidance_config(config: DictConfig) -> GuidanceConfig:
    """Parse one strict Hydra guidance mapping without hidden defaults."""

    if not isinstance(config, DictConfig):
        raise TypeError("guidance configuration must be a DictConfig")
    raw = OmegaConf.to_container(config, resolve=True)
    if not isinstance(raw, dict):
        raise TypeError("guidance configuration must resolve to a dictionary")
    name = raw.get("name")
    if name == "none":
        _require_exact_keys(raw, {"name"}, "none")
        return NoGuidanceConfig()
    if name != "orthogonal_reference":
        raise ValueError("guidance.name must be either 'none' or 'orthogonal_reference'")

    required = {
        "name",
        "formula_label",
        "lateral_scale",
        "longitudinal_scale",
        "lateral_max_offset_m",
        "longitudinal_max_speed_fraction",
        "trajectory_dt_s",
        "gradient_step_coefficient",
        "reference_refresh_cycles",
        "share_scene_encoding",
        "share_initial_noise",
        "share_transition_noise",
        "heading_norm_epsilon",
        "zero_speed_tolerance_mps",
    }
    _require_exact_keys(raw, required, "orthogonal_reference")
    formula_label = raw["formula_label"]
    if formula_label != _FORMULA_LABEL:
        raise ValueError(f"formula_label must equal {_FORMULA_LABEL!r}")
    lateral_scale = _bounded_scale(raw["lateral_scale"], "lateral_scale")
    longitudinal_scale = _bounded_scale(raw["longitudinal_scale"], "longitudinal_scale")
    lateral_max_offset_m = _positive_float(raw["lateral_max_offset_m"], "lateral_max_offset_m")
    longitudinal_max_speed_fraction = _positive_float(
        raw["longitudinal_max_speed_fraction"], "longitudinal_max_speed_fraction"
    )
    trajectory_dt_s = _positive_float(raw["trajectory_dt_s"], "trajectory_dt_s")
    gradient_step_coefficient = _finite_float(
        raw["gradient_step_coefficient"], "gradient_step_coefficient"
    )
    if gradient_step_coefficient != 1.0:
        raise ValueError("stage-2 guidance requires a unit coefficient")
    reference_refresh_cycles = raw["reference_refresh_cycles"]
    if type(reference_refresh_cycles) is not int or reference_refresh_cycles != 1:
        raise ValueError("reference must refresh every planning cycle")
    for field in ("share_scene_encoding", "share_initial_noise", "share_transition_noise"):
        if raw[field] is not True:
            raise ValueError(f"{field} must be true for stage-2 guidance")
    heading_norm_epsilon = _positive_float(raw["heading_norm_epsilon"], "heading_norm_epsilon")
    zero_speed_tolerance_mps = _positive_float(
        raw["zero_speed_tolerance_mps"], "zero_speed_tolerance_mps"
    )
    return OrthogonalReferenceGuidanceConfig(
        name="orthogonal_reference",
        formula_label=_FORMULA_LABEL,
        lateral_scale=lateral_scale,
        longitudinal_scale=longitudinal_scale,
        lateral_max_offset_m=lateral_max_offset_m,
        longitudinal_max_speed_fraction=longitudinal_max_speed_fraction,
        trajectory_dt_s=trajectory_dt_s,
        gradient_step_coefficient=gradient_step_coefficient,
        reference_refresh_cycles=reference_refresh_cycles,
        share_scene_encoding=True,
        share_initial_noise=True,
        share_transition_noise=True,
        heading_norm_epsilon=heading_norm_epsilon,
        zero_speed_tolerance_mps=zero_speed_tolerance_mps,
    )


def validate_guidance_sampler(config: GuidanceConfig, sampler: SamplerConfig) -> None:
    """Reject active guidance outside the standard-Gaussian DDIM-5 profile."""

    if isinstance(config, NoGuidanceConfig):
        return
    if not isinstance(sampler, Ddim5SamplerConfig):
        raise ValueError("orthogonal reference guidance requires DDIM-5")
    if sampler.initial_noise_scale != 1.0 or sampler.parity_label != "plannerrft_paper_text":
        raise ValueError("orthogonal reference guidance requires standard-Gaussian DDIM-5")


def validate_guidance_action(
    action: torch.Tensor,
    *,
    batch: int,
    device: torch.device,
) -> None:
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


def _require_exact_keys(raw: dict[object, object], expected: set[str], name: str) -> None:
    keys = set(raw)
    if keys != expected:
        missing = sorted(expected - keys)
        unexpected = sorted(str(key) for key in keys - expected)
        raise ValueError(
            f"{name} guidance keys mismatch; missing={missing}, unexpected={unexpected}"
        )


def _finite_float(value: object, name: str) -> float:
    if type(value) not in {int, float}:
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_float(value: object, name: str) -> float:
    result = _finite_float(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _bounded_scale(value: object, name: str) -> float:
    result = _finite_float(value, name)
    if not -1.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [-1, 1]")
    return result
