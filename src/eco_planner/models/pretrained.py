"""Strict official-checkpoint loading and inference facade."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from eco_planner.models.checkpoint import (
    OFFICIAL_EMA_TENSOR_COUNT,
    OFFICIAL_PARAMETER_COUNT,
    CheckpointLoadReport,
    extract_official_ema_state_dict,
)
from eco_planner.models.config import OfficialDiffusionPlannerConfig
from eco_planner.models.contracts import (
    validate_official_observation,
    validate_standard_normal_noise,
)
from eco_planner.models.diffusion_planner import DiffusionPlanner
from eco_planner.models.guidance import (
    GuidanceConfig,
    GuidanceDiagnostics,
    NoGuidanceConfig,
    OrthogonalGuidance,
    OrthogonalReferenceGuidanceConfig,
    validate_guidance_action,
    validate_guidance_sampler,
)
from eco_planner.models.planning_sampler import PlanningSampler
from eco_planner.models.sampling_config import SamplerConfig


class PretrainedDiffusionPlanner(nn.Module):
    """Frozen, official-EMA model with deterministic baseline sampling."""

    def __init__(
        self,
        config: OfficialDiffusionPlannerConfig,
        model: DiffusionPlanner,
        sampler_config: SamplerConfig,
        guidance_config: GuidanceConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = model
        self.sampler_config = sampler_config
        self.guidance_config = guidance_config or NoGuidanceConfig()
        validate_guidance_sampler(self.guidance_config, sampler_config)
        self._sampler = PlanningSampler(sampler_config)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.train(False)

    @property
    def _runtime_device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration as error:
            raise RuntimeError("Diffusion Planner must contain parameters") from error

    def train(self, mode: bool = True) -> PretrainedDiffusionPlanner:
        if mode:
            raise RuntimeError("PretrainedDiffusionPlanner is frozen and cannot enter train mode")
        super().train(False)
        return self

    def forward(
        self,
        observation: Mapping[str, torch.Tensor],
        standard_normal_noise: torch.Tensor,
        transition_generator: torch.Generator | None = None,
        guidance_action: torch.Tensor | None = None,
    ) -> PlannerInferenceResult:
        if self.training or self.model.training:
            raise RuntimeError("PretrainedDiffusionPlanner must remain in eval mode")
        device = self._runtime_device
        batch = validate_official_observation(observation, device)
        participants = 1 + self.config.predicted_neighbor_num
        validate_standard_normal_noise(
            standard_normal_noise,
            batch=batch,
            participants=participants,
            future_len=self.config.future_len,
            device=device,
        )
        inputs = self.config.observation_normalizer(observation)
        encoding = self.model.encode(inputs)
        ego_current = inputs["ego_current_state"][:, None, :4]
        neighbors_current = inputs["neighbor_agents_past"][
            :, : self.config.predicted_neighbor_num, -1, :4
        ]
        neighbor_current_mask = torch.sum(torch.ne(neighbors_current, 0), dim=-1) == 0
        current_states = torch.cat([ego_current, neighbors_current], dim=1)
        initial = torch.cat(
            [current_states[:, :, None], self._sampler.initial_noise_scale * standard_normal_noise],
            dim=2,
        ).reshape(batch, participants, -1)

        def denoiser(sample: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
            prediction = self.model.denoise(
                sample, timestep, encoding, inputs["route_lanes"], neighbor_current_mask
            )
            if self.sampler_config.name == "ddim5":
                prediction = prediction.to(dtype=sample.dtype)
            return prediction

        def constrain(sample: torch.Tensor) -> torch.Tensor:
            constrained = sample.reshape(batch, participants, self.config.future_len + 1, 4)
            constrained = constrained.clone()
            constrained[:, :, 0] = current_states
            return constrained.reshape(batch, participants, -1)

        guidance_randomness = (
            self._sampler.prepare_guidance_randomness(initial, transition_generator)
            if isinstance(self.guidance_config, OrthogonalReferenceGuidanceConfig)
            else None
        )
        normalized_sample = self._sampler.sample(
            initial,
            denoiser,
            constrain,
            transition_generator,
            guidance_randomness=guidance_randomness,
        )
        normalized = normalized_sample.reshape(batch, participants, self.config.future_len + 1, 4)
        prediction = self.config.state_normalizer.inverse(normalized)[:, :, 1:]
        if isinstance(self.guidance_config, NoGuidanceConfig):
            if guidance_action is not None:
                raise ValueError("guidance_action requires active guidance configuration")
            return PlannerInferenceResult(prediction=prediction)

        action = (
            torch.tensor(
                self.guidance_config.fixed_action,
                dtype=torch.float32,
                device=device,
            )
            .expand(batch, -1)
            .clone()
            if guidance_action is None
            else guidance_action
        )
        validate_guidance_action(action, batch=batch, device=device)
        if torch.count_nonzero(action).item() == 0:
            diagnostics = _zero_guidance_diagnostics(
                self.guidance_config,
                action,
                batch,
                self._sampler.num_steps,
            )
            return PlannerInferenceResult(
                prediction=prediction,
                reference_prediction=prediction,
                guidance_action=action,
                guidance_diagnostics=diagnostics,
            )

        guidance = OrthogonalGuidance(self.guidance_config, self.config.state_normalizer)

        def guidance_callback(sample: torch.Tensor, predicted_x_start: torch.Tensor) -> Any:
            return guidance.gradient(
                sample,
                predicted_x_start,
                prediction,
                current_states,
                action,
            )

        if guidance_randomness is None:
            raise RuntimeError("active guidance did not prepare shared DDIM randomness")
        guided_result = self._sampler.sample_guided(
            initial,
            denoiser,
            constrain,
            transition_generator,
            guidance_callback,
            gradient_step_coefficient=self.guidance_config.gradient_step_coefficient,
            guidance_randomness=guidance_randomness,
        )
        guided_normalized = guided_result.sample.reshape(
            batch, participants, self.config.future_len + 1, 4
        )
        guided_prediction = self.config.state_normalizer.inverse(guided_normalized)[:, :, 1:]
        diagnostics = _stack_guidance_diagnostics(
            self.guidance_config,
            action,
            guided_result.diagnostics,
            guidance.longitudinal_target_speed_delta_mps(
                prediction,
                current_states,
                action,
            ),
        )
        return PlannerInferenceResult(
            prediction=guided_prediction,
            reference_prediction=prediction,
            guidance_action=action,
            guidance_diagnostics=diagnostics,
        )


def load_official_diffusion_planner(
    args_path: Path,
    checkpoint_path: Path,
    sampler_config: SamplerConfig,
    guidance_config: GuidanceConfig | None = None,
) -> tuple[PretrainedDiffusionPlanner, CheckpointLoadReport]:
    """Load the pinned official EMA checkpoint without compatibility fallbacks."""

    config = OfficialDiffusionPlannerConfig.from_json(args_path)
    checkpoint: Any = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = extract_official_ema_state_dict(checkpoint)
    model = DiffusionPlanner(config)
    model.load_state_dict(state_dict, strict=True)
    planner = PretrainedDiffusionPlanner(config, model, sampler_config, guidance_config)
    return planner, CheckpointLoadReport(
        ema_tensor_count=OFFICIAL_EMA_TENSOR_COUNT,
        parameter_count=OFFICIAL_PARAMETER_COUNT,
    )


@dataclass
class PlannerInferenceResult:
    """Validated planner prediction and optional reference-guidance audit values."""

    prediction: torch.Tensor
    reference_prediction: torch.Tensor | None = None
    guidance_action: torch.Tensor | None = None
    guidance_diagnostics: GuidanceDiagnostics | None = None


def _zero_guidance_diagnostics(
    config: OrthogonalReferenceGuidanceConfig,
    action: torch.Tensor,
    batch: int,
    num_steps: int,
) -> GuidanceDiagnostics:
    zeros = torch.zeros((batch, num_steps), dtype=torch.float32, device=action.device)
    return GuidanceDiagnostics(
        lateral_target_offset_m=config.lateral_max_offset_m * action[:, 0],
        longitudinal_target_speed_fraction=(config.longitudinal_max_speed_fraction * action[:, 1]),
        longitudinal_target_speed_delta_mps=torch.zeros(
            (batch, 80), dtype=torch.float32, device=action.device
        ),
        lateral_objective_delta=zeros,
        longitudinal_objective_delta=zeros.clone(),
        applied_gradient_l2=zeros.clone(),
        applied_gradient_max_abs=zeros.clone(),
        raw_neighbor_gradient_l2=zeros.clone(),
        zero_speed_count=torch.zeros((batch, num_steps), dtype=torch.int64, device=action.device),
    )


def _stack_guidance_diagnostics(
    config: OrthogonalReferenceGuidanceConfig,
    action: torch.Tensor,
    steps: tuple[Any, ...],
    longitudinal_target_speed_delta_mps: torch.Tensor,
) -> GuidanceDiagnostics:
    if not steps:
        raise RuntimeError("guided DDIM returned no step diagnostics")

    def stack(name: str) -> torch.Tensor:
        return torch.stack([getattr(step, name) for step in steps], dim=1)

    return GuidanceDiagnostics(
        lateral_target_offset_m=config.lateral_max_offset_m * action[:, 0],
        longitudinal_target_speed_fraction=(config.longitudinal_max_speed_fraction * action[:, 1]),
        longitudinal_target_speed_delta_mps=longitudinal_target_speed_delta_mps,
        lateral_objective_delta=stack("lateral_objective_delta"),
        longitudinal_objective_delta=stack("longitudinal_objective_delta"),
        applied_gradient_l2=stack("applied_gradient_l2"),
        applied_gradient_max_abs=stack("applied_gradient_max_abs"),
        raw_neighbor_gradient_l2=stack("raw_neighbor_gradient_l2"),
        zero_speed_count=stack("zero_speed_count"),
    )
