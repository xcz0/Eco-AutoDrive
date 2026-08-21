"""Frozen checkpoint-backed diffusion planner."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from eco_planner.models.checkpoint import CheckpointLoadReport, extract_official_ema_state_dict
from eco_planner.models.config import (
    GuidanceConfig,
    NoGuidanceConfig,
    OfficialDiffusionPlannerConfig,
    OrthogonalPolicyGuidanceConfig,
    OrthogonalReferenceGuidanceConfig,
    SamplerConfig,
)
from eco_planner.models.guidance import (
    GuidanceDiagnostics,
    OrthogonalGuidance,
    stack_guidance_diagnostics,
    zero_guidance_diagnostics,
)
from eco_planner.models.network import DiffusionPlanner
from eco_planner.models.sampling import DiffusionSampler


@dataclass
class PlannerInferenceResult:
    """Validated planner prediction and optional reference-guidance audit values."""

    prediction: torch.Tensor
    reference_prediction: torch.Tensor | None = None
    guidance_action: torch.Tensor | None = None
    guidance_diagnostics: GuidanceDiagnostics | None = None


@dataclass(frozen=True)
class PlannerPolicyContext:
    """Frozen planner features and the physical ego reference for one policy decision."""

    scene_tokens: torch.Tensor
    scene_padding_mask: torch.Tensor
    navigation_tokens: torch.Tensor
    navigation_padding_mask: torch.Tensor
    reference_trajectory: torch.Tensor


@dataclass
class PreparedPolicyGuidance:
    """One-use DDIM reference pass retained until the policy selects an action."""

    initial: torch.Tensor
    denoiser: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    constrain: Callable[[torch.Tensor], torch.Tensor]
    transition_generator: torch.Generator | Sequence[torch.Generator | None] | None
    guidance_randomness: Any
    reference_prediction: torch.Tensor
    current_states: torch.Tensor
    policy_context: PlannerPolicyContext


class PretrainedDiffusionPlanner(nn.Module):
    """Assemble model encoding, sampling, and optional reference guidance."""

    def __init__(
        self,
        config: OfficialDiffusionPlannerConfig,
        model: DiffusionPlanner,
        sampler_config: SamplerConfig,
        guidance_config: GuidanceConfig | None = None,
    ) -> None:
        super().__init__()
        selected_guidance = guidance_config or NoGuidanceConfig()
        self.config = config
        self.model = model
        self.sampler_config = sampler_config
        self.guidance_config = selected_guidance
        self._sampler = DiffusionSampler(sampler_config)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.train(False)

    def forward(
        self,
        observation: Mapping[str, torch.Tensor],
        standard_normal_noise: torch.Tensor,
        transition_generator: torch.Generator | Sequence[torch.Generator | None] | None = None,
        guidance_action: torch.Tensor | None = None,
    ) -> PlannerInferenceResult:
        """Generate the reference trajectory and, when configured, a guided trajectory."""

        batch = observation["ego_current_state"].shape[0]
        participants = 1 + self.config.predicted_neighbor_num
        inputs = self.config.observation_normalizer(observation)
        encoding = self.model.encode(inputs)
        route_encoding = self.model.encode_route(inputs)
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
                sample, timestep, encoding, route_encoding, neighbor_current_mask
            )
            if self.sampler_config.name == "ddim5":
                return prediction.to(dtype=sample.dtype)
            return prediction

        def constrain(sample: torch.Tensor) -> torch.Tensor:
            constrained = sample.reshape(batch, participants, self.config.future_len + 1, 4).clone()
            constrained[:, :, 0] = current_states
            return constrained.reshape(batch, participants, -1)

        guidance_randomness = (
            self._sampler.prepare_guidance_randomness(initial, transition_generator)
            if isinstance(
                self.guidance_config,
                (OrthogonalReferenceGuidanceConfig, OrthogonalPolicyGuidanceConfig),
            )
            or isinstance(transition_generator, Sequence)
            else None
        )
        normalized_sample = self._sampler.sample(
            initial,
            denoiser,
            constrain,
            transition_generator,
            guidance_randomness=guidance_randomness,
        )
        prediction = self._prediction(normalized_sample, batch, participants)
        if isinstance(self.guidance_config, NoGuidanceConfig):
            return PlannerInferenceResult(prediction=prediction)
        return self._run_guided(
            initial,
            denoiser,
            constrain,
            transition_generator,
            guidance_randomness,
            prediction,
            current_states,
            guidance_action,
        )

    def prepare_policy_guidance(
        self,
        observation: Mapping[str, torch.Tensor],
        standard_normal_noise: torch.Tensor,
        transition_generator: torch.Generator | Sequence[torch.Generator | None] | None,
    ) -> PreparedPolicyGuidance:
        """Prepare one shared-encoding reference pass for a learned guidance action."""

        batch = observation["ego_current_state"].shape[0]
        participants = 1 + self.config.predicted_neighbor_num
        inputs = self.config.observation_normalizer(observation)
        features = self.model.encode_policy_features(inputs)
        encoding = features["scene_tokens"]
        route_encoding = features["route_encoding"]
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
                sample, timestep, encoding, route_encoding, neighbor_current_mask
            )
            return prediction.to(dtype=sample.dtype)

        def constrain(sample: torch.Tensor) -> torch.Tensor:
            constrained = sample.reshape(batch, participants, self.config.future_len + 1, 4).clone()
            constrained[:, :, 0] = current_states
            return constrained.reshape(batch, participants, -1)

        guidance_randomness = self._sampler.prepare_guidance_randomness(
            initial, transition_generator
        )
        reference_prediction = self._prediction(
            self._sampler.sample(
                initial,
                denoiser,
                constrain,
                transition_generator,
                guidance_randomness=guidance_randomness,
            ),
            batch,
            participants,
        )
        return PreparedPolicyGuidance(
            initial=initial,
            denoiser=denoiser,
            constrain=constrain,
            transition_generator=transition_generator,
            guidance_randomness=guidance_randomness,
            reference_prediction=reference_prediction,
            current_states=current_states,
            policy_context=PlannerPolicyContext(
                scene_tokens=features["scene_tokens"],
                scene_padding_mask=features["scene_padding_mask"],
                navigation_tokens=features["navigation_tokens"],
                navigation_padding_mask=features["navigation_padding_mask"],
                reference_trajectory=reference_prediction[:, 0],
            ),
        )

    def complete_policy_guidance(
        self,
        prepared: PreparedPolicyGuidance,
        guidance_action: torch.Tensor,
    ) -> PlannerInferenceResult:
        """Finish a prepared learned-guidance pass with the sampled policy action."""

        return self._run_guided(
            prepared.initial,
            prepared.denoiser,
            prepared.constrain,
            prepared.transition_generator,
            prepared.guidance_randomness,
            prepared.reference_prediction,
            prepared.current_states,
            guidance_action,
        )

    @property
    def runtime_device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _prediction(self, sample: torch.Tensor, batch: int, participants: int) -> torch.Tensor:
        normalized = sample.reshape(batch, participants, self.config.future_len + 1, 4)
        return self.config.state_normalizer.inverse(normalized)[:, :, 1:]

    def _run_guided(
        self,
        initial: torch.Tensor,
        denoiser: Any,
        constrain: Any,
        transition_generator: torch.Generator | Sequence[torch.Generator | None] | None,
        guidance_randomness: Any,
        reference_prediction: torch.Tensor,
        current_states: torch.Tensor,
        guidance_action: torch.Tensor | None,
    ) -> PlannerInferenceResult:
        config = self.guidance_config
        batch, participants, _ = initial.shape
        device = initial.device
        if guidance_action is None:
            action = (
                torch.tensor(config.fixed_action, dtype=torch.float32, device=device)
                .expand(batch, -1)
                .clone()
            )
        else:
            action = guidance_action
        if torch.count_nonzero(action).item() == 0:
            return PlannerInferenceResult(
                prediction=reference_prediction,
                reference_prediction=reference_prediction,
                guidance_action=action,
                guidance_diagnostics=zero_guidance_diagnostics(
                    config,
                    action,
                    future_len=self.config.future_len,
                    num_steps=self._sampler.num_steps,
                ),
            )
        guidance = OrthogonalGuidance(config, self.config.state_normalizer)

        def guidance_callback(sample: torch.Tensor, predicted_x_start: torch.Tensor) -> Any:
            return guidance.gradient(
                sample,
                predicted_x_start,
                reference_prediction,
                current_states,
                action,
            )

        guided_result = self._sampler.sample_guided(
            initial,
            denoiser,
            constrain,
            transition_generator,
            guidance_callback,
            gradient_step_coefficient=config.gradient_step_coefficient,
            guidance_randomness=guidance_randomness,
        )
        guided_prediction = self._prediction(guided_result.sample, batch, participants)
        return PlannerInferenceResult(
            prediction=guided_prediction,
            reference_prediction=reference_prediction,
            guidance_action=action,
            guidance_diagnostics=stack_guidance_diagnostics(
                config,
                action,
                guided_result.diagnostics,
                guidance.longitudinal_target_speed_delta_mps(
                    reference_prediction, current_states, action
                ),
            ),
        )


def load_official_diffusion_planner(
    args_path: Path,
    checkpoint_path: Path,
    sampler_config: SamplerConfig,
    guidance_config: GuidanceConfig | None = None,
) -> tuple[PretrainedDiffusionPlanner, CheckpointLoadReport]:
    config = OfficialDiffusionPlannerConfig.from_json(args_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = extract_official_ema_state_dict(checkpoint)
    model = DiffusionPlanner(config)
    model.load_state_dict(state_dict, strict=True)
    planner = PretrainedDiffusionPlanner(
        config,
        model,
        sampler_config,
        guidance_config or NoGuidanceConfig(),
    )
    return planner, CheckpointLoadReport(
        ema_tensor_count=len(state_dict),
        parameter_count=sum(value.numel() for value in state_dict.values()),
    )
