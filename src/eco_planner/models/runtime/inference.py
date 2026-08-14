"""Sampling orchestration for a loaded Diffusion Planner network."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from eco_planner.models.checkpoint.config import OfficialDiffusionPlannerConfig
from eco_planner.models.diffusion.model import DiffusionPlanner
from eco_planner.models.guidance import (
    GuidanceConfig,
    GuidanceDiagnostics,
    NoGuidanceConfig,
    OrthogonalGuidance,
    OrthogonalReferenceGuidanceConfig,
    validate_guidance_action,
)
from eco_planner.models.guidance.diagnostics import (
    stack_guidance_diagnostics,
    zero_guidance_diagnostics,
)
from eco_planner.models.runtime.validation import (
    validate_official_observation,
    validate_standard_normal_noise,
)
from eco_planner.models.sampling.config import SamplerConfig
from eco_planner.models.sampling.planner import PlanningSampler


@dataclass
class PlannerInferenceResult:
    """Validated planner prediction and optional reference-guidance audit values."""

    prediction: torch.Tensor
    reference_prediction: torch.Tensor | None = None
    guidance_action: torch.Tensor | None = None
    guidance_diagnostics: GuidanceDiagnostics | None = None


class DiffusionInferenceEngine:
    """Assemble model encoding, sampling, and optional reference guidance."""

    def __init__(
        self,
        config: OfficialDiffusionPlannerConfig,
        model: DiffusionPlanner,
        sampler_config: SamplerConfig,
        guidance_config: GuidanceConfig,
    ) -> None:
        self._config = config
        self._model = model
        self._sampler_config = sampler_config
        self._guidance_config = guidance_config
        self._sampler = PlanningSampler(sampler_config)

    def run(
        self,
        observation: Mapping[str, torch.Tensor],
        standard_normal_noise: torch.Tensor,
        transition_generator: torch.Generator | None,
        guidance_action: torch.Tensor | None,
    ) -> PlannerInferenceResult:
        """Generate the reference trajectory and, when configured, a guided trajectory."""

        device = self.runtime_device
        batch = validate_official_observation(observation, device, self._config)
        participants = 1 + self._config.predicted_neighbor_num
        validate_standard_normal_noise(
            standard_normal_noise,
            batch=batch,
            participants=participants,
            future_len=self._config.future_len,
            device=device,
        )
        inputs = self._config.observation_normalizer(observation)
        encoding = self._model.encode(inputs)
        ego_current = inputs["ego_current_state"][:, None, :4]
        neighbors_current = inputs["neighbor_agents_past"][
            :, : self._config.predicted_neighbor_num, -1, :4
        ]
        neighbor_current_mask = torch.sum(torch.ne(neighbors_current, 0), dim=-1) == 0
        current_states = torch.cat([ego_current, neighbors_current], dim=1)
        initial = torch.cat(
            [current_states[:, :, None], self._sampler.initial_noise_scale * standard_normal_noise],
            dim=2,
        ).reshape(batch, participants, -1)

        def denoiser(sample: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
            prediction = self._model.denoise(
                sample, timestep, encoding, inputs["route_lanes"], neighbor_current_mask
            )
            if self._sampler_config.name == "ddim5":
                return prediction.to(dtype=sample.dtype)
            return prediction

        def constrain(sample: torch.Tensor) -> torch.Tensor:
            constrained = sample.reshape(
                batch, participants, self._config.future_len + 1, 4
            ).clone()
            constrained[:, :, 0] = current_states
            return constrained.reshape(batch, participants, -1)

        guidance_randomness = (
            self._sampler.prepare_guidance_randomness(initial, transition_generator)
            if isinstance(self._guidance_config, OrthogonalReferenceGuidanceConfig)
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
        if isinstance(self._guidance_config, NoGuidanceConfig):
            if guidance_action is not None:
                raise ValueError("guidance_action requires active guidance configuration")
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

    @property
    def runtime_device(self) -> torch.device:
        try:
            return next(self._model.parameters()).device
        except StopIteration as error:
            raise RuntimeError("Diffusion Planner must contain parameters") from error

    def _prediction(self, sample: torch.Tensor, batch: int, participants: int) -> torch.Tensor:
        normalized = sample.reshape(batch, participants, self._config.future_len + 1, 4)
        return self._config.state_normalizer.inverse(normalized)[:, :, 1:]

    def _run_guided(
        self,
        initial: torch.Tensor,
        denoiser: Any,
        constrain: Any,
        transition_generator: torch.Generator | None,
        guidance_randomness: Any,
        reference_prediction: torch.Tensor,
        current_states: torch.Tensor,
        guidance_action: torch.Tensor | None,
    ) -> PlannerInferenceResult:
        config = self._guidance_config
        if not isinstance(config, OrthogonalReferenceGuidanceConfig):
            raise RuntimeError("unsupported guidance configuration")
        batch, participants, _ = initial.shape
        device = initial.device
        action = (
            torch.tensor(config.fixed_action, dtype=torch.float32, device=device)
            .expand(batch, -1)
            .clone()
            if guidance_action is None
            else guidance_action
        )
        validate_guidance_action(action, batch=batch, device=device)
        if torch.count_nonzero(action).item() == 0:
            return PlannerInferenceResult(
                prediction=reference_prediction,
                reference_prediction=reference_prediction,
                guidance_action=action,
                guidance_diagnostics=zero_guidance_diagnostics(
                    config,
                    action,
                    future_len=self._config.future_len,
                    num_steps=self._sampler.num_steps,
                ),
            )
        guidance = OrthogonalGuidance(config, self._config.state_normalizer)

        def guidance_callback(sample: torch.Tensor, predicted_x_start: torch.Tensor) -> Any:
            return guidance.gradient(
                sample,
                predicted_x_start,
                reference_prediction,
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
