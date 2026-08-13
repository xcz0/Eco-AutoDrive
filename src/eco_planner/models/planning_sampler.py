"""Unified facade for the planner's fixed sampler profiles."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from eco_planner.models.diffusers_sampler import DiffusersDdimSampler, DiffusersDpmSampler
from eco_planner.models.guidance import GuidanceGradientResult
from eco_planner.models.sampling_config import Ddim5SamplerConfig, SamplerConfig
from eco_planner.models.sampling_contracts import DdimGuidedSampleResult


@dataclass(frozen=True)
class GuidanceSamplingRandomness:
    """Transition randomness shared by the reference and guided DDIM passes."""

    variance_noises: tuple[torch.Tensor, ...] | None = None


class PlanningSampler:
    """Select and invoke one configured sampler without exposing its backend API."""

    def __init__(self, config: SamplerConfig) -> None:
        self.config = config
        self._sampler = self._new_implementation(config)

    @property
    def initial_noise_scale(self) -> float:
        """Return the fixed initial future-noise scale for this sampler profile."""

        return (
            self.config.initial_noise_scale if isinstance(self.config, Ddim5SamplerConfig) else 0.5
        )

    @property
    def num_steps(self) -> int:
        """Return the number of denoising transitions represented by this profile."""

        return self.config.num_steps if isinstance(self.config, Ddim5SamplerConfig) else 10

    def prepare_guidance_randomness(
        self,
        initial_sample: torch.Tensor,
        generator: torch.Generator | None,
    ) -> GuidanceSamplingRandomness:
        """Capture one DDIM random stream for semantically identical paired passes."""

        config = self._ddim_config()
        if config.ddim_stochasticity == 0.0:
            return GuidanceSamplingRandomness()
        if generator is None:
            raise ValueError("stochastic DDIM sampling requires an explicit torch.Generator")
        draws = tuple(
            torch.randn(
                initial_sample.shape,
                dtype=initial_sample.dtype,
                device=initial_sample.device,
                generator=generator,
            )
            for _ in range(config.num_steps - 1)
        )
        return GuidanceSamplingRandomness(
            variance_noises=(*draws, torch.zeros_like(initial_sample))
        )

    def sample(
        self,
        initial_sample: torch.Tensor,
        model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        current_state_constraint: Callable[[torch.Tensor], torch.Tensor],
        generator: torch.Generator | None = None,
        *,
        guidance_randomness: GuidanceSamplingRandomness | None = None,
    ) -> torch.Tensor:
        """Run this profile's unguided sampling pass."""

        if isinstance(self.config, Ddim5SamplerConfig):
            timesteps = torch.tensor(
                self.config.timesteps,
                dtype=initial_sample.dtype,
                device=initial_sample.device,
            )
            return self._sampler.sample(
                initial_sample,
                model,
                current_state_constraint,
                timesteps,
                self.config.num_steps,
                self.config.ddim_stochasticity,
                generator,
                variance_noises=(
                    None if guidance_randomness is None else guidance_randomness.variance_noises
                ),
            )
        return self._sampler.sample(initial_sample, model, current_state_constraint)

    def sample_guided(
        self,
        initial_sample: torch.Tensor,
        model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        current_state_constraint: Callable[[torch.Tensor], torch.Tensor],
        generator: torch.Generator | None,
        guidance: Callable[[torch.Tensor, torch.Tensor], GuidanceGradientResult],
        *,
        gradient_step_coefficient: float,
        guidance_randomness: GuidanceSamplingRandomness,
    ) -> DdimGuidedSampleResult:
        """Run the configured DDIM profile with the project's guidance policy."""

        config = self._ddim_config()
        timesteps = torch.tensor(
            config.timesteps,
            dtype=initial_sample.dtype,
            device=initial_sample.device,
        )
        return self._sampler.sample_guided(
            initial_sample,
            model,
            current_state_constraint,
            timesteps,
            config.num_steps,
            config.ddim_stochasticity,
            generator,
            guidance,
            gradient_step_coefficient=gradient_step_coefficient,
            variance_noises=guidance_randomness.variance_noises,
        )

    @staticmethod
    def _new_implementation(
        config: SamplerConfig,
    ) -> DiffusersDdimSampler | DiffusersDpmSampler:
        if isinstance(config, Ddim5SamplerConfig):
            return DiffusersDdimSampler()
        return DiffusersDpmSampler()

    def _ddim_config(self) -> Ddim5SamplerConfig:
        if not isinstance(self.config, Ddim5SamplerConfig):
            raise RuntimeError("guidance is only supported by the DDIM-5 sampler profile")
        return self.config
