"""Strict official-checkpoint loading and inference facade."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from eco_planner.models.baseline_sampler import BaselineDpmSampler
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
from eco_planner.models.ddim_sampler import DdimSampler
from eco_planner.models.diffusion_planner import DiffusionPlanner
from eco_planner.models.sampling_config import (
    Ddim5SamplerConfig,
    SamplerConfig,
)


class PretrainedDiffusionPlanner(nn.Module):
    """Frozen, official-EMA model with deterministic baseline sampling."""

    def __init__(
        self,
        config: OfficialDiffusionPlannerConfig,
        model: DiffusionPlanner,
        sampler_config: SamplerConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = model
        self.sampler_config = sampler_config
        self._sampler = (
            DdimSampler()
            if isinstance(sampler_config, Ddim5SamplerConfig)
            else BaselineDpmSampler()
        )
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
    ) -> torch.Tensor:
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
        initial_noise_scale = (
            self.sampler_config.initial_noise_scale
            if isinstance(self.sampler_config, Ddim5SamplerConfig)
            else 0.5
        )
        initial = torch.cat(
            [current_states[:, :, None], initial_noise_scale * standard_normal_noise], dim=2
        ).reshape(batch, participants, -1)

        def denoiser(sample: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
            prediction = self.model.denoise(
                sample, timestep, encoding, inputs["route_lanes"], neighbor_current_mask
            )
            if isinstance(self.sampler_config, Ddim5SamplerConfig):
                prediction = prediction.to(dtype=sample.dtype)
            return prediction

        def constrain(sample: torch.Tensor) -> torch.Tensor:
            constrained = sample.reshape(batch, participants, self.config.future_len + 1, 4)
            constrained = constrained.clone()
            constrained[:, :, 0] = current_states
            return constrained.reshape(batch, participants, -1)

        if isinstance(self.sampler_config, Ddim5SamplerConfig):
            timesteps = torch.tensor(
                self.sampler_config.timesteps,
                dtype=initial.dtype,
                device=initial.device,
            )
            normalized_sample = self._sampler.sample(
                initial,
                denoiser,
                constrain,
                timesteps,
                self.sampler_config.num_steps,
                self.sampler_config.ddim_stochasticity,
                transition_generator,
            )
        else:
            normalized_sample = self._sampler.sample(initial, denoiser, constrain)
        normalized = normalized_sample.reshape(batch, participants, self.config.future_len + 1, 4)
        return self.config.state_normalizer.inverse(normalized)[:, :, 1:]


def load_official_diffusion_planner(
    args_path: Path,
    checkpoint_path: Path,
    sampler_config: SamplerConfig,
) -> tuple[PretrainedDiffusionPlanner, CheckpointLoadReport]:
    """Load the pinned official EMA checkpoint without compatibility fallbacks."""

    config = OfficialDiffusionPlannerConfig.from_json(args_path)
    checkpoint: Any = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = extract_official_ema_state_dict(checkpoint)
    model = DiffusionPlanner(config)
    model.load_state_dict(state_dict, strict=True)
    planner = PretrainedDiffusionPlanner(config, model, sampler_config)
    return planner, CheckpointLoadReport(
        ema_tensor_count=OFFICIAL_EMA_TENSOR_COUNT,
        parameter_count=OFFICIAL_PARAMETER_COUNT,
    )
