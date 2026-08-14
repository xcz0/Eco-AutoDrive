"""Public pretrained-planner facade and official checkpoint loader."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from eco_planner.models.checkpoint.config import OfficialDiffusionPlannerConfig
from eco_planner.models.checkpoint.loader import (
    OFFICIAL_EMA_TENSOR_COUNT,
    OFFICIAL_PARAMETER_COUNT,
    CheckpointLoadReport,
    extract_official_ema_state_dict,
)
from eco_planner.models.diffusion.model import DiffusionPlanner
from eco_planner.models.guidance import GuidanceConfig, NoGuidanceConfig, validate_guidance_sampler
from eco_planner.models.runtime.inference import (
    DiffusionInferenceEngine,
    PlannerInferenceResult,
    PreparedPolicyGuidance,
)
from eco_planner.models.sampling.config import SamplerConfig


class PretrainedDiffusionPlanner(nn.Module):
    """Frozen official-EMA model with a separate inference orchestration engine."""

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
        self._engine = DiffusionInferenceEngine(config, model, sampler_config, self.guidance_config)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.train(False)

    def train(self, mode: bool = True) -> PretrainedDiffusionPlanner:
        if mode:
            raise RuntimeError("PretrainedDiffusionPlanner is frozen and cannot enter train mode")
        super().train(False)
        return self

    @property
    def _runtime_device(self) -> torch.device:
        """Expose the model device for existing runtime integrations."""

        return self._engine.runtime_device

    def forward(
        self,
        observation: Mapping[str, torch.Tensor],
        standard_normal_noise: torch.Tensor,
        transition_generator: torch.Generator | None = None,
        guidance_action: torch.Tensor | None = None,
    ) -> PlannerInferenceResult:
        if self.training or self.model.training:
            raise RuntimeError("PretrainedDiffusionPlanner must remain in eval mode")
        return self._engine.run(
            observation,
            standard_normal_noise,
            transition_generator,
            guidance_action,
        )

    def prepare_policy_guidance(
        self,
        observation: Mapping[str, torch.Tensor],
        standard_normal_noise: torch.Tensor,
        transition_generator: torch.Generator | None,
    ) -> PreparedPolicyGuidance:
        """Prepare a learned-guidance reference pass while retaining shared planner features."""

        return self._engine.prepare_policy_guidance(
            observation, standard_normal_noise, transition_generator
        )

    def complete_policy_guidance(
        self,
        prepared: PreparedPolicyGuidance,
        guidance_action: torch.Tensor,
    ) -> PlannerInferenceResult:
        """Complete a prepared learned-guidance pass with the policy action."""

        return self._engine.complete_policy_guidance(prepared, guidance_action)


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
