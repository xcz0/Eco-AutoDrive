"""Frozen planner feature extraction for the exploration policy."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from eco_planner.models.diffusion.model import DiffusionPlanner
from eco_planner.rl.policy import ExplorationPolicyContext


class FrozenPlannerPolicyFeatureExtractor(nn.Module):
    """Expose frozen scene/navigation policy features without building an autograd graph."""

    def __init__(self, planner_model: DiffusionPlanner) -> None:
        super().__init__()
        if planner_model.training:
            raise ValueError("planner feature extractor requires an eval-mode planner")
        if any(parameter.requires_grad for parameter in planner_model.parameters()):
            raise ValueError(
                "planner feature extractor requires every planner parameter to be frozen"
            )
        object.__setattr__(self, "_planner_model", planner_model)

    def forward(
        self,
        normalized_observation: Mapping[str, torch.Tensor],
        reference_trajectory: torch.Tensor,
    ) -> ExplorationPolicyContext:
        planner_model: DiffusionPlanner = self._planner_model
        if planner_model.training:
            raise RuntimeError("frozen planner changed to training mode")
        if any(parameter.requires_grad for parameter in planner_model.parameters()):
            raise RuntimeError("frozen planner parameter became trainable")
        with torch.no_grad():
            features = planner_model.encode_policy_features(normalized_observation)
        return ExplorationPolicyContext(
            scene_tokens=features["scene_tokens"].detach(),
            scene_padding_mask=features["scene_padding_mask"].detach(),
            navigation_tokens=features["navigation_tokens"].detach(),
            navigation_padding_mask=features["navigation_padding_mask"].detach(),
            reference_trajectory=reference_trajectory.detach(),
        )
