"""Adapter from neutral diffusion representations to exploration policy inputs."""

from __future__ import annotations

import torch

from eco_planner.planning.diffusion import DiffusionRepresentations
from eco_planner.planning.policy.model import ExplorationPolicyContext


def build_policy_inputs(
    representations: DiffusionRepresentations,
    reference_prediction: torch.Tensor,
) -> ExplorationPolicyContext:
    """Map one diffusion encoding and ego reference to the policy input contract."""

    return ExplorationPolicyContext(
        scene_tokens=representations.scene_tokens,
        scene_padding_mask=representations.scene_padding_mask,
        navigation_tokens=representations.navigation_tokens,
        navigation_padding_mask=representations.navigation_padding_mask,
        reference_trajectory=reference_prediction[:, 0],
    )
