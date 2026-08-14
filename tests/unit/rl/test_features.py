from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import torch

from eco_planner.models.config import OfficialDiffusionPlannerConfig
from eco_planner.models.network import DiffusionPlanner
from eco_planner.rl import (
    ExplorationPolicy,
    ExplorationPolicyConfig,
    FrozenPlannerPolicyFeatureExtractor,
)


def test_frozen_planner_features_encode_once_and_remain_unchanged_after_backward(
    official_model_config: OfficialDiffusionPlannerConfig,
    stage0_observation: dict[str, torch.Tensor],
    exploration_policy_config: ExplorationPolicyConfig,
) -> None:
    planner = DiffusionPlanner(official_model_config).eval()
    planner.requires_grad_(False)
    before = {name: value.detach().clone() for name, value in planner.state_dict().items()}
    extractor = FrozenPlannerPolicyFeatureExtractor(planner)
    normalized = official_model_config.observation_normalizer(stage0_observation)
    reference = torch.zeros((1, 80, 4), dtype=torch.float32)
    reference[..., 0] = torch.arange(1, 81, dtype=torch.float32)
    reference[..., 2] = 1.0

    with patch.object(
        planner, "encode_policy_features", wraps=planner.encode_policy_features
    ) as encode:
        context = extractor(normalized, reference)
    policy = ExplorationPolicy(
        replace(exploration_policy_config, hidden_dim=192, cross_attention_heads=6)
    )
    output = policy(context)
    output.value.sum().backward()

    assert encode.call_count == 1
    assert context.scene_tokens.shape == (1, 107, 192)
    assert context.scene_padding_mask.shape == (1, 107)
    assert context.navigation_tokens.shape == (1, 1, 192)
    assert context.navigation_padding_mask.shape == (1, 1)
    assert all(not value.requires_grad for value in before.values())
    assert all(parameter.grad is None for parameter in planner.parameters())
    for name, value in planner.state_dict().items():
        assert torch.equal(value, before[name])
