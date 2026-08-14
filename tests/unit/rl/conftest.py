from __future__ import annotations

import pytest
import torch

from eco_planner.rl.config import ExplorationPolicyConfig
from eco_planner.rl.policy import ExplorationPolicyContext


@pytest.fixture
def exploration_policy_config() -> ExplorationPolicyConfig:
    return ExplorationPolicyConfig(
        name="exploration_beta",
        hidden_dim=12,
        reference_horizon=80,
        reference_state_dim=4,
        reference_mixer_depth=2,
        reference_token_mlp_hidden_dim=16,
        reference_channel_mlp_hidden_dim=24,
        cross_attention_heads=3,
        cross_attention_dropout=0.0,
        fusion_mlp_depth=2,
        fusion_hidden_dim=16,
        initial_concentration=2.0,
        minimum_concentration=1e-4,
    )


@pytest.fixture
def exploration_policy_context() -> ExplorationPolicyContext:
    batch = 3
    scene = torch.linspace(-1.0, 1.0, batch * 5 * 12).reshape(batch, 5, 12)
    navigation = torch.linspace(1.0, -1.0, batch * 2 * 12).reshape(batch, 2, 12)
    reference = torch.zeros((batch, 80, 4), dtype=torch.float32)
    reference[..., 0] = torch.arange(1, 81, dtype=torch.float32) * 0.5
    reference[..., 2] = 1.0
    return ExplorationPolicyContext(
        scene_tokens=scene,
        scene_padding_mask=torch.tensor(
            [[False, False, True, True, True]] * batch, dtype=torch.bool
        ),
        navigation_tokens=navigation,
        navigation_padding_mask=torch.tensor([[False, True]] * batch, dtype=torch.bool),
        reference_trajectory=reference,
    )
