from __future__ import annotations

import pytest
import torch

from eco_planner.models import Ddim5SamplerConfig
from eco_planner.models.planner import load_official_diffusion_planner
from eco_planner.rl import ExplorationPolicy, ExplorationPolicyConfig
from eco_planner.rl.policy import ExplorationPolicyContext


@pytest.mark.slow
def test_real_checkpoint_features_feed_exploration_policy(
    baseline_checkpoint_dir,
    baseline_observation: dict[str, torch.Tensor],
) -> None:
    sampler = Ddim5SamplerConfig(
        name="ddim5",
        num_steps=5,
        timesteps=(1.0, 0.8, 0.6, 0.4, 0.2, 0.0),
        initial_noise_scale=1.0,
        ddim_stochasticity=0.0,
        parity_label="plannerrft_paper_text",
    )
    planner, _ = load_official_diffusion_planner(
        baseline_checkpoint_dir / "args.json",
        baseline_checkpoint_dir / "model.pth",
        sampler,
    )
    normalized = planner.config.observation_normalizer(baseline_observation)
    noise = torch.randn((1, 11, 80, 4), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        prepared = planner.model.prepare_policy_guidance(
            normalized, noise, torch.Generator().manual_seed(1)
        )
    context = ExplorationPolicyContext(
        scene_tokens=prepared.policy_context.scene_tokens,
        scene_padding_mask=prepared.policy_context.scene_padding_mask,
        navigation_tokens=prepared.policy_context.navigation_tokens,
        navigation_padding_mask=prepared.policy_context.navigation_padding_mask,
        reference_trajectory=prepared.policy_context.reference_trajectory,
    )
    config = ExplorationPolicyConfig(
        name="exploration_beta",
        hidden_dim=planner.config.hidden_dim,
        reference_horizon=80,
        reference_state_dim=4,
        reference_mixer_depth=2,
        reference_token_mlp_hidden_dim=128,
        reference_channel_mlp_hidden_dim=384,
        cross_attention_heads=6,
        cross_attention_dropout=0.0,
        fusion_mlp_depth=2,
        fusion_hidden_dim=256,
        initial_concentration=2.0,
        minimum_concentration=1e-4,
    )
    output, action = ExplorationPolicy(config).act(context, "mean")

    assert output.value.shape == (1,)
    assert action.base_action.shape == (1, 2)
    assert torch.equal(action.guidance_action, torch.zeros((1, 2)))
    assert torch.isfinite(action.joint_guidance_entropy).all()
