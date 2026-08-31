from __future__ import annotations

import torch

from eco_planner.rl.policy import (
    ExplorationPolicy,
    ExplorationPolicyConfig,
    ExplorationPolicyContext,
)


def _config() -> ExplorationPolicyConfig:
    return ExplorationPolicyConfig(
        hidden_dim=12,
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


def _context() -> ExplorationPolicyContext:
    return ExplorationPolicyContext(
        scene_tokens=torch.zeros((2, 2, 12)),
        scene_padding_mask=torch.zeros((2, 2), dtype=torch.bool),
        navigation_tokens=torch.zeros((2, 1, 12)),
        navigation_padding_mask=torch.zeros((2, 1), dtype=torch.bool),
        reference_trajectory=torch.zeros((2, 80, 4)),
    )


def test_beta_policy_action_and_log_prob_are_positive_and_finite() -> None:
    with torch.random.fork_rng():
        torch.manual_seed(0)
        policy = ExplorationPolicy(_config())
    output, action = policy.act(_context(), "sample", torch.Generator().manual_seed(23))

    assert output.distribution.parameters.alpha.shape == (2, 2)
    assert output.distribution.parameters.beta.shape == (2, 2)
    assert torch.all(output.distribution.parameters.alpha > 0.0)
    assert torch.all(output.distribution.parameters.beta > 0.0)
    assert torch.all((action.base_action > 0.0) & (action.base_action < 1.0))
    assert torch.all((action.guidance_action > -1.0) & (action.guidance_action < 1.0))
    assert all(
        torch.isfinite(value).all()
        for value in (
            output.value,
            action.joint_guidance_log_prob,
            action.joint_guidance_entropy,
        )
    )

    generator = torch.Generator().manual_seed(23)
    replay_state = generator.get_state().clone()
    global_state = torch.random.get_rng_state().clone()
    _, first = policy.act(_context(), "sample", generator)
    consumed_state = generator.get_state().clone()
    replay_generator = torch.Generator()
    replay_generator.set_state(replay_state)
    _, replay = policy.act(_context(), "sample", replay_generator)

    torch.testing.assert_close(first.base_action, replay.base_action, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(first.guidance_action, replay.guidance_action, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(
        first.joint_guidance_log_prob,
        replay.joint_guidance_log_prob,
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        first.joint_guidance_entropy,
        replay.joint_guidance_entropy,
        rtol=1e-6,
        atol=1e-6,
    )
    assert torch.equal(consumed_state, replay_generator.get_state())
    assert torch.equal(global_state, torch.random.get_rng_state())


def test_deterministic_policy_mean_does_not_consume_any_rng_state() -> None:
    policy = ExplorationPolicy(_config())
    global_state = torch.random.get_rng_state().clone()

    output, action = policy.act(_context(), "mean")

    torch.testing.assert_close(action.guidance_action, output.distribution.mean)
    assert torch.equal(global_state, torch.random.get_rng_state())
