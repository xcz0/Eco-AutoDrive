from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch
from torch.distributions import Beta

from eco_planner.rl import (
    BetaGuidanceDistribution,
    BetaGuidanceParameters,
    ExplorationPolicy,
    ExplorationPolicyConfig,
    ExplorationPolicyContext,
)


def test_forward_returns_positive_finite_parameters_and_exact_value_shape(
    exploration_policy_config: ExplorationPolicyConfig,
    exploration_policy_context: ExplorationPolicyContext,
) -> None:
    policy = ExplorationPolicy(exploration_policy_config)
    output = policy(exploration_policy_context)

    assert output.parameters.alpha.shape == (3, 2)
    assert output.parameters.beta.shape == (3, 2)
    assert output.value.shape == (3,)
    assert output.value.dtype == exploration_policy_context.scene_tokens.dtype
    assert output.value.device == exploration_policy_context.scene_tokens.device
    assert output.parameters.alpha.dtype == exploration_policy_context.scene_tokens.dtype
    assert output.parameters.alpha.device == exploration_policy_context.scene_tokens.device
    assert torch.isfinite(output.parameters.alpha).all()
    assert torch.isfinite(output.parameters.beta).all()
    assert torch.all(output.parameters.alpha > 0.0)
    assert torch.all(output.parameters.beta > 0.0)


def test_symmetric_initialization_has_zero_guidance_mean_and_nonzero_variance(
    exploration_policy_config: ExplorationPolicyConfig,
    exploration_policy_context: ExplorationPolicyContext,
) -> None:
    output = ExplorationPolicy(exploration_policy_config)(exploration_policy_context)
    action = output.distribution.mean()
    alpha = output.parameters.alpha
    beta = output.parameters.beta
    guidance_variance = 4.0 * alpha * beta / ((alpha + beta).square() * (alpha + beta + 1.0))

    torch.testing.assert_close(alpha, beta, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        action.guidance_action, torch.zeros_like(action.guidance_action), rtol=0.0, atol=0.0
    )
    assert torch.all(guidance_variance > 0.0)


def test_joint_log_prob_and_entropy_include_exact_affine_jacobian() -> None:
    alpha = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
    beta = torch.tensor([[5.0, 4.0], [3.0, 2.0]])
    distribution = BetaGuidanceDistribution(BetaGuidanceParameters(alpha, beta))
    base_action = torch.tensor([[0.25, 0.75], [0.4, 0.6]])
    action = distribution.evaluate_base_action(base_action)
    expected_base_log_prob = Beta(alpha, beta).log_prob(base_action).sum(dim=-1)
    expected_base_entropy = Beta(alpha, beta).entropy().sum(dim=-1)

    torch.testing.assert_close(action.joint_base_log_prob, expected_base_log_prob)
    torch.testing.assert_close(
        action.joint_guidance_log_prob, expected_base_log_prob - 2.0 * math.log(2.0)
    )
    torch.testing.assert_close(
        action.joint_guidance_entropy, expected_base_entropy + 2.0 * math.log(2.0)
    )
    torch.testing.assert_close(action.guidance_action, 2.0 * base_action - 1.0)


def test_sample_and_rsample_semantics_use_explicit_replayable_rng_only(
    exploration_policy_config: ExplorationPolicyConfig,
    exploration_policy_context: ExplorationPolicyContext,
) -> None:
    policy = ExplorationPolicy(exploration_policy_config)
    output = policy(exploration_policy_context)
    global_state = torch.random.get_rng_state().clone()
    first = output.distribution.sample(torch.Generator().manual_seed(23))
    replay = output.distribution.sample(torch.Generator().manual_seed(23))
    different = output.distribution.sample(torch.Generator().manual_seed(24))

    assert torch.equal(first.base_action, replay.base_action)
    assert not torch.equal(first.base_action, different.base_action)
    assert torch.equal(torch.random.get_rng_state(), global_state)
    assert not first.base_action.requires_grad
    reparameterized = output.distribution.rsample(torch.Generator().manual_seed(23))
    assert reparameterized.base_action.requires_grad
    reparameterized.guidance_action.sum().backward()
    assert policy.actor_head.bias.grad is not None


def test_old_new_policy_can_recompute_the_same_stored_base_action(
    exploration_policy_config: ExplorationPolicyConfig,
    exploration_policy_context: ExplorationPolicyContext,
) -> None:
    old_policy = ExplorationPolicy(exploration_policy_config)
    new_policy = ExplorationPolicy(exploration_policy_config)
    new_policy.load_state_dict(old_policy.state_dict())
    stored = old_policy(exploration_policy_context).distribution.sample(
        torch.Generator().manual_seed(9)
    )
    with torch.no_grad():
        new_policy.actor_head.bias[0] += 0.5
    old_recomputed = old_policy(exploration_policy_context).distribution.evaluate_base_action(
        stored.base_action
    )
    new_recomputed = new_policy(exploration_policy_context).distribution.evaluate_base_action(
        stored.base_action
    )

    torch.testing.assert_close(old_recomputed.joint_base_log_prob, stored.joint_base_log_prob)
    assert not torch.equal(old_recomputed.joint_base_log_prob, new_recomputed.joint_base_log_prob)


@pytest.mark.parametrize("boundary", [0.0, 1.0])
def test_base_action_boundaries_are_rejected_without_clamping(boundary: float) -> None:
    parameters = BetaGuidanceParameters(torch.full((1, 2), 2.0), torch.full((1, 2), 2.0))
    with pytest.raises(ValueError, match="strictly inside"):
        BetaGuidanceDistribution(parameters).evaluate_base_action(torch.full((1, 2), boundary))


@pytest.mark.parametrize("boundary", [-1.0, 1.0])
def test_guidance_action_boundaries_are_rejected_without_clamping(boundary: float) -> None:
    parameters = BetaGuidanceParameters(torch.full((1, 2), 2.0), torch.full((1, 2), 2.0))
    with pytest.raises(ValueError, match="strictly inside"):
        BetaGuidanceDistribution(parameters).evaluate_guidance_action(torch.full((1, 2), boundary))


def test_beta_parameters_reject_nonpositive_or_nonfinite_values() -> None:
    valid = torch.full((1, 2), 2.0)
    for invalid in (torch.tensor([[0.0, 1.0]]), torch.tensor([[float("nan"), 1.0]])):
        with pytest.raises(ValueError):
            BetaGuidanceDistribution(BetaGuidanceParameters(invalid, valid))


@pytest.mark.parametrize(
    ("context_update", "error"),
    [
        ({"scene_tokens": torch.zeros((3, 5, 11))}, "scene tokens"),
        ({"scene_tokens": torch.full((3, 5, 12), float("nan"))}, "finite"),
        ({"navigation_tokens": torch.zeros((3, 2, 12), dtype=torch.float64)}, "dtype"),
        ({"reference_trajectory": torch.zeros((3, 79, 4))}, r"\[B, 80, 4\]"),
        ({"scene_padding_mask": torch.zeros((3, 5))}, "bool"),
        (
            {
                "scene_padding_mask": torch.ones((3, 5), dtype=torch.bool),
                "navigation_padding_mask": torch.ones((3, 2), dtype=torch.bool),
            },
            "at least one valid",
        ),
    ],
)
def test_policy_rejects_invalid_context(
    exploration_policy_config: ExplorationPolicyConfig,
    exploration_policy_context: ExplorationPolicyContext,
    context_update: dict[str, torch.Tensor],
    error: str,
) -> None:
    policy = ExplorationPolicy(exploration_policy_config)
    with pytest.raises((TypeError, ValueError), match=error):
        policy(replace(exploration_policy_context, **context_update))


def test_sampling_mode_requires_exact_generator_semantics(
    exploration_policy_config: ExplorationPolicyConfig,
    exploration_policy_context: ExplorationPolicyContext,
) -> None:
    policy = ExplorationPolicy(exploration_policy_config)
    with pytest.raises(ValueError, match="requires"):
        policy.act(exploration_policy_context, "sample")
    with pytest.raises(ValueError, match="must not"):
        policy.act(
            exploration_policy_context,
            "mean",
            torch.Generator().manual_seed(0),
        )
    with pytest.raises(ValueError, match="sampling must"):
        policy.act(exploration_policy_context, "mode")  # type: ignore[arg-type]


@pytest.mark.gpu
def test_policy_forward_and_sampling_on_cuda(
    exploration_policy_config: ExplorationPolicyConfig,
    exploration_policy_context: ExplorationPolicyContext,
) -> None:
    device = torch.device("cuda")
    policy = ExplorationPolicy(exploration_policy_config).to(device)
    context = ExplorationPolicyContext(
        scene_tokens=exploration_policy_context.scene_tokens.to(device),
        scene_padding_mask=exploration_policy_context.scene_padding_mask.to(device),
        navigation_tokens=exploration_policy_context.navigation_tokens.to(device),
        navigation_padding_mask=exploration_policy_context.navigation_padding_mask.to(device),
        reference_trajectory=exploration_policy_context.reference_trajectory.to(device),
    )
    output, action = policy.act(context, "rsample", torch.Generator(device=device).manual_seed(41))

    assert output.value.device.type == "cuda"
    assert action.base_action.device.type == "cuda"
    assert torch.isfinite(action.joint_guidance_log_prob).all()
