"""Regression tests for the affine-Beta interior concentration parameterization."""

from __future__ import annotations

import pytest
import torch

from eco_planner.planning.policy import (
    ExplorationPolicy,
    ExplorationPolicyConfig,
    ExplorationPolicyContext,
)


def _config_kwargs() -> dict:
    return dict(
        hidden_dim=12,
        reference_mixer_depth=2,
        reference_token_mlp_hidden_dim=16,
        reference_channel_mlp_hidden_dim=24,
        cross_attention_heads=3,
        cross_attention_dropout=0.0,
        fusion_mlp_depth=2,
        fusion_hidden_dim=16,
        initial_concentration=2.0,
        interior_epsilon=1e-4,
    )


def _context() -> ExplorationPolicyContext:
    return ExplorationPolicyContext(
        scene_tokens=torch.zeros((1, 2, 12)),
        scene_padding_mask=torch.zeros((1, 2), dtype=torch.bool),
        navigation_tokens=torch.zeros((1, 1, 12)),
        navigation_padding_mask=torch.zeros((1, 1), dtype=torch.bool),
        reference_trajectory=torch.zeros((1, 80, 4)),
    )


def _forward_concentrations(
    policy: ExplorationPolicy,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    context = _context()
    alpha, beta, state_value = policy.forward_tensors(
        context.scene_tokens,
        context.scene_padding_mask,
        context.navigation_tokens,
        context.navigation_padding_mask,
        context.reference_trajectory,
    )
    return alpha, beta, state_value


def test_symmetric_initialization_hits_exact_initial_concentration_with_zero_mean() -> None:
    with torch.random.fork_rng():
        torch.manual_seed(0)
        policy = ExplorationPolicy(ExplorationPolicyConfig(**_config_kwargs()))

    alpha, beta, state_value = _forward_concentrations(policy)
    output = policy.output_from_tensors(alpha, beta, state_value)

    assert torch.allclose(alpha, torch.full_like(alpha, 2.0), rtol=1e-5, atol=1e-5)
    assert torch.allclose(beta, torch.full_like(beta, 2.0), rtol=1e-5, atol=1e-5)
    guidance_mean = output.distribution.mean
    assert torch.allclose(guidance_mean, torch.zeros_like(guidance_mean), atol=1e-6)


def test_extreme_raw_parameters_stay_inside_the_interior_region() -> None:
    """E-056 regression: never reach the singular boundary regime alpha or beta < 1."""

    config = ExplorationPolicyConfig(**_config_kwargs())
    with torch.random.fork_rng():
        torch.manual_seed(0)
        policy = ExplorationPolicy(config)
    with torch.no_grad():
        policy.actor_head.weight.zero_()
        policy.actor_head.bias.fill_(-1.0e4)

    alpha, beta, _ = _forward_concentrations(policy)
    floor = 1.0 + config.interior_epsilon

    assert torch.isfinite(alpha).all()
    assert torch.isfinite(beta).all()
    assert torch.all(alpha > 1.0)
    assert torch.all(beta > 1.0)
    assert torch.all(alpha >= floor - 1e-7)
    assert torch.all(beta >= floor - 1e-7)


@pytest.mark.parametrize("initial_concentration", [1.0, 1.0 + 1e-4])
def test_config_rejects_initial_concentration_at_or_below_the_interior_floor(
    initial_concentration: float,
) -> None:
    kwargs = _config_kwargs()
    kwargs["initial_concentration"] = initial_concentration

    with pytest.raises(ValueError, match="must exceed 1.0 \\+ interior_epsilon"):
        ExplorationPolicyConfig(**kwargs)
