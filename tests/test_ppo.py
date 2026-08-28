from __future__ import annotations

import math

import numpy as np
import torch

from eco_planner.envs.metadrive.reward import MetaDriveBuiltinRewardAudit
from eco_planner.rl.optimization import PPOConfig, PPOUpdater, compute_episode_gae
from eco_planner.rl.policy import (
    ExplorationPolicy,
    ExplorationPolicyConfig,
    ExplorationPolicyContext,
)
from eco_planner.rl.rollout import (
    DecisionAudit,
    ExecutionTransitionAudit,
    RolloutEpisodeBuilder,
    RolloutProvenance,
    build_training_decision,
)


def _policy_config() -> ExplorationPolicyConfig:
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


def _ppo_config() -> PPOConfig:
    return PPOConfig(
        name="test",
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coefficient=0.5,
        entropy_coefficient=0.01,
        learning_rate=0.00025,
        adam_epsilon=1e-5,
        weight_decay=0.0,
        max_gradient_norm=0.5,
        epochs=1,
        batch_size=2,
        minibatch_size=2,
        minibatch_seed=7,
        scheduler_total_optimizer_steps=1,
        scheduler_minimum_learning_rate=0.0,
    )


def _context() -> ExplorationPolicyContext:
    return ExplorationPolicyContext(
        scene_tokens=torch.zeros((1, 2, 12)),
        scene_padding_mask=torch.zeros((1, 2), dtype=torch.bool),
        navigation_tokens=torch.zeros((1, 1, 12)),
        navigation_padding_mask=torch.zeros((1, 1), dtype=torch.bool),
        reference_trajectory=torch.zeros((1, 80, 4)),
    )


def _decision_audit() -> DecisionAudit:
    context = _context()
    return DecisionAudit(
        prediction=np.zeros((1, 11, 80, 4), dtype=np.float32),
        initial_noise=torch.zeros((1, 11, 80, 4)),
        policy_context=context,
        base_action=torch.tensor([[0.25, 0.75]]),
        guidance_action=torch.tensor([[-0.5, 0.5]]),
        old_joint_guidance_log_prob=torch.tensor([0.5]),
        old_value=torch.tensor([1.0]),
        beta_alpha=torch.full((1, 2), 2.0),
        beta_beta=torch.full((1, 2), 2.0),
        diffusion_rng_state=torch.ones(5, dtype=torch.uint8),
        policy_rng_state=torch.ones(5, dtype=torch.uint8),
    )


def _execution_audit(
    reward: float, *, terminated: bool, truncated: bool
) -> ExecutionTransitionAudit:
    return ExecutionTransitionAudit(
        reward=reward,
        dense_reward=reward,
        terminal_override=0.0,
        route_completion_delta=0.1,
        distance_m=1.0,
        speed_mps=2.0,
        stopped=False,
        position_error_m=0.0,
        heading_error_rad=0.0,
        arrive_dest=False,
        out_of_road=False,
        crash_vehicle=False,
        crash_object=False,
        crash_building=False,
        crash_human=False,
        crash_sidewalk=False,
        terminated=terminated,
        truncated=truncated,
        reward_audit=MetaDriveBuiltinRewardAudit(
            profile_name="metadrive_builtin_v1",
            reward_total=reward,
            dense_reward=reward,
            terminal_override=0.0,
            step_distance_m=1.0,
            native_step_energy_ml=0.0,
            native_episode_energy_ml=0.0,
            executed_fuel_proxy_step_energy_ml=0.05,
            executed_fuel_proxy_ml_per_km=50.0,
            energy_distance_valid=True,
        ),
    )


def _episode(*, reward: float, terminated: bool, truncated: bool, bootstrap: float):
    context = _context()
    decision = build_training_decision(
        context,
        torch.tensor([[-0.5, 0.5]]),
        torch.tensor([0.5]),
        torch.tensor([1.0]),
    )
    builder = RolloutEpisodeBuilder()
    builder.append(
        decision,
        _decision_audit(),
        _execution_audit(reward, terminated=terminated, truncated=truncated),
        RolloutProvenance(0, 1, 2, 0),
    )
    tail_kind = "terminated" if terminated else "truncated" if truncated else "rollout_limit"
    return builder.finish(tail_kind, torch.tensor([bootstrap]))


def test_gae_uses_terminal_and_truncated_bootstrap_semantics() -> None:
    config = _ppo_config()
    terminal = compute_episode_gae(
        _episode(reward=0.25, terminated=True, truncated=False, bootstrap=0.0), config
    )
    truncated = compute_episode_gae(
        _episode(reward=0.25, terminated=False, truncated=True, bootstrap=2.0), config
    )

    torch.testing.assert_close(terminal["advantage"], torch.tensor([[-0.75]]))
    torch.testing.assert_close(terminal["value_target"], torch.tensor([[0.25]]))
    torch.testing.assert_close(truncated["advantage"], torch.tensor([[1.23]]))
    torch.testing.assert_close(truncated["value_target"], torch.tensor([[2.23]]))
    assert torch.isfinite(truncated["advantage"]).all()
    assert torch.isfinite(truncated["value_target"]).all()


def test_ppo_update_changes_policy_and_reports_finite_metrics() -> None:
    with torch.random.fork_rng():
        torch.manual_seed(0)
        policy = ExplorationPolicy(_policy_config())
    policy_before = {name: value.detach().clone() for name, value in policy.state_dict().items()}

    report = PPOUpdater(policy, _ppo_config()).update(
        (
            _episode(reward=0.25, terminated=True, truncated=False, bootstrap=0.0),
            _episode(reward=2.0, terminated=True, truncated=False, bootstrap=0.0),
        )
    )

    assert any(
        not torch.equal(value, policy_before[name]) for name, value in policy.state_dict().items()
    )
    assert report.sample_count == 2
    metrics = (value for value in report.__dict__.values() if isinstance(value, float))
    assert all(math.isfinite(value) for value in metrics)
