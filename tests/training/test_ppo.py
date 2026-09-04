from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from eco_planner.rl.artifacts import TrainingUpdateSummary, build_update_summary
from eco_planner.rl.optimization import PPOConfig, PPOUpdater, compute_episode_gae
from eco_planner.rl.policy import (
    ExplorationPolicy,
    ExplorationPolicyConfig,
    ExplorationPolicyContext,
)
from eco_planner.rl.reward import RewardComponents, RewardDiagnostics, RewardResult
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
        target_kl=None,
        value_coefficient=0.5,
        entropy_coefficient=0.01,
        gradient_diagnostics=False,
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
        reward_result=RewardResult(
            profile_name="plannerrft_energy_v1",
            total=reward,
            base_total=reward,
            safety_gate=1.0,
            components=RewardComponents(1.0, 0.5, 1.0, 1.0, 0.5),
            diagnostics=RewardDiagnostics(
                collision_score=1.0,
                drivable_score=1.0,
                wrong_direction_score=1.0,
                has_ttc_candidate=False,
                min_ttc_s=10.0,
                route_progress_delta_m=1.0,
                speed_mps=2.0,
                speed_limit_mps=10.0,
                overspeed_mps=0.0,
                longitudinal_acceleration_mps2=0.0,
                lateral_acceleration_mps2=0.0,
                jerk_mps3=0.0,
                yaw_rate_radps=0.0,
                step_distance_m=1.0,
                native_step_energy_ml=0.0,
                native_episode_energy_ml=0.0,
                executed_fuel_proxy_step_energy_ml=0.05,
                executed_fuel_proxy_ml_per_km=50.0,
                energy_distance_valid=True,
            ),
        ),
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


def test_gae_treats_simultaneous_termination_and_truncation_as_terminal() -> None:
    episode = _episode(reward=0.25, terminated=True, truncated=True, bootstrap=0.0)

    assert episode.tail_kind == "terminated"
    assert bool(episode.training["next", "terminated"][-1].item())
    assert bool(episode.training["next", "truncated"][-1].item())
    torch.testing.assert_close(episode.tail_bootstrap_value, torch.zeros(1))

    trajectory = compute_episode_gae(episode, _ppo_config())
    torch.testing.assert_close(trajectory["advantage"], torch.tensor([[-0.75]]))
    torch.testing.assert_close(trajectory["value_target"], torch.tensor([[0.25]]))


@pytest.mark.smoke
def test_ppo_update_changes_policy_and_reports_finite_training_summary() -> None:
    with torch.random.fork_rng():
        torch.manual_seed(0)
        policy = ExplorationPolicy(_policy_config())
    policy_before = {name: value.detach().clone() for name, value in policy.state_dict().items()}
    episodes = (
        _episode(reward=0.25, terminated=True, truncated=False, bootstrap=0.0),
        _episode(reward=2.0, terminated=True, truncated=False, bootstrap=0.0),
    )
    report = PPOUpdater(policy, _ppo_config()).update(episodes)
    summary = build_update_summary(0, episodes, report)

    assert any(
        not torch.equal(value, policy_before[name]) for name, value in policy.state_dict().items()
    )
    assert report.sample_count == 2
    assert report.evaluated_minibatch_count == 1
    assert report.optimizer_step_count == 1
    assert not report.kl_early_stopped
    assert report.kl_early_stop_trigger is None
    assert report.cumulative_kl_early_stop_count == 0
    assert report.policy_ratio_mean > 0.0
    assert report.policy_ratio_std >= 0.0
    assert report.policy_ratio_p95 > 0.0
    assert report.policy_ratio_max > 0.0
    metrics = (value for value in report.__dict__.values() if isinstance(value, float))
    assert all(math.isfinite(value) for value in metrics)
    assert math.isfinite(report.mean_value_target)
    assert math.isfinite(report.std_value_target)
    assert report.std_value_target >= 0.0
    assert isinstance(summary, TrainingUpdateSummary)
    assert summary.update_index == 0
    assert summary.episode_count == 2
    assert summary.sample_count == 2
    assert math.isclose(summary.mean_episode_length, 1.0)
    assert math.isfinite(summary.mean_state_value)
    assert math.isfinite(summary.std_state_value)
    assert summary.std_state_value >= 0.0
    assert math.isfinite(summary.mean_value_target)
    assert math.isfinite(summary.std_value_target)
    assert summary.std_value_target >= 0.0
    for field in (
        summary.beta_alpha_mean,
        summary.beta_alpha_min,
        summary.beta_alpha_max,
        summary.beta_beta_mean,
        summary.beta_beta_min,
        summary.beta_beta_max,
        summary.action_mean,
        summary.action_std,
        summary.action_min,
        summary.action_max,
    ):
        assert len(field) == 2
        assert all(math.isfinite(value) for value in field)
    for dim in range(2):
        assert summary.beta_alpha_min[dim] <= summary.beta_alpha_mean[dim]
        assert summary.beta_alpha_mean[dim] <= summary.beta_alpha_max[dim]
        assert summary.beta_beta_min[dim] <= summary.beta_beta_mean[dim]
        assert summary.beta_beta_mean[dim] <= summary.beta_beta_max[dim]
        assert summary.action_min[dim] <= summary.action_mean[dim]
        assert summary.action_mean[dim] <= summary.action_max[dim]
        assert summary.action_std[dim] >= 0.0
    assert summary.reward_profile == "plannerrft_energy_v1"


def test_target_kl_stops_before_triggering_minibatch_optimizer_step_and_resumes_state() -> None:
    config = _ppo_config().model_copy(
        update={
            "target_kl": 1e-12,
            "epochs": 4,
            "batch_size": 4,
            "minibatch_size": 2,
            "scheduler_total_optimizer_steps": 8,
        }
    )
    episodes = tuple(
        _episode(
            reward=float(index + 1),
            terminated=True,
            truncated=False,
            bootstrap=0.0,
        )
        for index in range(4)
    )
    updater = PPOUpdater(ExplorationPolicy(_policy_config()), config)

    report = updater.update(episodes)

    assert report.kl_early_stopped
    assert report.kl_early_stop_trigger is not None
    assert report.kl_early_stop_trigger > 1.5 * config.target_kl
    assert report.evaluated_minibatch_count == report.optimizer_step_count + 1
    assert 0 <= report.optimizer_step_count < config.optimizer_steps_per_update
    assert updater.completed_optimizer_steps == report.optimizer_step_count
    restored = PPOUpdater(ExplorationPolicy(_policy_config()), config)
    restored.restore_checkpoint_state(updater.checkpoint_state())
    assert restored.checkpoint_state()["completed_optimizer_steps"] == report.optimizer_step_count
    assert restored.checkpoint_state()["kl_early_stop_count"] == 1


def test_optional_gradient_diagnostics_report_finite_parameter_group_norms() -> None:
    config = _ppo_config().model_copy(update={"gradient_diagnostics": True})
    episodes = (
        _episode(reward=0.25, terminated=True, truncated=False, bootstrap=0.0),
        _episode(reward=2.0, terminated=True, truncated=False, bootstrap=0.0),
    )

    report = PPOUpdater(ExplorationPolicy(_policy_config()), config).update(episodes)

    assert report.gradient_diagnostics is not None
    values = report.gradient_diagnostics.__dict__.values()
    assert all(math.isfinite(value) and value >= 0.0 for value in values)
    assert any(value > 0.0 for value in values)


def test_scheduler_horizon_covers_every_epoch_and_minibatch_across_updates() -> None:
    config = _ppo_config().model_copy(
        update={
            "epochs": 4,
            "batch_size": 4,
            "minibatch_size": 2,
            "scheduler_total_optimizer_steps": 32,
        }
    )
    episodes = tuple(
        _episode(
            reward=float(index + 1),
            terminated=True,
            truncated=False,
            bootstrap=0.0,
        )
        for index in range(4)
    )
    updater = PPOUpdater(ExplorationPolicy(_policy_config()), config)

    reports = [updater.update(episodes) for _ in range(4)]

    assert [report.optimizer_step_count for report in reports] == [8, 8, 8, 8]
    assert updater.completed_optimizer_steps == 32
