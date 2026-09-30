"""Synthetic training inputs with explicit configuration and isolated summary RNG."""

from __future__ import annotations

import torch
from tensordict import TensorDictBase

from eco_planner.jobs import compose_job_config
from eco_planner.planning.policy import (
    ExplorationPolicy,
    ExplorationPolicyConfig,
    ExplorationPolicyContext,
    policy_context_tensordict,
)
from eco_planner.reward import PlannerRFTRewardResult, RewardComponents, RewardDiagnostics
from eco_planner.reward.result import RewardProfileName
from eco_planner.rl.artifacts import build_update_summary
from eco_planner.rl.config import parse_training_config
from eco_planner.rl.optimization import PPOConfig, PPOUpdater
from eco_planner.rl.rollout import (
    ExecutionTransitionAudit,
    RolloutEpisodeBuilder,
    RolloutProvenance,
    build_training_decision,
    build_training_transition,
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


def _decision_audit(context: ExplorationPolicyContext | None = None) -> TensorDictBase:
    if context is None:
        context = _context()
    return policy_context_tensordict(context).update(
        dict(
            prediction=torch.zeros((1, 11, 80, 4)),
            initial_noise=torch.zeros((1, 11, 80, 4)),
            base_action=torch.tensor([[0.25, 0.75]]),
            guidance_action=torch.tensor([[-0.5, 0.5]]),
            old_joint_guidance_log_prob=torch.tensor([[0.5]]),
            state_value=torch.tensor([[1.0]]),
            beta_alpha=torch.full((1, 2), 2.0),
            beta_beta=torch.full((1, 2), 2.0),
            diffusion_rng_state=torch.ones((1, 5), dtype=torch.uint8),
            policy_rng_state=torch.ones((1, 5), dtype=torch.uint8),
        )
    )


def _execution_audit(
    reward: float,
    *,
    terminated: bool,
    truncated: bool,
    profile_name: RewardProfileName = "plannerrft_energy_v1",
) -> ExecutionTransitionAudit:
    result = PlannerRFTRewardResult(
        profile_name=profile_name,
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
    )
    return ExecutionTransitionAudit(
        reward_result=result,
        substep_results=(result,),
        route_completion_delta=0.1,
        distance_m=1.0,
        speed_mps=2.0,
        stopped=False,
        collision=False,
        wrong_direction=False,
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


def _episode(
    *,
    reward: float,
    terminated: bool,
    truncated: bool,
    bootstrap: float,
    profile_name: RewardProfileName = "plannerrft_energy_v1",
):
    context = _context()
    decision = build_training_decision(
        context,
        torch.tensor([[-0.5, 0.5]]),
        torch.tensor([0.5]),
        torch.tensor([1.0]),
    )
    builder = RolloutEpisodeBuilder()
    execution = _execution_audit(
        reward, terminated=terminated, truncated=truncated, profile_name=profile_name
    )
    builder.append(
        build_training_transition(
            decision, execution.reward_result, terminated=terminated, truncated=truncated
        ),
        _decision_audit(),
        execution,
        RolloutProvenance(0, 1, 2, 0),
    )
    tail_kind = "terminated" if terminated else "truncated" if truncated else "rollout_limit"
    return builder.finish(tail_kind, torch.tensor([bootstrap]))


def _behavior_policy_episode(
    policy: ExplorationPolicy, guidance_action: torch.Tensor, reward: float
):
    context = _context()
    with torch.no_grad():
        outputs = policy.forward_tensordict(
            policy_context_tensordict(context).to(next(policy.parameters()).device)
        ).cpu()
        distribution = policy.output_from_tensordict(outputs).distribution
        old_log_prob = distribution.log_prob(guidance_action)
    decision = build_training_decision(
        context,
        guidance_action,
        old_log_prob,
        outputs["state_value"],
    )
    decision_audit = _decision_audit().update(
        dict(
            base_action=(guidance_action + 1.0) / 2.0,
            guidance_action=guidance_action,
            old_joint_guidance_log_prob=old_log_prob.reshape(-1, 1),
            state_value=outputs["state_value"],
            beta_alpha=outputs["alpha"],
            beta_beta=outputs["beta"],
        )
    )
    builder = RolloutEpisodeBuilder()
    execution = _execution_audit(reward, terminated=True, truncated=False)
    builder.append(
        build_training_transition(
            decision, execution.reward_result, terminated=True, truncated=False
        ),
        decision_audit,
        execution,
        RolloutProvenance(0, 1, 2, 0),
    )
    return builder.finish("terminated", torch.zeros(1))


def _update_report(episodes):
    with torch.random.fork_rng():
        torch.manual_seed(0)
        policy = ExplorationPolicy(_policy_config())
    return PPOUpdater(policy, _ppo_config()).update(episodes)


def _config(tmp_path, *, seed=0, enabled=True, resume=None):
    raw = compose_job_config(
        "jobs/training/ppo",
        [
            "components/resources=rtx3050_laptop",
            f"runtime.seed={seed}",
            "training.replay_id=0",
        ],
    )
    raw.tracking.tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    raw.tracking.artifact_location = (tmp_path / "artifacts").as_posix()
    raw.tracking.enabled = enabled
    raw.training.resume_checkpoint_path = resume
    return parse_training_config(raw)


def build_training_update_summary():
    with torch.random.fork_rng():
        torch.manual_seed(0)
        policy = ExplorationPolicy(_policy_config())
        episodes = tuple(
            _episode(reward=r, terminated=True, truncated=False, bootstrap=0.0) for r in (0.25, 2.0)
        )
        return build_update_summary(0, episodes, PPOUpdater(policy, _ppo_config()).update(episodes))
