from __future__ import annotations

import torch

from eco_planner.rl.config import PPOConfig
from eco_planner.rl.distributions import AffineBeta
from eco_planner.rl.policy import ExplorationPolicy, ExplorationPolicyContext
from eco_planner.rl.ppo import PPOUpdater, estimate_episode_gae
from eco_planner.rl.rollout import build_rollout_transition, finalize_rollout_episode


def _config() -> PPOConfig:
    return PPOConfig(
        name="test",
        gamma=0.99,
        gae_lambda=0.95,
        normalize_advantage=True,
        clip_epsilon=0.2,
        value_loss="l2",
        clip_value=False,
        value_coefficient=0.5,
        entropy_coefficient=0.01,
        optimizer="adam",
        learning_rate=0.00025,
        adam_epsilon=1e-5,
        weight_decay=0.0,
        max_gradient_norm=0.5,
        epochs=1,
        batch_size=2,
        minibatch_size=2,
        minibatch_seed=7,
        scheduler="cosine",
        scheduler_total_optimizer_steps=1,
        scheduler_minimum_learning_rate=0.0,
    )


def _context(hidden_dim: int) -> ExplorationPolicyContext:
    return ExplorationPolicyContext(
        scene_tokens=torch.zeros((1, 2, hidden_dim)),
        scene_padding_mask=torch.zeros((1, 2), dtype=torch.bool),
        navigation_tokens=torch.zeros((1, 1, hidden_dim)),
        navigation_padding_mask=torch.zeros((1, 1), dtype=torch.bool),
        reference_trajectory=torch.zeros((1, 80, 4)),
    )


def _episode(policy: ExplorationPolicy):
    context = _context(policy.config.hidden_dim)
    with torch.no_grad():
        output, action = policy.act(context, "sample", torch.Generator().manual_seed(4))
    transitions = []
    for index, reward in enumerate((1.0, 2.0)):
        transitions.append(
            build_rollout_transition(
                policy_context=context,
                base_action=action.base_action,
                guidance_action=action.guidance_action,
                old_joint_guidance_log_prob=action.joint_guidance_log_prob,
                state_value=output.value,
                beta_alpha=output.parameters.alpha,
                beta_beta=output.parameters.beta,
                initial_noise=torch.zeros((1, 11, 80, 4)),
                diffusion_rng_state=torch.ones(5, dtype=torch.uint8),
                policy_rng_state=torch.ones(5, dtype=torch.uint8),
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
                terminated=index == 1,
                truncated=False,
                map_seed=0,
                noise_seed=1,
                policy_action_seed=2,
                planning_cycle_index=index,
            )
        )
    return finalize_rollout_episode(transitions, "terminated", torch.zeros(1))


def test_gae_consumes_the_rollout_tensor_dict_directly(exploration_policy_config) -> None:
    episode = _episode(ExplorationPolicy(exploration_policy_config))
    estimate = estimate_episode_gae(episode, _config())
    assert estimate.advantage.shape == (2, 1)
    assert estimate.value_target.shape == (2, 1)


def test_ppo_update_uses_canonical_affine_beta_distribution(exploration_policy_config) -> None:
    policy = ExplorationPolicy(exploration_policy_config)
    before = policy.actor_head.bias.detach().clone()
    report = PPOUpdater(policy, _config()).update((_episode(policy),))
    assert report.sample_count == 2
    assert not torch.equal(policy.actor_head.bias, before)
    distribution = AffineBeta(torch.full((1, 2), 2.0), torch.full((1, 2), 2.0))
    assert distribution.log_prob(torch.zeros((1, 2))).shape == (1,)
