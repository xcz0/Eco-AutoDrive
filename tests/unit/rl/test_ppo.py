from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from eco_planner.rl.config import PPOConfig
from eco_planner.rl.distributions import AffineBeta
from eco_planner.rl.policy import ExplorationPolicy, ExplorationPolicyContext
from eco_planner.rl.ppo import (
    PPOUpdater,
    _batch_trajectories,
    _build_torchrl_policy_adapters,
)
from eco_planner.rl.rollout import (
    build_rollout_audit,
    build_training_decision,
    build_training_transition,
    finalize_rollout_episode,
    set_training_transition_next_state_value,
)


def _config() -> PPOConfig:
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
    training_transitions = []
    audit_transitions = []
    for index, reward in enumerate((1.0, 2.0)):
        decision = build_training_decision(
            context,
            action.guidance_action,
            action.joint_guidance_log_prob,
            output.value,
        )
        if training_transitions:
            set_training_transition_next_state_value(
                training_transitions[-1], decision["state_value"]
            )
        training_transitions.append(
            build_training_transition(
                decision,
                reward=reward,
                terminated=index == 1,
                truncated=False,
            )
        )
        audit_transitions.append(
            build_rollout_audit(
                policy_context=context,
                base_action=action.base_action,
                guidance_action=action.guidance_action,
                old_joint_guidance_log_prob=action.joint_guidance_log_prob,
                state_value=output.value,
                beta_alpha=output.distribution.parameters.alpha,
                beta_beta=output.distribution.parameters.beta,
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
    return finalize_rollout_episode(
        training_transitions, audit_transitions, "terminated", torch.zeros(1)
    )


def _sample_ids(updater: PPOUpdater, values: range) -> list[int]:
    sample_ids = torch.tensor(list(values), dtype=torch.int64)
    updater._minibatch_replay_buffer.extend(
        TensorDict({"sample_id": sample_ids}, batch_size=[sample_ids.numel()])
    )
    return torch.cat(
        [minibatch["sample_id"] for minibatch in updater._minibatch_replay_buffer]
    ).tolist()


def test_gae_consumes_the_rollout_tensor_dict_directly(exploration_policy_config) -> None:
    episode = _episode(ExplorationPolicy(exploration_policy_config))
    batch = _batch_trajectories((episode,), _config())
    assert batch["advantage"].shape == (2, 1)
    assert batch["value_target"].shape == (2, 1)


def test_ppo_batch_excludes_audit_only_rollout_fields(exploration_policy_config) -> None:
    episode = _episode(ExplorationPolicy(exploration_policy_config))
    batch = _batch_trajectories((episode,), _config())

    assert set(batch.keys()) == {
        "scene_tokens",
        "scene_padding_mask",
        "navigation_tokens",
        "navigation_padding_mask",
        "reference_trajectory",
        "guidance_action",
        "old_joint_guidance_log_prob",
        "advantage",
        "value_target",
    }
    assert batch["guidance_action"].shape == (2, 2)


def test_ppo_update_uses_canonical_affine_beta_distribution(exploration_policy_config) -> None:
    policy = ExplorationPolicy(exploration_policy_config)
    before = policy.actor_head.bias.detach().clone()
    report = PPOUpdater(policy, _config()).update((_episode(policy),))
    assert report.sample_count == 2
    assert not torch.equal(policy.actor_head.bias, before)
    distribution = AffineBeta(torch.full((1, 2), 2.0), torch.full((1, 2), 2.0))
    assert distribution.log_prob(torch.zeros((1, 2))).shape == (1,)


def test_torchrl_adapters_match_one_direct_policy_forward(exploration_policy_config) -> None:
    policy = ExplorationPolicy(exploration_policy_config)
    batch = _batch_trajectories((_episode(policy),), _config())
    context = ExplorationPolicyContext(
        **{
            key: batch[key]
            for key in (
                "scene_tokens",
                "scene_padding_mask",
                "navigation_tokens",
                "navigation_padding_mask",
                "reference_trajectory",
            )
        }
    )
    with torch.no_grad():
        expected = policy(context)
    actor, critic = _build_torchrl_policy_adapters(policy)
    adapted = batch.clone()
    actor.get_dist(adapted)
    adapted = critic(adapted)

    torch.testing.assert_close(adapted["alpha"], expected.distribution.parameters.alpha)
    torch.testing.assert_close(adapted["beta"], expected.distribution.parameters.beta)
    torch.testing.assert_close(adapted["state_value"], expected.value.unsqueeze(-1))


def test_ppo_loss_evaluates_the_shared_policy_once_per_minibatch(
    exploration_policy_config, monkeypatch
) -> None:
    policy = ExplorationPolicy(exploration_policy_config)
    updater = PPOUpdater(policy, _config())
    batch = _batch_trajectories((_episode(policy),), _config())
    original_forward_tensors = policy.forward_tensors
    calls = 0

    def counted_forward_tensors(*args: torch.Tensor):
        nonlocal calls
        calls += 1
        return original_forward_tensors(*args)

    monkeypatch.setattr(policy, "forward_tensors", counted_forward_tensors)
    losses = updater.loss_module(batch)
    total_loss = losses["loss_objective"] + losses["loss_critic"] + losses["loss_entropy"]
    total_loss.backward()

    assert calls == 1
    assert policy.fusion_trunk[0].weight.grad is not None
    assert policy.actor_head.weight.grad is not None
    assert policy.value_head.weight.grad is not None


def test_ppo_update_aggregates_multi_minibatch_diagnostics_once(
    exploration_policy_config, monkeypatch
) -> None:
    config = _config().model_copy(
        update={
            "epochs": 2,
            "batch_size": 4,
            "minibatch_size": 2,
            "scheduler_total_optimizer_steps": 4,
        }
    )
    policy = ExplorationPolicy(exploration_policy_config)

    class FixedLoss(torch.nn.Module):
        def __init__(self, parameter: torch.nn.Parameter) -> None:
            super().__init__()
            self.parameter = parameter
            self.calls = 0

        def forward(self, minibatch: object) -> dict[str, torch.Tensor]:
            assert not isinstance(minibatch, torch.Tensor)
            assert set(minibatch.keys()) == {
                "scene_tokens",
                "scene_padding_mask",
                "navigation_tokens",
                "navigation_padding_mask",
                "reference_trajectory",
                "guidance_action",
                "old_joint_guidance_log_prob",
                "advantage",
                "value_target",
            }
            self.calls += 1
            scale = self.parameter.new_tensor(float(self.calls))
            anchor = self.parameter.reshape(-1)[0] * 0.0
            return {
                "loss_objective": anchor + scale,
                "loss_critic": anchor + 10.0 * scale,
                "loss_entropy": anchor - 0.5 * scale,
                "kl_approx": anchor + 0.01 * scale,
                "clip_fraction": anchor + 0.1 * scale,
                "entropy": anchor + 1.5 * scale,
                "explained_variance": anchor + 3.0 * scale,
            }

    updater = PPOUpdater(policy, config)
    fixed_loss = FixedLoss(next(policy.parameters()))
    updater.loss_module = fixed_loss
    original_cpu = torch.Tensor.cpu
    cpu_call_count = 0

    def track_cpu(value: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        nonlocal cpu_call_count
        cpu_call_count += 1
        return original_cpu(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", track_cpu)
    report = updater.update((_episode(policy), _episode(policy)))

    assert fixed_loss.calls == 4
    assert cpu_call_count == 1
    assert report.sample_count == 4
    assert report.optimizer_step_count == 4
    assert report.mean_policy_loss == pytest.approx(2.5)
    assert report.mean_value_loss == pytest.approx(25.0)
    assert report.mean_entropy_loss == pytest.approx(-1.25)
    assert report.mean_total_loss == pytest.approx(26.25)
    assert report.mean_approximate_kl == pytest.approx(0.025)
    assert report.mean_clip_fraction == pytest.approx(0.25)
    assert report.mean_entropy == pytest.approx(3.75)
    assert report.mean_explained_variance == pytest.approx(7.5)
    assert report.maximum_pre_clip_gradient_norm == pytest.approx(0.0)
    assert report.final_learning_rate == pytest.approx(0.0)


def test_replay_buffer_sampler_has_a_fixed_seed_minibatch_sequence(
    exploration_policy_config,
) -> None:
    config = _config().model_copy(
        update={
            "batch_size": 4,
            "minibatch_size": 2,
            "scheduler_total_optimizer_steps": 2,
        }
    )
    updater = PPOUpdater(ExplorationPolicy(exploration_policy_config), config)

    assert _sample_ids(updater, range(4)) == [3, 2, 0, 1]


def test_replay_buffer_sampler_resume_preserves_future_minibatches(
    exploration_policy_config,
) -> None:
    config = _config().model_copy(
        update={
            "batch_size": 4,
            "minibatch_size": 2,
            "scheduler_total_optimizer_steps": 2,
        }
    )
    source = PPOUpdater(ExplorationPolicy(exploration_policy_config), config)
    _sample_ids(source, range(4))
    checkpoint_state = source.checkpoint_state()
    restored = PPOUpdater(ExplorationPolicy(exploration_policy_config), config)
    restored.restore_checkpoint_state(checkpoint_state)

    assert _sample_ids(source, range(10, 14)) == _sample_ids(restored, range(10, 14))
    assert torch.equal(
        source.checkpoint_state()["minibatch_generator_state"],
        restored.checkpoint_state()["minibatch_generator_state"],
    )
