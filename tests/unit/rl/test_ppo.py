from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch
from tensordict import TensorDict
from tensordict.nn import ProbabilisticTensorDictModule, TensorDictModule
from torch import nn
from torchrl.objectives import ClipPPOLoss

from eco_planner.rl import (
    BetaGuidanceDistribution,
    BetaGuidanceParameters,
    ExplorationPolicy,
    ExplorationPolicyConfig,
    ExplorationPolicyContext,
    MetaDriveRolloutReward,
    PPOOptimizationConfig,
    PPOUpdater,
    RolloutBuffer,
    RolloutEpisode,
    RolloutTransition,
    estimate_episode_gae,
)
from eco_planner.rl.ppo import _AffineBetaDistribution, _normalize_full_batch_advantage


def _ppo_config(**updates: object) -> PPOOptimizationConfig:
    config = PPOOptimizationConfig(
        name="ppo_stage5_smoke",
        gamma=0.99,
        gae_lambda=0.95,
        normalize_advantage=True,
        clip_epsilon=0.2,
        value_loss="l2",
        clip_value=False,
        value_coefficient=0.5,
        entropy_coefficient=0.01,
        optimizer="adam",
        learning_rate=2.5e-4,
        adam_epsilon=1e-5,
        weight_decay=0.0,
        max_gradient_norm=0.5,
        epochs=4,
        batch_size=4,
        minibatch_size=4,
        minibatch_seed=7,
        scheduler="cosine",
        scheduler_total_optimizer_steps=4,
        scheduler_minimum_learning_rate=0.0,
    )
    return replace(config, **updates)


def _policy_config() -> ExplorationPolicyConfig:
    return ExplorationPolicyConfig(
        name="exploration_beta",
        hidden_dim=12,
        reference_horizon=80,
        reference_state_dim=4,
        reference_mixer_depth=1,
        reference_token_mlp_hidden_dim=16,
        reference_channel_mlp_hidden_dim=24,
        cross_attention_heads=3,
        cross_attention_dropout=0.2,
        fusion_mlp_depth=1,
        fusion_hidden_dim=16,
        initial_concentration=2.0,
        minimum_concentration=1e-4,
    )


def _context(scale: float = 1.0) -> ExplorationPolicyContext:
    reference = torch.zeros((1, 80, 4), dtype=torch.float32)
    reference[..., 0] = torch.arange(1, 81, dtype=torch.float32) * scale
    reference[..., 2] = 1.0
    return ExplorationPolicyContext(
        scene_tokens=torch.linspace(-scale, scale, 60).reshape(1, 5, 12),
        scene_padding_mask=torch.tensor([[False, False, False, True, True]]),
        navigation_tokens=torch.full((1, 2, 12), scale, dtype=torch.float32),
        navigation_padding_mask=torch.tensor([[False, True]]),
        reference_trajectory=reference,
    )


def _transition(
    index: int,
    *,
    reward: float,
    value: float,
    terminated: bool = False,
    truncated: bool = False,
    context: ExplorationPolicyContext | None = None,
    base_action: torch.Tensor | None = None,
    old_log_prob: torch.Tensor | None = None,
) -> RolloutTransition:
    base = torch.tensor([[0.25, 0.75]], dtype=torch.float32) if base_action is None else base_action
    log_prob = torch.tensor([0.0], dtype=torch.float32) if old_log_prob is None else old_log_prob
    return RolloutTransition(
        policy_context=_context() if context is None else context,
        base_action=base,
        guidance_action=2.0 * base - 1.0,
        old_joint_guidance_log_prob=log_prob,
        old_value=torch.tensor([value], dtype=torch.float32),
        initial_noise=torch.zeros((1, 11, 80, 4), dtype=torch.float32),
        diffusion_rng_state=torch.Generator().manual_seed(3).get_state(),
        policy_rng_state=torch.Generator().manual_seed(4).get_state(),
        reward=MetaDriveRolloutReward(
            substep_scores=torch.tensor([reward], dtype=torch.float32),
            total_score=torch.tensor([reward], dtype=torch.float32),
        ),
        terminated=terminated,
        truncated=truncated,
        bootstrap_mask=not terminated,
        scenario_name="straight",
        map_sequence="S",
        map_seed=1,
        noise_seed=2,
        policy_action_seed=3,
        planning_cycle_index=index,
        executed_substep_count=1,
    )


def _episode(
    *,
    tail_kind: str = "rollout_limit",
    tail_value: float = 3.0,
) -> RolloutEpisode:
    buffer = RolloutBuffer()
    buffer.append(_transition(0, reward=0.5, value=1.0))
    buffer.append(
        _transition(
            1,
            reward=1.0,
            value=2.0,
            terminated=tail_kind == "terminated",
            truncated=tail_kind == "truncated",
        )
    )
    return buffer.finalize(  # type: ignore[arg-type]
        tail_kind, torch.tensor([tail_value], dtype=torch.float32)
    )


@pytest.mark.parametrize(
    ("tail_kind", "tail_value", "expected"),
    [
        ("rollout_limit", 3.0, (2.524, 1.7)),
        ("truncated", 3.0, (2.524, 1.7)),
        ("terminated", 0.0, (0.58, -1.0)),
    ],
)
def test_torchrl_gae_matches_terminal_truncation_and_rollout_tail_hand_math(
    tail_kind: str,
    tail_value: float,
    expected: tuple[float, float],
) -> None:
    config = _ppo_config(gamma=0.9, gae_lambda=0.8)
    estimate = estimate_episode_gae(_episode(tail_kind=tail_kind, tail_value=tail_value), config)
    torch.testing.assert_close(
        estimate.advantage[:, 0], torch.tensor(expected), atol=1e-6, rtol=1e-6
    )
    torch.testing.assert_close(
        estimate.value_target[:, 0], torch.tensor(expected) + torch.tensor([1.0, 2.0])
    )


def test_episode_gae_is_computed_without_cross_episode_leakage() -> None:
    config = _ppo_config(gamma=0.9, gae_lambda=0.8)
    first = estimate_episode_gae(_episode(tail_value=3.0), config)
    second = estimate_episode_gae(_episode(tail_value=30.0), config)
    assert first.advantage[-1].item() == pytest.approx(1.7)
    assert second.advantage[-1].item() == pytest.approx(26.0)


def test_full_batch_advantage_normalization_uses_sample_standard_deviation() -> None:
    batch = TensorDict({"advantage": torch.tensor([[1.0], [2.0], [4.0]])}, [3])
    _normalize_full_batch_advantage(batch)
    assert batch["advantage"].mean().item() == pytest.approx(0.0, abs=1e-7)
    assert batch["advantage"].std(correction=1).item() == pytest.approx(1.0)


@pytest.mark.parametrize("values", [[[1.0]], [[2.0], [2.0]], [[1.0], [float("nan")]]])
def test_advantage_normalization_rejects_small_zero_variance_or_nonfinite_batches(
    values: list[list[float]],
) -> None:
    batch = TensorDict({"advantage": torch.tensor(values)}, [len(values)])
    with pytest.raises(ValueError, match="two samples|zero variance|finite"):
        _normalize_full_batch_advantage(batch)


def test_affine_beta_distribution_matches_existing_policy_probability_contract() -> None:
    parameters = BetaGuidanceParameters(
        alpha=torch.tensor([[2.0, 3.0]]), beta=torch.tensor([[4.0, 5.0]])
    )
    base_action = torch.tensor([[0.25, 0.75]])
    expected = BetaGuidanceDistribution(parameters).evaluate_base_action(base_action)
    actual = _AffineBetaDistribution(parameters.alpha, parameters.beta)
    torch.testing.assert_close(
        actual.log_prob(expected.guidance_action), expected.joint_guidance_log_prob
    )
    torch.testing.assert_close(actual.entropy(), expected.joint_guidance_entropy)
    with pytest.raises(ValueError, match="strictly inside"):
        actual.log_prob(torch.tensor([[-1.0, 0.0]]))


class _IdentityValue(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value


def test_torchrl_clip_ppo_loss_matches_hand_calculated_objective_value_and_entropy() -> None:
    actor = ProbabilisticTensorDictModule(
        in_keys={"alpha": "alpha", "beta": "beta"},
        out_keys=["guidance_action"],
        distribution_class=_AffineBetaDistribution,
        return_log_prob=True,
        log_prob_key="joint_guidance_log_prob",
    )
    critic = TensorDictModule(_IdentityValue(), in_keys=["new_value"], out_keys=["state_value"])
    loss = ClipPPOLoss(
        actor,
        critic,
        clip_epsilon=0.2,
        entropy_bonus=True,
        entropy_coeff=0.01,
        critic_coeff=0.5,
        loss_critic_type="l2",
        normalize_advantage=False,
        functional=False,
        reduction="mean",
        clip_value=None,
    )
    loss.set_keys(
        action="guidance_action",
        sample_log_prob="old_log_prob",
        value="state_value",
        advantage="advantage",
        value_target="value_target",
    )
    alpha = torch.tensor([[2.0, 2.0], [2.0, 2.0]])
    beta = torch.tensor([[3.0, 3.0], [3.0, 3.0]])
    action = torch.tensor([[-0.5, 0.5], [0.0, 0.25]])
    distribution = _AffineBetaDistribution(alpha, beta)
    new_log_prob = distribution.log_prob(action).detach()
    ratio = torch.tensor([1.5, 0.5])
    batch = TensorDict(
        {
            "alpha": alpha,
            "beta": beta,
            "guidance_action": action,
            "old_log_prob": new_log_prob - ratio.log(),
            "advantage": torch.tensor([[1.0], [-1.0]]),
            "new_value": torch.tensor([[2.0], [0.0]]),
            "state_value": torch.zeros((2, 1)),
            "value_target": torch.ones((2, 1)),
        },
        [2],
    )
    result = loss(batch)
    expected_entropy = distribution.entropy().mean()
    assert result["loss_objective"].item() == pytest.approx(-0.2)
    assert result["loss_critic"].item() == pytest.approx(0.5)
    assert result["loss_entropy"].item() == pytest.approx(-0.01 * expected_entropy.item())
    expected_total = -0.2 + 0.5 - 0.01 * expected_entropy.item()
    actual_total = result["loss_objective"] + result["loss_critic"] + result["loss_entropy"]
    assert actual_total.item() == pytest.approx(expected_total)
    assert result["clip_fraction"].item() == pytest.approx(1.0)


def _policy_episode(policy: ExplorationPolicy) -> RolloutEpisode:
    buffer = RolloutBuffer()
    for index, reward in enumerate((0.1, 0.4, -0.2, 1.0)):
        context = _context(scale=1.0 + index * 0.1)
        base_action = torch.tensor([[0.2 + index * 0.05, 0.75 - index * 0.05]])
        with torch.no_grad():
            output = policy(context)
            action = output.distribution.evaluate_base_action(base_action)
        buffer.append(
            _transition(
                index,
                reward=reward,
                value=float(output.value.item()),
                context=context,
                base_action=base_action,
                old_log_prob=action.joint_guidance_log_prob.detach(),
            )
        )
    with torch.no_grad():
        tail_value = policy(_context(scale=1.5)).value.detach().cpu()
    return buffer.finalize("rollout_limit", tail_value)


def test_ppo_updater_changes_only_policy_and_advances_cosine_per_optimizer_step() -> None:
    torch.manual_seed(11)
    policy = ExplorationPolicy(_policy_config())
    episode = _policy_episode(policy)
    sentinel = nn.Linear(2, 2)
    sentinel_before = {
        name: value.detach().clone() for name, value in sentinel.state_dict().items()
    }
    policy_before = {name: value.detach().clone() for name, value in policy.state_dict().items()}
    old_log_probs = [step.old_joint_guidance_log_prob.clone() for step in episode.transitions]
    updater = PPOUpdater(policy, _ppo_config())
    report = updater.update([episode])
    assert not policy.training
    assert report.sample_count == 4
    assert report.optimizer_step_count == 4
    assert report.final_learning_rate == pytest.approx(0.0, abs=1e-12)
    assert math.isfinite(report.mean_total_loss)
    assert not torch.equal(policy.actor_head.weight, policy_before["actor_head.weight"])
    assert not torch.equal(policy.value_head.weight, policy_before["value_head.weight"])
    assert any(
        not torch.equal(value, policy_before[name])
        for name, value in policy.state_dict().items()
        if name.startswith("fusion_trunk")
    )
    for name, value in sentinel.state_dict().items():
        assert torch.equal(value, sentinel_before[name])
    for step, expected in zip(episode.transitions, old_log_probs, strict=True):
        assert torch.equal(step.old_joint_guidance_log_prob, expected)
    with pytest.raises(RuntimeError, match="scheduler horizon"):
        updater.update([episode])


def test_minibatch_indices_cover_every_sample_once_per_epoch() -> None:
    policy = ExplorationPolicy(_policy_config())
    updater = PPOUpdater(
        policy,
        _ppo_config(epochs=2, minibatch_size=2, scheduler_total_optimizer_steps=4),
    )
    first = updater._epoch_minibatch_indices(4)
    second = updater._epoch_minibatch_indices(4)
    for epoch in (first, second):
        flattened = torch.cat(epoch)
        assert sorted(flattened.tolist()) == [0, 1, 2, 3]
        assert all(index.numel() == 2 for index in epoch)


def test_ppo_updater_rejects_nonfinite_loss_before_parameter_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = ExplorationPolicy(_policy_config())
    episode = _policy_episode(policy)
    updater = PPOUpdater(policy, _ppo_config())
    before = {name: value.detach().clone() for name, value in policy.state_dict().items()}

    def nonfinite_loss(_batch: TensorDict) -> TensorDict:
        return TensorDict(
            {
                "loss_objective": torch.tensor(float("nan")),
                "loss_critic": torch.tensor(0.0),
                "loss_entropy": torch.tensor(0.0),
            },
            [],
        )

    monkeypatch.setattr(updater.loss_module, "forward", nonfinite_loss)
    with pytest.raises(FloatingPointError, match="total PPO loss"):
        updater.update([episode])
    for name, value in policy.state_dict().items():
        assert torch.equal(value, before[name])


def test_ppo_updater_rejects_nonfinite_gradient_before_parameter_step() -> None:
    policy = ExplorationPolicy(_policy_config())
    episode = _policy_episode(policy)
    updater = PPOUpdater(policy, _ppo_config())
    before = {name: value.detach().clone() for name, value in policy.state_dict().items()}
    hook = policy.value_head.weight.register_hook(
        lambda gradient: torch.full_like(gradient, float("inf"))
    )
    try:
        with pytest.raises(RuntimeError, match="non-finite"):
            updater.update([episode])
    finally:
        hook.remove()
    for name, value in policy.state_dict().items():
        assert torch.equal(value, before[name])
