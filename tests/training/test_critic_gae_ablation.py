from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from eco_planner.experiments.credit.decisions import (
    CREDIT_FORMS,
    AttributionThresholds,
    evaluate_attribution,
)
from eco_planner.rl.optimization import compute_episode_gae
from eco_planner.rl.optimization.credit import (
    credit_batch,
    discounted_return_batch,
    zero_critic_values,
)
from eco_planner.rl.rollout import (
    RolloutEpisodeBuilder,
    RolloutProvenance,
    build_training_decision,
)
from tests.training.test_ppo import (
    _context,
    _decision_audit,
    _execution_audit,
    _ppo_config,
)


def _multi_step_episode(rewards: list[float], next_values: list[float], bootstrap: float):
    context = _context()
    builder = RolloutEpisodeBuilder()
    for reward in rewards:
        decision = build_training_decision(
            context,
            torch.tensor([[-0.5, 0.5]]),
            torch.tensor([0.5]),
            torch.tensor([1.0]),
        )
        builder.append(
            decision,
            _decision_audit(),
            _execution_audit(reward, terminated=False, truncated=False),
            RolloutProvenance(0, 1, 2, 0),
        )
    episode = builder.finish("rollout_limit", torch.tensor([bootstrap]))
    # These synthetic critic diagnostics intentionally set arbitrary V(s') values.
    training = episode.training.clone()
    training["next", "state_value"] = torch.tensor([*next_values[:-1], bootstrap]).unsqueeze(-1)
    return replace(episode, training=training)


def test_zero_critic_values_removes_values_and_bootstrap_only():
    episode = _multi_step_episode([0.25, -0.5], [0.7, 0.3], 0.4)
    zeroed = zero_critic_values(episode)
    assert zeroed.tail_kind == episode.tail_kind
    torch.testing.assert_close(zeroed.tail_bootstrap_value, episode.tail_bootstrap_value)
    assert (zeroed.training["state_value"] == 0).all()
    assert (zeroed.training["next", "state_value"] == 0).all()
    for key in episode.training.keys(include_nested=True, leaves_only=True):
        if key not in {"state_value", ("next", "state_value")}:
            torch.testing.assert_close(zeroed.training[key], episode.training[key], rtol=0, atol=0)
    assert (episode.training["state_value"] == 1.0).all()
    assert (episode.training["next", "state_value"] != 0).all()


def test_reward_only_gae_matches_gamma_lambda_return_without_bootstrap():
    config = _ppo_config()
    rewards = [0.25, -0.5, 1.0]
    episode = _multi_step_episode(rewards, [0.7, -0.2, 0.3], 0.4)
    reward_only = compute_episode_gae(zero_critic_values(episode), config)
    gamma_lambda = config.gamma * config.gae_lambda
    expected = [0.0] * len(rewards)
    running = 0.0
    for step in reversed(range(len(rewards))):
        running = rewards[step] + gamma_lambda * running
        expected[step] = running
    np.testing.assert_allclose(reward_only["advantage"].reshape(-1), expected, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(
        reward_only["value_target"].reshape(-1), expected, rtol=1e-5, atol=1e-7
    )
    standard = compute_episode_gae(episode, config)
    values = [1.0] * len(rewards)
    next_values = [0.7, -0.2, 0.4]
    delta = [rewards[t] + config.gamma * next_values[t] - values[t] for t in range(len(rewards))]
    standard_expected = [0.0] * len(rewards)
    running = 0.0
    for step in reversed(range(len(rewards))):
        running = delta[step] + gamma_lambda * running
        standard_expected[step] = running
    np.testing.assert_allclose(
        standard["advantage"].reshape(-1), standard_expected, rtol=1e-5, atol=1e-7
    )


def test_discounted_return_batch_matches_recursive_returns():
    config = _ppo_config()
    episodes = [
        _multi_step_episode([0.25, -0.5, 1.0], [0.7, -0.2, 0.3], 0.4),
        _multi_step_episode([0.1, 0.2], [-0.3, 0.5], -0.1),
    ]
    snapshots = [(episode.training.clone(), episode.audit.clone()) for episode in episodes]
    batch = discounted_return_batch(episodes, config.gamma)
    assert batch.batch_size[0] == 5
    np.testing.assert_allclose(batch["value_target"], batch["advantage"], rtol=0, atol=0)
    expected = []
    for episode in episodes:
        rewards = episode.training["next", "reward"].reshape(-1).tolist()
        returns = [0.0] * len(rewards)
        running = 0.0
        for step in reversed(range(len(rewards))):
            running = rewards[step] + config.gamma * running
            returns[step] = running
        expected.extend(returns)
    np.testing.assert_allclose(batch["advantage"].reshape(-1), expected, rtol=1e-5, atol=1e-7)
    # The critic-free return discounts by gamma, not gamma * gae_lambda.
    reward_only = credit_batch(episodes, config, "reward_only_gae")["advantage"].reshape(-1)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(batch["advantage"].reshape(-1), reward_only, rtol=1e-3)
    for episode, (training, audit) in zip(episodes, snapshots, strict=True):
        assert (episode.training == training).all()
        assert (episode.audit == audit).all()


def test_discounted_return_respects_every_tail_and_ignores_critic_bootstrap():
    from tests.training.test_ppo import _episode

    episodes = [
        _episode(
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            bootstrap=0.0 if terminated else 900.0,
        )
        for reward, (terminated, truncated) in enumerate(
            [(True, False), (False, True), (False, False), (True, True)], start=1
        )
    ]
    batch = discounted_return_batch(episodes, 0.5)
    torch.testing.assert_close(batch["value_target"].flatten(), torch.tensor([1.0, 2.0, 3.0, 4.0]))
    batch["advantage"].zero_()
    torch.testing.assert_close(batch["value_target"].flatten(), torch.tensor([1.0, 2.0, 3.0, 4.0]))


def _pair(cosine: float, *, rmse: float = 0.2, sign_flip: float = 0.0, form: str) -> dict:
    return {
        "credit_form": form,
        "sign_flip_fraction": sign_flip,
        "normalized_advantage_rmse": rmse,
        "gradients": {"actor_head": {"cosine": cosine, "norm_ratio_j_over_i": 1.0}},
    }


def test_attribution_branches():
    thresholds = _study().gate
    fails = {"cosine": 0.9999, "rmse": 0.01, "sign_flip": 0.0}
    passes = {"cosine": 0.9, "rmse": 0.3, "sign_flip": 0.1}
    dominated = evaluate_attribution(
        [
            _pair(**fails, form="standard_gae"),
            _pair(**passes, form="reward_only_gae"),
            _pair(**passes, form="discounted_return"),
        ],
        thresholds,
    )
    assert dominated["attribution"] == "critic_gae_common_term_dominated"
    assert not dominated["endpoint_identifiable_under_standard_gae"]

    collinear = evaluate_attribution(
        [_pair(**fails, form=form) for form in CREDIT_FORMS], thresholds
    )
    assert collinear["attribution"] == "reward_batch_collinearity"
    assert not collinear["endpoint_identifiable_under_standard_gae"]

    temporal = evaluate_attribution(
        [
            _pair(**fails, form="standard_gae"),
            _pair(**fails, form="reward_only_gae"),
            _pair(**passes, form="discounted_return"),
        ],
        thresholds,
    )
    assert temporal["attribution"] == "temporal_credit_structure_sensitivity"

    identifiable = evaluate_attribution(
        [_pair(**passes, form=form) for form in CREDIT_FORMS], thresholds
    )
    assert identifiable["attribution"] is None
    assert identifiable["endpoint_identifiable_under_standard_gae"]

    with pytest.raises(ValueError, match="one endpoint pair per credit form"):
        evaluate_attribution([_pair(**fails, form="standard_gae")], thresholds)


def _study():
    return SimpleNamespace(
        gate=AttributionThresholds(
            endpoint_max_actor_head_cosine=0.95,
            min_normalized_advantage_rmse=0.1,
            min_sign_flip_fraction=0.1,
        )
    )
