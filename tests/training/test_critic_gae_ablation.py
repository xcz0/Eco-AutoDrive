from __future__ import annotations

import numpy as np
import pytest
import torch

from eco_planner.analysis.reporting.fixed import render_ablation_report
from eco_planner.experiments.critic_gae_ablation.config import AblationConfig
from eco_planner.experiments.critic_gae_ablation.diagnostics import (
    CREDIT_FORMS,
    analyze_critic_gae_ablation,
    credit_batch,
    discounted_return_batch,
    evaluate_attribution,
    zero_critic_values,
)
from eco_planner.rl.optimization import PPOUpdater, compute_episode_gae
from eco_planner.rl.policy import ExplorationPolicy
from eco_planner.rl.rollout import (
    RolloutEpisodeBuilder,
    RolloutProvenance,
    build_training_decision,
)
from tests.training.test_ppo import (
    _behavior_policy_episode,
    _context,
    _decision_audit,
    _execution_audit,
    _policy_config,
    _ppo_config,
)
from tests.training.test_reward import _no_energy_config


def _study(**overrides: object) -> AblationConfig:
    values: dict[str, object] = {
        "quantiles": [0.0, 0.5, 1.0],
        "progress_target_score": 0.6,
        "comfort_target_score": 0.6,
        "calibration_match_tolerance": {"rtol": 1e-7, "atol": 1e-9},
        "expected_calibration": {
            "full_score_delta_m": 1.0,
            "longitudinal_acceleration_limit_mps2": 3.0,
            "lateral_acceleration_limit_mps2": 3.0,
            "jerk_limit_mps3": 5.0,
            "yaw_rate_limit_radps": 0.5,
        },
        "reference_match_tolerance": {"rtol": 1e-5, "atol": 1e-6},
        "gate": {
            "endpoint_max_actor_head_cosine": 0.99,
            "min_normalized_advantage_rmse": 0.10,
            "min_sign_flip_fraction": 0.05,
        },
    }
    values.update(overrides)
    return AblationConfig.model_validate(values)


def _multi_step_episode(rewards: list[float], next_values: list[float], bootstrap: float):
    context = _context()
    builder = RolloutEpisodeBuilder()
    for reward, next_value in zip(rewards, next_values, strict=True):
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
        builder.link_next_state_value(torch.tensor([[next_value]]))
    return builder.finish("rollout_limit", torch.tensor([bootstrap]))


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


def test_ablation_structures_and_identities(monkeypatch):
    torch.manual_seed(0)
    policy = ExplorationPolicy(_policy_config())
    updater = PPOUpdater(policy, _ppo_config().model_copy(update={"batch_size": 4}))
    episodes = [
        _behavior_policy_episode(policy, torch.tensor([action]), reward=0.5)
        for action in [(-0.5, 0.2), (0.3, -0.7), (-0.1, -0.4), (0.6, 0.8)]
    ]
    progress = [0.2, 0.4, 0.6, 0.8]
    energy = [0.9, 0.5, 0.7, 0.3]
    gates = [1.0, 0.5, 1.0, 0.25]
    for i, episode in enumerate(episodes):
        episode.audit["reward_component_progress"].fill_(progress[i])
        episode.audit["reward_component_energy"].fill_(energy[i])
        episode.audit["reward_safety_gate"].fill_(gates[i])
    snapshots = [(e.training.clone(), e.audit.clone()) for e in episodes]
    parameters = {k: v.clone() for k, v in policy.state_dict().items()}
    state_values = np.concatenate([e.training["state_value"].numpy().reshape(-1) for e in episodes])

    def forbidden(*args, **kwargs):
        pytest.fail("optimizer/scheduler step is forbidden")

    monkeypatch.setattr(updater.optimizer, "step", forbidden)
    monkeypatch.setattr(updater.scheduler, "step", forbidden)
    study = _study()
    summary, arrays = analyze_critic_gae_ablation(
        updater,
        episodes,
        _no_energy_config(),
        study.quantiles,
        np.array([0, 0, 1, 1]),
        study.gate,
    )
    assert [arm["label"] for arm in summary["arms"]] == ["r0", "energy_only"]
    assert [pair["credit_form"] for pair in summary["pairs"]] == list(CREDIT_FORMS)
    assert summary["optimizer_steps"] == 0

    for arm_label in ("r0", "energy_only"):
        for form in CREDIT_FORMS:
            raw = arrays[f"arm_{arm_label}__{form}__raw_advantage"]
            centered = arrays[f"arm_{arm_label}__{form}__center_advantage"]
            normalized = arrays[f"arm_{arm_label}__{form}__normalized_advantage"]
            np.testing.assert_allclose(centered, raw - raw.mean(), rtol=1e-6, atol=1e-9)
            sigma = raw.std(ddof=1)
            np.testing.assert_allclose(normalized, centered / sigma, rtol=1e-5, atol=1e-8)
            np.testing.assert_allclose(
                arrays[f"arm_{arm_label}__{form}__gradient_z_actor_head"],
                arrays[f"arm_{arm_label}__{form}__gradient_center_actor_head"] / sigma,
                rtol=1e-4,
                atol=1e-9,
            )
        # Single-step terminated episodes: reward-only GAE and returns reduce to the reward,
        # while standard GAE subtracts the critic value of the state.
        np.testing.assert_allclose(
            arrays[f"arm_{arm_label}__reward_only_gae__raw_advantage"],
            arrays[f"arm_{arm_label}_reward"],
            rtol=1e-6,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            arrays[f"arm_{arm_label}__discounted_return__raw_advantage"],
            arrays[f"arm_{arm_label}_reward"],
            rtol=1e-6,
            atol=1e-7,
        )
    np.testing.assert_allclose(
        arrays["arm_r0__standard_gae__raw_advantage"],
        arrays["arm_r0_reward"] - state_values,
        rtol=1e-6,
        atol=1e-7,
    )

    attribution = summary["attribution"]
    assert attribution["attribution"] in {
        "critic_gae_common_term_dominated",
        "temporal_credit_structure_sensitivity",
        "reward_batch_collinearity",
        None,
    }
    assert (
        attribution["gate_c_endpoint_identifiable_under_standard_gae"]
        == attribution["endpoint"]["standard_gae"]["identifiable"]
    )
    for form in CREDIT_FORMS:
        assert set(attribution["endpoint"][form]) == {
            "actor_head_cosine",
            "normalized_advantage_rmse",
            "sign_flip_fraction",
            "identifiable",
        }
    report = render_ablation_report(summary)
    assert "Task C4: critic / GAE common-term ablation" in report
    assert "C4 attribution:" in report
    for episode, (training, audit) in zip(episodes, snapshots, strict=True):
        assert (episode.training == training).all()
        assert (episode.audit == audit).all()
    assert all(torch.equal(v, parameters[k]) for k, v in policy.state_dict().items())
    assert updater.optimizer.state == {}
    assert updater.completed_optimizer_steps == 0
    assert all(p.grad is None for p in policy.parameters())


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
    assert not dominated["gate_c_endpoint_identifiable_under_standard_gae"]

    collinear = evaluate_attribution(
        [_pair(**fails, form=form) for form in CREDIT_FORMS], thresholds
    )
    assert collinear["attribution"] == "reward_batch_collinearity"
    assert not collinear["gate_c_endpoint_identifiable_under_standard_gae"]

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
    assert identifiable["gate_c_endpoint_identifiable_under_standard_gae"]

    with pytest.raises(ValueError, match="one endpoint pair per credit form"):
        evaluate_attribution([_pair(**fails, form="standard_gae")], thresholds)


def test_ablation_config_rejects_invalid_axes():
    with pytest.raises(ValueError, match="quantiles"):
        _study(quantiles=[0.25, 1.0])
    with pytest.raises(ValueError, match="quantiles"):
        _study(quantiles=[0.0, 0.5, 0.5, 1.0])
