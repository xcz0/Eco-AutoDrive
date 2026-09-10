from __future__ import annotations

import numpy as np
import pytest
import torch

from eco_planner.experiments.reward.fixed_batch.gradients import actor_gradients
from eco_planner.experiments.reward.fixed_batch.rewards import COMPONENTS, reward_profile, reweight
from eco_planner.experiments.reward.lambda_identifiability.diagnostics import (
    advantage_comparison,
    analyze,
    cosine,
)
from eco_planner.rl.optimization import PPOUpdater, compute_episode_gae
from eco_planner.rl.optimization.ppo import build_ppo_batch, normalize_full_batch_advantage
from eco_planner.rl.policy import ExplorationPolicy
from eco_planner.rl.reward import (
    evaluate_plannerrft_energy_step,
    evaluate_plannerrft_no_energy_step,
)
from tests.training.test_ppo import (
    _behavior_policy_episode,
    _episode,
    _policy_config,
    _ppo_config,
)
from tests.training.test_reward import _metrics, _no_energy_config


@pytest.mark.parametrize("weight", [0.0, 1.0, 16.0])
@pytest.mark.parametrize("gate", [0.0, 0.3, 1.0])
def test_reweight_matches_reward_profiles_without_mutating_source(weight, gate):
    profile = reward_profile(_no_energy_config(), weight)
    evaluator = (
        evaluate_plannerrft_no_energy_step if weight == 0 else evaluate_plannerrft_energy_step
    )
    result = evaluator(profile, _metrics())
    episode = _episode(reward=0.25, terminated=True, truncated=False, bootstrap=0.0)
    for name in COMPONENTS:
        episode.audit[f"reward_component_{name}"].fill_(getattr(result.components, name))
    episode.audit["reward_safety_gate"].fill_(gate)
    original = episode.training.clone()
    matched = reweight(episode, profile)
    assert matched.training["next", "reward"].item() == pytest.approx(
        result.base_total * gate,
        abs=1e-7,
    )
    assert (episode.training == original).all()
    assert matched.reward_profile == profile.name


@pytest.mark.parametrize("terminated,truncated", [(True, False), (False, True), (False, False)])
def test_reweight_preserves_bootstrap_and_episode_gae(terminated, truncated):
    episode = _episode(
        reward=0.25,
        terminated=terminated,
        truncated=truncated,
        bootstrap=0.0 if terminated else 2.0,
    )
    matched = reweight(episode, _no_energy_config())
    trajectory = compute_episode_gae(matched, _ppo_config())
    expected = matched.training["next", "reward"] - matched.training["state_value"]
    if not terminated:
        expected = expected + 0.99 * matched.training["next", "state_value"]
    torch.testing.assert_close(trajectory["advantage"], expected)
    assert matched.tail_kind == episode.tail_kind
    torch.testing.assert_close(matched.tail_bootstrap_value, episode.tail_bootstrap_value)


def test_pair_statistics_cover_ties_signs_and_undefined_vectors():
    x = np.array([-2.0, -2.0, 0.0, 4.0])
    same = advantage_comparison(x, 3 * x)
    assert same["pearson"] == pytest.approx(1.0)
    assert same["spearman"] == pytest.approx(1.0)
    assert same["sign_flip_fraction"] == 0
    opposite = advantage_comparison(x, -x)
    assert opposite["spearman"] == pytest.approx(-1.0)
    assert opposite["sign_flip_fraction"] == 0.75
    assert opposite["zero_fraction_i"] == 0.25
    assert cosine(x, np.zeros(4)) is None
    assert advantage_comparison(np.ones(4), x)["pearson"] is None
    # Average ranks: [1.5, 1.5, 3, 4] versus [1, 2, 3.5, 3.5].
    tied = advantage_comparison(x, np.array([0.0, 1.0, 2.0, 2.0]))
    assert tied["spearman"] == pytest.approx(8 / 9)


def test_backward_only_matches_ppo_and_preserves_inputs(monkeypatch):
    torch.manual_seed(0)
    policy = ExplorationPolicy(_policy_config())
    updater = PPOUpdater(policy, _ppo_config().model_copy(update={"batch_size": 4}))
    episodes = [
        _behavior_policy_episode(policy, torch.tensor([action]), reward=0.5)
        for action in [(-0.5, 0.2), (0.3, -0.7), (-0.1, -0.4), (0.6, 0.8)]
    ]
    for i, episode in enumerate(episodes):
        episode.audit["reward_component_progress"].fill_([0.1, 0.9, 0.2, 0.8][i])
        episode.audit["reward_component_energy"].fill_([1.0, 0.0, 0.9, 0.1][i])
    snapshots = [(e.training.clone(), e.audit.clone()) for e in episodes]
    parameters = {k: v.clone() for k, v in policy.state_dict().items()}

    def forbidden(*args, **kwargs):
        pytest.fail("optimizer/scheduler step is forbidden")

    monkeypatch.setattr(updater.optimizer, "step", forbidden)
    monkeypatch.setattr(updater.scheduler, "step", forbidden)
    summary, arrays = analyze(
        updater, episodes, _no_energy_config(), [0.0, 16.0], [0.0, 0.5, 1.0], np.array([0, 0, 1, 1])
    )
    pair = summary["pairs"][0]
    assert pair["gradients"]["shared_trunk"]["cosine"] is None
    assert pair["gradients"]["shared_trunk"]["norm_ratio_j_over_i"] is None
    assert summary["arms"][0]["gradient_norms"]["shared_trunk"] == 0
    assert pair["pearson"] < 0
    np.testing.assert_allclose(
        arrays["pair_0_1_normalized_advantage_delta"],
        arrays["arm_1_normalized_advantage"] - arrays["arm_0_normalized_advantage"],
        rtol=1e-6,
    )
    for episode, (training, audit) in zip(episodes, snapshots, strict=True):
        assert (episode.training == training).all()
        assert (episode.audit == audit).all()
    assert all(torch.equal(v, parameters[k]) for k, v in policy.state_dict().items())
    assert updater.optimizer.state == {}
    assert updater.completed_optimizer_steps == 0
    assert all(p.grad is None for p in policy.parameters())

    matched = [reweight(e, _no_energy_config()) for e in episodes]
    batch = build_ppo_batch(matched, updater.config)
    normalize_full_batch_advantage(batch)
    updater.loss_module(batch)["loss_objective"].backward()
    gradient, _ = actor_gradients(policy)
    for group, value in gradient.items():
        np.testing.assert_array_equal(value, arrays[f"arm_0_gradient_{group}"])
    assert np.linalg.norm(gradient["actor_head"]) ** 2 == pytest.approx(
        np.linalg.norm(gradient["lateral"]) ** 2 + np.linalg.norm(gradient["longitudinal"]) ** 2,
        rel=1e-6,
    )


def test_positive_scale_is_removed_by_full_batch_normalization():
    from tensordict import TensorDict

    a = TensorDict({"advantage": torch.tensor([[-2.0], [1.0], [3.0]])}, batch_size=[3])
    b = a.clone()
    b["advantage"] = b["advantage"] * 7
    normalize_full_batch_advantage(a)
    normalize_full_batch_advantage(b)
    torch.testing.assert_close(a["advantage"], b["advantage"])
