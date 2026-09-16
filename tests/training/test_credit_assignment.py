from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from eco_planner.analysis.statistics import advantage_comparison, cosine
from eco_planner.experiments.credit.decisions import (
    CREDIT_FORMS,
    AttributionThresholds,
    GateThresholds,
    evaluate_attribution,
    evaluate_gate,
)
from eco_planner.rl.optimization import compute_episode_gae
from eco_planner.rl.optimization.credit import (
    credit_batch,
    discounted_return_batch,
    zero_critic_values,
)
from eco_planner.rl.optimization.ppo import normalize_full_batch_advantage
from eco_planner.rl.reward import (
    PlannerRFTNoEnergyRewardConfig,
    evaluate_plannerrft_energy_step,
    evaluate_plannerrft_no_energy_step,
)
from eco_planner.rl.reward.calibration import rescore
from eco_planner.rl.reward.calibration_config import EnergyBandConfig
from eco_planner.rl.reward.reweighting import (
    COMPONENTS,
    energy_only_reward,
    reward_profile,
    reweight,
)
from eco_planner.rl.rollout import (
    RolloutEpisodeBuilder,
    RolloutProvenance,
    build_training_decision,
)
from tests.training.test_ppo import (
    _context,
    _decision_audit,
    _episode,
    _execution_audit,
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


def test_positive_scale_is_removed_by_full_batch_normalization():
    from tensordict import TensorDict

    a = TensorDict({"advantage": torch.tensor([[-2.0], [1.0], [3.0]])}, batch_size=[3])
    b = a.clone()
    b["advantage"] = b["advantage"] * 7
    normalize_full_batch_advantage(a)
    normalize_full_batch_advantage(b)
    torch.testing.assert_close(a["advantage"], b["advantage"])


@pytest.mark.parametrize("gate", [0.0, 0.3, 1.0])
def test_energy_only_reward_composes_gate_times_energy_and_preserves_source(gate):
    episode = _episode(reward=0.25, terminated=True, truncated=False, bootstrap=0.0)
    episode.audit["reward_component_energy"].fill_(0.4)
    episode.audit["reward_safety_gate"].fill_(gate)
    training, audit = episode.training.clone(), episode.audit.clone()
    matched = energy_only_reward(episode)
    assert matched.training["next", "reward"].item() == pytest.approx(0.4 * gate)
    assert matched.audit["reward_base_total"].item() == pytest.approx(0.4)
    assert matched.audit["reward_total"].item() == pytest.approx(0.4 * gate)
    for key in episode.training.keys(include_nested=True, leaves_only=True):
        if key != ("next", "reward"):
            torch.testing.assert_close(matched.training[key], episode.training[key], rtol=0, atol=0)
    for key in episode.audit.keys():
        if key not in {"reward_base_total", "reward_total"}:
            torch.testing.assert_close(matched.audit[key], episode.audit[key], rtol=0, atol=0)
    assert (episode.training == training).all()
    assert (episode.audit == audit).all()
    assert matched.tail_kind == episode.tail_kind
    torch.testing.assert_close(matched.tail_bootstrap_value, episode.tail_bootstrap_value)


def _objective_pair(
    cosine: float,
    *,
    rmse: float = 0.2,
    sign_flip: float = 0.0,
    norm_ratio: float = 1.0,
    lambda_j: float | None = 16.0,
) -> dict:
    return {
        "lambda_j": lambda_j,
        "sign_flip_fraction": sign_flip,
        "normalized_advantage_rmse": rmse,
        "gradients": {"actor_head": {"cosine": cosine, "norm_ratio_j_over_i": norm_ratio}},
    }


def _forms(raw_cosine: float, center_cosine: float) -> dict:
    return {
        form: {"gradients": {"actor_head": {"cosine": value}}}
        for form, value in (("raw", raw_cosine), ("center", center_cosine))
    }


def test_gate_labels_and_verdicts():
    thresholds = _objective_study().gate
    collinear = evaluate_gate(
        _objective_pair(0.9999),
        _forms(0.9999, 0.9999),
        [_objective_pair(0.99995), _objective_pair(0.99999)],
        thresholds,
    )
    assert not collinear["passed"]
    assert collinear["attribution"] == "objective_batch_collinearity"

    suppressed = evaluate_gate(
        _objective_pair(0.9995),
        _forms(0.5, 0.9995),
        [_objective_pair(0.9997), _objective_pair(0.9998)],
        thresholds,
    )
    assert not suppressed["passed"]
    assert suppressed["attribution"] == "normalization_suppressed_identifiability"

    endpoint = _objective_pair(0.9, rmse=0.3, sign_flip=0.1)
    too_weak = evaluate_gate(
        endpoint, _forms(0.9, 0.9), [_objective_pair(0.995), _objective_pair(0.99)], thresholds
    )
    assert not too_weak["passed"]
    assert too_weak["attribution"] == "relative_scale_lambda_parameterization_too_weak"

    passed = evaluate_gate(
        endpoint,
        _forms(0.9, 0.9),
        [
            _objective_pair(0.97),
            _objective_pair(0.95, lambda_j=64.0),
            _objective_pair(0.92, lambda_j=256.0),
        ],
        thresholds,
    )
    assert passed["passed"]
    assert passed["attribution"] is None
    assert passed["endpoint"]["identifiable"]
    assert passed["stress_angular_distance_nondecreasing"]

    non_monotone = evaluate_gate(
        endpoint,
        _forms(0.9, 0.9),
        [
            _objective_pair(0.95),
            _objective_pair(0.97, lambda_j=64.0),
            _objective_pair(0.92, lambda_j=256.0),
        ],
        thresholds,
    )
    assert not non_monotone["passed"]
    assert non_monotone["attribution"] is None
    assert any("not nondecreasing" in r for r in non_monotone["failure_reasons"])


def _band_profile() -> PlannerRFTNoEnergyRewardConfig:
    payload = _no_energy_config().model_dump(mode="python")
    payload["energy"] = {
        "mode": "calibrated_band",
        "reference_ml_per_km": 50.0,
        "minimum_step_distance_m": 0.01,
        "band_full_score_ml_per_km": 44.0,
        "band_zero_score_ml_per_km": 50.0,
    }
    return PlannerRFTNoEnergyRewardConfig.model_validate(payload)


def test_rescore_recomputes_energy_only_in_band_mode():
    episodes = []
    for intensity, valid in ((46.0, True), (43.0, True), (47.0, False)):
        episode = _episode(reward=0.25, terminated=True, truncated=False, bootstrap=0.0)
        episode.audit["executed_fuel_proxy_ml_per_km"].fill_(intensity)
        episode.audit["energy_distance_valid"].fill_(valid)
        episodes.append(episode)

    untouched = [rescore(episode, _no_energy_config()) for episode in episodes]
    for episode, matched in zip(episodes, untouched, strict=True):
        torch.testing.assert_close(
            matched.audit["reward_component_energy"],
            episode.audit["reward_component_energy"],
            rtol=0,
            atol=0,
        )

    matched = [rescore(episode, _band_profile()) for episode in episodes]
    assert matched[0].audit["reward_component_energy"].item() == pytest.approx((50.0 - 46.0) / 6.0)
    assert matched[1].audit["reward_component_energy"].item() == 1.0
    assert matched[2].audit["reward_component_energy"].item() == 0.0

    endpoint = energy_only_reward(matched[0])
    assert endpoint.training["next", "reward"].item() == pytest.approx((50.0 - 46.0) / 6.0)
    r0 = reweight(matched[0], _no_energy_config())
    torch.testing.assert_close(
        r0.audit["reward_base_total"], untouched[0].audit["reward_base_total"], rtol=0, atol=0
    )
    stress = reweight(matched[0], reward_profile(_band_profile(), 16.0))
    assert stress.audit["reward_base_total"].item() == pytest.approx(
        (5.0 + 5.0 + 2.0 + 4.0 + 16.0 * ((50.0 - 46.0) / 6.0)) / 32.0
    )


def _band_study(**overrides: object) -> EnergyBandConfig:
    values: dict[str, object] = {
        "full_score_intensity_quantile": 0.10,
        "zero_score_intensity_quantile": 0.90,
    }
    values.update(overrides)
    return EnergyBandConfig.model_validate(values)


def test_energy_band_config_validation():
    assert _band_study().full_score_intensity_quantile == 0.1
    with pytest.raises(ValueError):
        _band_study(full_score_intensity_quantile=0.6)
    with pytest.raises(ValueError):
        _band_study(zero_score_intensity_quantile=0.4)


def _objective_study():
    return SimpleNamespace(
        gate=GateThresholds(
            endpoint_max_actor_head_cosine=0.95,
            min_normalized_advantage_rmse=0.1,
            min_sign_flip_fraction=0.1,
            min_stress_fraction_of_endpoint_separation=0.5,
        )
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


def _credit_pair(cosine: float, *, rmse: float = 0.2, sign_flip: float = 0.0, form: str) -> dict:
    return {
        "credit_form": form,
        "sign_flip_fraction": sign_flip,
        "normalized_advantage_rmse": rmse,
        "gradients": {"actor_head": {"cosine": cosine, "norm_ratio_j_over_i": 1.0}},
    }


def test_attribution_branches():
    thresholds = _credit_study().gate
    fails = {"cosine": 0.9999, "rmse": 0.01, "sign_flip": 0.0}
    passes = {"cosine": 0.9, "rmse": 0.3, "sign_flip": 0.1}
    dominated = evaluate_attribution(
        [
            _credit_pair(**fails, form="standard_gae"),
            _credit_pair(**passes, form="reward_only_gae"),
            _credit_pair(**passes, form="discounted_return"),
        ],
        thresholds,
    )
    assert dominated["attribution"] == "critic_gae_common_term_dominated"
    assert not dominated["endpoint_identifiable_under_standard_gae"]

    collinear = evaluate_attribution(
        [_credit_pair(**fails, form=form) for form in CREDIT_FORMS], thresholds
    )
    assert collinear["attribution"] == "reward_batch_collinearity"
    assert not collinear["endpoint_identifiable_under_standard_gae"]

    temporal = evaluate_attribution(
        [
            _credit_pair(**fails, form="standard_gae"),
            _credit_pair(**fails, form="reward_only_gae"),
            _credit_pair(**passes, form="discounted_return"),
        ],
        thresholds,
    )
    assert temporal["attribution"] == "temporal_credit_structure_sensitivity"

    identifiable = evaluate_attribution(
        [_credit_pair(**passes, form=form) for form in CREDIT_FORMS], thresholds
    )
    assert identifiable["attribution"] is None
    assert identifiable["endpoint_identifiable_under_standard_gae"]

    with pytest.raises(ValueError, match="one endpoint pair per credit form"):
        evaluate_attribution([_credit_pair(**fails, form="standard_gae")], thresholds)


def _credit_study():
    return SimpleNamespace(
        gate=AttributionThresholds(
            endpoint_max_actor_head_cosine=0.95,
            min_normalized_advantage_rmse=0.1,
            min_sign_flip_fraction=0.1,
        )
    )
