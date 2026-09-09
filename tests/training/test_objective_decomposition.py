from __future__ import annotations

import numpy as np
import pytest
import torch

from eco_planner.analysis.reporting.fixed import render_decomposition_report
from eco_planner.experiments.objective_decomposition import (
    DecompositionConfig,
    analyze_decomposition,
    energy_only_reward,
    evaluate_gate,
)
from eco_planner.experiments.objective_decomposition_runner import verify_expected_calibration
from eco_planner.rl.optimization import PPOUpdater
from eco_planner.rl.policy import ExplorationPolicy
from tests.training.test_ppo import (
    _behavior_policy_episode,
    _episode,
    _policy_config,
    _ppo_config,
)
from tests.training.test_reward import _no_energy_config


def _study(**overrides: object) -> DecompositionConfig:
    values: dict[str, object] = {
        "lambdas": [2.0, 8.0],
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
        "gate": {
            "endpoint_max_actor_head_cosine": 0.99,
            "min_normalized_advantage_rmse": 0.10,
            "min_sign_flip_fraction": 0.05,
            "min_stress_fraction_of_endpoint_separation": 0.5,
        },
    }
    values.update(overrides)
    return DecompositionConfig.model_validate(values)


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


def test_decomposition_forms_and_gradient_identities(monkeypatch):
    torch.manual_seed(0)
    policy = ExplorationPolicy(_policy_config())
    updater = PPOUpdater(policy, _ppo_config().model_copy(update={"batch_size": 4}))
    episodes = [
        _behavior_policy_episode(policy, torch.tensor([action]), reward=0.5)
        for action in [(-0.5, 0.2), (0.3, -0.7), (-0.1, -0.4), (0.6, 0.8)]
    ]
    progress = [0.2, 0.4, 0.6, 0.8]
    energy = [0.9, 0.5, 0.7, 0.3]
    comfort = [0.1, 0.5, 0.9, 0.3]
    gates = [1.0, 0.5, 1.0, 0.25]
    for i, episode in enumerate(episodes):
        episode.audit["reward_component_progress"].fill_(progress[i])
        episode.audit["reward_component_energy"].fill_(energy[i])
        episode.audit["reward_component_comfort"].fill_(comfort[i])
        episode.audit["reward_safety_gate"].fill_(gates[i])
    snapshots = [(e.training.clone(), e.audit.clone()) for e in episodes]
    parameters = {k: v.clone() for k, v in policy.state_dict().items()}

    def forbidden(*args, **kwargs):
        pytest.fail("optimizer/scheduler step is forbidden")

    monkeypatch.setattr(updater.optimizer, "step", forbidden)
    monkeypatch.setattr(updater.scheduler, "step", forbidden)
    study = _study()
    summary, arrays = analyze_decomposition(
        updater,
        episodes,
        _no_energy_config(),
        study.lambdas,
        study.quantiles,
        np.array([0, 0, 1, 1]),
        study.gate,
    )
    assert [arm["kind"] for arm in summary["arms"]] == ["r0", "lambda", "lambda", "energy_only"]
    assert [arm["label"] for arm in summary["arms"]] == [
        "r0",
        "lambda_2",
        "lambda_8",
        "energy_only",
    ]

    for i in range(4):
        raw = arrays[f"arm_{i}_raw_advantage"]
        centered = arrays[f"arm_{i}_center_advantage"]
        normalized = arrays[f"arm_{i}_normalized_advantage"]
        np.testing.assert_allclose(centered, raw - raw.mean(), rtol=1e-6, atol=1e-9)
        np.testing.assert_allclose(normalized, centered / raw.std(ddof=1), rtol=1e-5, atol=1e-8)
        np.testing.assert_array_equal(np.sign(normalized), np.sign(centered))
        sigma = raw.std(ddof=1)
        np.testing.assert_allclose(
            arrays[f"arm_{i}_gradient_z_actor_head"],
            arrays[f"arm_{i}_gradient_center_actor_head"] / sigma,
            rtol=1e-4,
            atol=1e-9,
        )

    w2, w8 = 16 / 18, 16 / 24
    np.testing.assert_allclose(
        arrays["arm_1_reward"],
        w2 * arrays["arm_0_reward"] + (1 - w2) * arrays["arm_3_reward"],
        rtol=1e-6,
    )
    for i, w in ((1, w2), (2, w8)):
        np.testing.assert_allclose(
            arrays[f"arm_{i}_raw_advantage"],
            w * arrays["arm_0_raw_advantage"] + (1 - w) * arrays["arm_3_raw_advantage"],
            rtol=1e-5,
            atol=1e-7,
        )
        s0 = arrays["arm_0_raw_advantage"].std(ddof=1)
        s3 = arrays["arm_3_raw_advantage"].std(ddof=1)
        si = arrays[f"arm_{i}_raw_advantage"].std(ddof=1)
        np.testing.assert_allclose(
            arrays[f"arm_{i}_gradient_z_actor_head"],
            (
                w * s0 * arrays["arm_0_gradient_z_actor_head"]
                + (1 - w) * s3 * arrays["arm_3_gradient_z_actor_head"]
            )
            / si,
            rtol=1e-4,
            atol=1e-9,
        )

    endpoint = next(
        pair
        for pair in summary["pairs"]
        if pair["arm_i"] == "r0" and pair["arm_j"] == "energy_only"
    )
    for entry in summary["endpoint_forms"].values():
        assert entry["pearson"] == pytest.approx(endpoint["pearson"], abs=1e-6)
        assert entry["spearman"] == pytest.approx(endpoint["spearman"], abs=1e-6)
    center_cosine = summary["endpoint_forms"]["center"]["gradients"]["actor_head"]["cosine"]
    assert center_cosine == pytest.approx(endpoint["gradients"]["actor_head"]["cosine"], abs=1e-6)

    assert summary["gate"]["gate_c_passed"] in (True, False)
    assert summary["gate"]["attribution"] is not None or summary["gate"]["gate_c_passed"]
    report = render_decomposition_report(summary)
    assert "Endpoint attribution: R0 vs Energy-only" in report
    assert "Verdict: **PASSED**" in report or "Verdict: **FAILED**" in report
    for episode, (training, audit) in zip(episodes, snapshots, strict=True):
        assert (episode.training == training).all()
        assert (episode.audit == audit).all()
    assert all(torch.equal(v, parameters[k]) for k, v in policy.state_dict().items())
    assert updater.optimizer.state == {}
    assert updater.completed_optimizer_steps == 0
    assert all(p.grad is None for p in policy.parameters())


def _pair(
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
    thresholds = _study().gate
    collinear = evaluate_gate(
        _pair(0.9999), _forms(0.9999, 0.9999), [_pair(0.99995), _pair(0.99999)], thresholds
    )
    assert not collinear["gate_c_passed"]
    assert collinear["attribution"] == "objective_batch_collinearity"

    suppressed = evaluate_gate(
        _pair(0.9995), _forms(0.5, 0.9995), [_pair(0.9997), _pair(0.9998)], thresholds
    )
    assert not suppressed["gate_c_passed"]
    assert suppressed["attribution"] == "normalization_suppressed_identifiability"

    endpoint = _pair(0.9, rmse=0.3, sign_flip=0.1)
    too_weak = evaluate_gate(endpoint, _forms(0.9, 0.9), [_pair(0.995), _pair(0.99)], thresholds)
    assert not too_weak["gate_c_passed"]
    assert too_weak["attribution"] == "relative_scale_lambda_parameterization_too_weak"

    passed = evaluate_gate(
        endpoint,
        _forms(0.9, 0.9),
        [_pair(0.97), _pair(0.95, lambda_j=64.0), _pair(0.92, lambda_j=256.0)],
        thresholds,
    )
    assert passed["gate_c_passed"]
    assert passed["attribution"] is None
    assert passed["endpoint"]["identifiable"]
    assert passed["stress_angular_distance_nondecreasing"]

    non_monotone = evaluate_gate(
        endpoint,
        _forms(0.9, 0.9),
        [_pair(0.95), _pair(0.97, lambda_j=64.0), _pair(0.92, lambda_j=256.0)],
        thresholds,
    )
    assert not non_monotone["gate_c_passed"]
    assert non_monotone["attribution"] is None
    assert any("not nondecreasing" in r for r in non_monotone["failure_reasons"])


def test_expected_calibration_verification_matches_and_fails():
    study = _study()
    payload = _no_energy_config().model_dump()
    payload["progress"]["full_score_delta_m"] = 1.0
    payload["comfort"]["jerk_limit_mps3"] = 5.0
    matched = _no_energy_config().model_validate(payload)
    checks = verify_expected_calibration(matched, study)
    assert set(checks) == {
        "progress.full_score_delta_m",
        "comfort.longitudinal_acceleration_limit_mps2",
        "comfort.lateral_acceleration_limit_mps2",
        "comfort.jerk_limit_mps3",
        "comfort.yaw_rate_limit_radps",
    }
    payload["comfort"]["jerk_limit_mps3"] = 111.7
    diverged = _no_energy_config().model_validate(payload)
    with pytest.raises(ValueError, match="E-034 frozen value"):
        verify_expected_calibration(diverged, study)


def test_decomposition_config_rejects_invalid_axes():
    with pytest.raises(ValueError, match="positive and strictly increasing"):
        _study(lambdas=[0.0, 16.0])
    with pytest.raises(ValueError, match="positive and strictly increasing"):
        _study(lambdas=[64.0, 16.0])
    with pytest.raises(ValueError, match="quantiles"):
        _study(quantiles=[0.25, 1.0])
