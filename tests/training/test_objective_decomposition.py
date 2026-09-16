from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from eco_planner.experiments.credit.decisions import GateThresholds, evaluate_gate
from eco_planner.rl.reward import PlannerRFTNoEnergyRewardConfig
from eco_planner.rl.reward.calibration import rescore
from eco_planner.rl.reward.calibration_config import EnergyBandConfig
from eco_planner.rl.reward.reweighting import (
    energy_only_reward,
    reward_profile,
    reweight,
)
from tests.training.test_ppo import (
    _episode,
)
from tests.training.test_reward import _no_energy_config


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
    assert not collinear["passed"]
    assert collinear["attribution"] == "objective_batch_collinearity"

    suppressed = evaluate_gate(
        _pair(0.9995), _forms(0.5, 0.9995), [_pair(0.9997), _pair(0.9998)], thresholds
    )
    assert not suppressed["passed"]
    assert suppressed["attribution"] == "normalization_suppressed_identifiability"

    endpoint = _pair(0.9, rmse=0.3, sign_flip=0.1)
    too_weak = evaluate_gate(endpoint, _forms(0.9, 0.9), [_pair(0.995), _pair(0.99)], thresholds)
    assert not too_weak["passed"]
    assert too_weak["attribution"] == "relative_scale_lambda_parameterization_too_weak"

    passed = evaluate_gate(
        endpoint,
        _forms(0.9, 0.9),
        [_pair(0.97), _pair(0.95, lambda_j=64.0), _pair(0.92, lambda_j=256.0)],
        thresholds,
    )
    assert passed["passed"]
    assert passed["attribution"] is None
    assert passed["endpoint"]["identifiable"]
    assert passed["stress_angular_distance_nondecreasing"]

    non_monotone = evaluate_gate(
        endpoint,
        _forms(0.9, 0.9),
        [_pair(0.95), _pair(0.97, lambda_j=64.0), _pair(0.92, lambda_j=256.0)],
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


def _study():
    return SimpleNamespace(
        gate=GateThresholds(
            endpoint_max_actor_head_cosine=0.95,
            min_normalized_advantage_rmse=0.1,
            min_sign_flip_fraction=0.1,
            min_stress_fraction_of_endpoint_separation=0.5,
        )
    )
