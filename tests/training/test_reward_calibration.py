from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from eco_planner.analysis.reporting.fixed import render_report
from eco_planner.experiments.reward.calibration.config import CalibrationConfig
from eco_planner.experiments.reward.calibration.diagnostics import dynamic_range_audit
from eco_planner.experiments.reward.fixed_batch.artifacts import load_batch
from eco_planner.experiments.reward.fixed_batch.calibration import (
    calibrate,
    raw_arrays,
    rescore,
    scored_arrays,
    verify_original_components,
)
from eco_planner.experiments.reward.fixed_batch.rewards import reward_profile
from eco_planner.rl.artifacts import write_rollout_episode
from eco_planner.rl.reward.components.comfort import component_score
from tests.training.test_ppo import _episode
from tests.training.test_reward import _no_energy_config


def _study():
    return CalibrationConfig(
        progress_target_score=0.6,
        comfort_target_score=0.6,
        lambdas=[0.0, 1.0, 2.0, 4.0, 8.0, 16.0],
        quantiles=[0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0],
    )


def _raw():
    return {
        "route_progress_delta_m": np.array([1.0, 1.2, 1.4]),
        "longitudinal_acceleration_mps2": np.array([-6.0, 7.0, 8.0]),
        "lateral_acceleration_mps2": np.array([1.0, 2.0, 10.0]),
        "jerk_mps3": np.array([100.0, 140.0, 1000.0]),
        "yaw_rate_radps": np.array([0.0, 0.02, 0.1]),
    }


def test_calibration_restores_dynamic_range_without_hiding_outliers():
    base, raw = _no_energy_config(), _raw()
    profile = calibrate(raw, base, _study())
    assert profile.progress.full_score_delta_m == pytest.approx(2.0)
    assert profile.comfort.longitudinal_acceleration_limit_mps2 == pytest.approx(5.0)
    assert profile.comfort.jerk_limit_mps3 == pytest.approx(100.0)
    assert profile.comfort.lateral_acceleration_limit_mps2 == 3.0
    assert profile.comfort.yaw_rate_limit_radps == 0.5
    assert base.comfort.jerk_limit_mps3 == 5.0
    scores = scored_arrays(raw, profile)
    assert scores["progress"][1] == pytest.approx(0.6)
    assert scores["comfort"][1] == pytest.approx(0.6)
    assert scores["comfort"][-1] == 0  # High jerk remains visible after calibration.
    for weight in _study().lambdas:
        arm = reward_profile(profile, weight)
        assert arm.comfort == profile.comfort
        assert arm.progress == profile.progress
        assert arm.energy == base.energy
    assert profile.ttc == base.ttc and profile.speed == base.speed and profile.gates == base.gates


@pytest.mark.parametrize("value,expected", [(0, 1), (3, 1), (4.5, 0.5), (6, 0), (100, 0)])
def test_comfort_physical_score_boundaries(value, expected):
    assert component_score(value, 3) == expected


def test_audit_keeps_original_exceedances_and_counts_tied_minima():
    raw, base, study = _raw(), _no_energy_config(), _study()
    summary, arrays = dynamic_range_audit(
        raw, base, calibrate(raw, base, study), study, np.array([0, 0, 1]), np.array([0, 1, 0])
    )
    all_stats = summary["all"]
    assert all_stats["raw"]["jerk_mps3"]["original_limit_exceeded_fraction"] == 1
    original = all_stats["scores"]["original"]
    assert original["jerk_mps3"]["minimum_fraction_including_ties"] == 1
    assert original["longitudinal_acceleration_mps2"]["minimum_fraction_including_ties"] == 1
    assert original["comfort"]["zero_fraction"] == 1
    assert summary["per_planning_cycle"]["0"]["sample_count"] == 2
    assert summary["per_scenario"]["1"]["sample_count"] == 1
    np.testing.assert_array_equal(arrays["jerk_mps3"], raw["jerk_mps3"])


@pytest.mark.parametrize("terminated,truncated", [(True, False), (False, True), (False, False)])
def test_rescore_preserves_actions_values_boundaries_and_source(terminated, truncated):
    base = _no_energy_config()
    episode = _episode(
        reward=0.4, terminated=terminated, truncated=truncated, bootstrap=0.0 if terminated else 2.0
    )
    episode.audit["jerk_mps3"].fill_(140)
    original = rescore(episode, base)
    verify_original_components([original], base)
    snapshots = original.training.clone(), original.audit.clone()
    profile = calibrate(raw_arrays([original]), base, _study())
    changed = rescore(original, profile)
    assert changed.audit["reward_component_comfort"].item() == pytest.approx(0.6)
    assert changed.training["next", "reward"].item() != original.training["next", "reward"].item()
    for key in original.training.keys(include_nested=True, leaves_only=True):
        if key != ("next", "reward"):
            torch.testing.assert_close(
                changed.training[key], original.training[key], rtol=0, atol=0
            )
    for key in original.audit.keys():
        if key not in {
            "reward_component_progress",
            "reward_component_comfort",
            "reward_base_total",
            "reward_total",
        }:
            torch.testing.assert_close(changed.audit[key], original.audit[key], rtol=0, atol=0)
    assert changed.tail_kind == original.tail_kind
    torch.testing.assert_close(changed.tail_bootstrap_value, original.tail_bootstrap_value)
    assert (original.training == snapshots[0]).all() and (original.audit == snapshots[1]).all()


def test_loader_uses_sample_order_not_lexical_slot_order_and_detects_mismatch(tmp_path):
    episodes, samples = [], []
    for slot in (2, 10):
        episode = _episode(reward=float(slot), terminated=False, truncated=False, bootstrap=2.0)
        episode.audit["map_seed"].fill_(slot)
        write_rollout_episode(tmp_path / f"updates/update-000/slot-{slot}-episode-0.npz", episode)
        episodes.append(episode)
        samples.append(
            {
                "scenario_index": slot,
                "episode_index": 0,
                "planning_cycle_index": 0,
                "map_seed": slot,
            }
        )
    (tmp_path / "sample_index.json").write_text(json.dumps({"samples": samples}))
    torch.save([e.training.to_dict() for e in episodes], tmp_path / "training-batch.pt")
    restored, index = load_batch(tmp_path)
    assert index == samples
    assert [e.training["next", "reward"].item() for e in restored] == [2, 10]
    samples[0]["map_seed"] = 99
    (tmp_path / "sample_index.json").write_text(json.dumps({"samples": samples}))
    with pytest.raises(AssertionError):
        load_batch(tmp_path)


def test_calibration_rejects_absent_positive_progress_and_nonfinite_measurements():
    raw = _raw()
    raw["route_progress_delta_m"][:] = 0
    with pytest.raises(ValueError, match="positive route progress"):
        calibrate(raw, _no_energy_config(), _study())
    raw = _raw()
    raw["jerk_mps3"][0] = np.nan
    with pytest.raises(ValueError, match="finite motion"):
        calibrate(raw, _no_energy_config(), _study())


def test_reused_batch_report_does_not_claim_new_collection():
    summary = {"undefined_reason": "zero norm", "components": {}, "arms": [], "pairs": []}
    assert "New batch;" in render_report(summary)
    report = render_report(summary, batch_origin="Reused fixed source batch")
    assert "Reused fixed source batch;" in report and "New batch;" not in report
