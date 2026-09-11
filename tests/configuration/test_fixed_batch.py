"""Synthetic storage and offline diagnostic chain, without simulation or research artifacts."""

import json

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from eco_planner.artifacts import write_json
from eco_planner.experiments.reward.calibration import run as run_calibration
from eco_planner.experiments.reward.critic_gae_ablation.runner import run as run_ablation
from eco_planner.experiments.reward.fixed_batch.artifacts import (
    load_batch,
    load_fixed_batch,
    verify_reference,
    write_batch,
)
from eco_planner.experiments.reward.fixed_batch.calibration import calibrate, raw_arrays, rescore
from eco_planner.experiments.reward.fixed_batch.rewards import reweight
from eco_planner.experiments.reward.fixed_batch.runtime import restore_runtime
from eco_planner.experiments.reward.lambda_identifiability import run as run_lambda
from eco_planner.experiments.reward.objective_decomposition.runner import run as run_decomposition
from eco_planner.rl.artifacts import policy_state_hash
from eco_planner.rl.config import parse_training_config
from eco_planner.rl.optimization import save_exploration_policy_checkpoint
from eco_planner.rl.policy import ExplorationPolicy
from tests.analysis.test_reports import assert_report
from tests.training.test_critic_gae_ablation import _study as ablation_config
from tests.training.test_objective_decomposition import _study as decomposition_config
from tests.training.test_ppo import _behavior_policy_episode, _policy_config, _ppo_config
from tests.training.test_reward import _no_energy_config


@pytest.fixture
def fixed_source(tmp_path, compose_config):
    raw = compose_config(
        "jobs/training/ppo",
        [
            "components/resources=rtx3050_laptop",
            "runtime.seed=0",
            "runtime.accelerator=cpu",
            "training.replay_id=0",
            "training.transitions_per_environment=2",
            "training.update_count=1",
        ],
    )
    raw.policy = _policy_config().model_dump()
    raw.ppo = (
        _ppo_config()
        .model_copy(update={"batch_size": 4, "scheduler_total_optimizer_steps": 2})
        .model_dump()
    )
    raw.reward = _no_energy_config().model_dump()
    config = parse_training_config(raw)
    with torch.random.fork_rng():
        torch.manual_seed(0)
        policy = ExplorationPolicy(config.policy)
    slots = [[], []]
    for index, action in enumerate([(-0.5, 0.2), (0.3, -0.7), (-0.1, -0.4), (0.6, 0.8)]):
        episode = _behavior_policy_episode(policy, torch.tensor([action]), reward=0.5)
        episode.audit["route_progress_delta_m"].fill_(0.2 * (index + 1))
        episode.audit["reward_component_energy"].fill_([0.9, 0.5, 0.7, 0.3][index])
        episode.audit["executed_fuel_proxy_ml_per_km"].fill_([46.0, 48.0, 47.0, 49.0][index])
        episode.audit["reward_safety_gate"].fill_([1.0, 0.5, 1.0, 0.25][index])
        episode.audit["map_seed"].fill_(config.scenarios[index // 2].seed)
        slots[index // 2].append(rescore(reweight(episode, config.reward), config.reward))
    source = tmp_path / "batch"
    source.mkdir()
    episodes, samples = write_batch(source, slots, config.scenarios)
    OmegaConf.save(raw, source / "resolved_config.yaml", resolve=True)
    save_exploration_policy_checkpoint(source / "policy-initial.pt", policy)
    write_json(source / "runtime_metadata.json", {"fixture": "synthetic CPU"})
    write_json(
        source / "summary.json",
        {
            "kind": "fixed-batch",
            "optimizer_steps": 0,
            "initial_policy_hash": policy_state_hash(policy),
            "sample_count": len(samples),
            "episode_count": len(episodes),
        },
    )
    return source, episodes, samples


def test_fixed_batch_roundtrip(fixed_source):
    source, episodes, samples = fixed_source
    restored = load_fixed_batch(source)
    assert restored.samples == samples
    np.testing.assert_array_equal(restored.scenario_ids, [0, 0, 1, 1])
    for before, after in zip(episodes, restored.episodes, strict=True):
        for key in before.training.keys(include_nested=True, leaves_only=True):
            torch.testing.assert_close(before.training[key], after.training[key], rtol=0, atol=0)
            assert before.training[key].dtype == after.training[key].dtype
        for key in after.audit.keys():
            torch.testing.assert_close(before.audit[key], after.audit[key], rtol=0, atol=0)
        assert before.tail_kind == after.tail_kind
        torch.testing.assert_close(before.tail_bootstrap_value, after.tail_bootstrap_value)
    runtime = restore_runtime(source, restored)
    runtime.verify_unchanged()
    with torch.no_grad():
        next(runtime.updater.policy.parameters()).add_(1)
    with pytest.raises(RuntimeError, match="changed the policy"):
        runtime.verify_unchanged()


def test_batch_rejects_out_of_order_episode(fixed_source):
    source, _, samples = fixed_source
    write_json(source / "sample_index.json", {"samples": [samples[0], samples[1], samples[0]]})
    with pytest.raises(ValueError, match="out of order"):
        load_batch(source)


@pytest.mark.parametrize("lambdas", [[2.0], [2.0, 8.0, 16.0]])
def test_full_offline_chain_and_reference_endpoints(fixed_source, tmp_path, lambdas):
    source, episodes, _ = fixed_source
    before = {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    axes = {"lambdas": [0.0, 2.0, 8.0], "quantiles": [0.0, 0.5, 1.0]}
    lambda_config = tmp_path / "lambda.yaml"
    OmegaConf.save(OmegaConf.create(axes), lambda_config)
    lambda_dir = tmp_path / "lambda"
    run_lambda(source, lambda_config, lambda_dir, figures=False)
    cal_config = tmp_path / "cal.yaml"
    calibration = {**axes, "progress_target_score": 0.6, "comfort_target_score": 0.6}
    OmegaConf.save(OmegaConf.create(calibration), cal_config)
    run_calibration(source, lambda_dir, cal_config, tmp_path / "calibration", figures=False)
    base = _no_energy_config()
    calibrated = calibrate(raw_arrays(episodes), base, decomposition_config())
    expected = {
        "full_score_delta_m": calibrated.progress.full_score_delta_m,
        **{
            key: getattr(calibrated.comfort, key)
            for key in (
                "longitudinal_acceleration_limit_mps2",
                "lateral_acceleration_limit_mps2",
                "jerk_limit_mps3",
                "yaw_rate_limit_radps",
            )
        },
    }
    decomp_config = tmp_path / "decomp.yaml"
    OmegaConf.save(
        OmegaConf.create(
            decomposition_config(lambdas=lambdas, expected_calibration=expected).model_dump()
        ),
        decomp_config,
    )
    decomp_dir = tmp_path / "decomp"
    run_decomposition(source, decomp_config, decomp_dir, figures=False)
    intensity = np.asarray([46.0, 48.0, 47.0, 49.0])
    band = {
        "full_score_intensity_quantile": 0.10,
        "zero_score_intensity_quantile": 0.90,
        "expected_full_score_ml_per_km": float(np.quantile(intensity, 0.10)),
        "expected_zero_score_ml_per_km": float(np.quantile(intensity, 0.90)),
        "match_tolerance": {"rtol": 1e-6, "atol": 0.0},
    }
    band_config = tmp_path / "decomp-band.yaml"
    OmegaConf.save(
        OmegaConf.create(
            decomposition_config(
                lambdas=lambdas, expected_calibration=expected, energy_band=band
            ).model_dump()
        ),
        band_config,
    )
    band_dir = tmp_path / "decomp-band"
    run_decomposition(source, band_config, band_dir, figures=False)
    band_summary = json.loads((band_dir / "summary.json").read_text())
    assert band_summary["calibrated_reward"]["energy"]["mode"] == "calibrated_band"
    assert band_summary["energy_band_verification"]["energy.band_full_score_ml_per_km"][
        "actual"
    ] == pytest.approx(float(np.quantile(intensity, 0.10)))
    band_scores = np.clip(
        (float(np.quantile(intensity, 0.90)) - np.asarray([46.0, 48.0, 47.0, 49.0]))
        / (float(np.quantile(intensity, 0.90)) - float(np.quantile(intensity, 0.10))),
        0.0,
        1.0,
    )
    assert band_summary["components"]["reward_component_energy"]["mean"] == pytest.approx(
        float(band_scores.mean())
    )
    assert band_summary["optimizer_steps"] == 0 and band_summary["policy_unchanged"]
    abl_config = tmp_path / "ablation.yaml"
    OmegaConf.save(
        OmegaConf.create(ablation_config(expected_calibration=expected).model_dump()), abl_config
    )
    run_ablation(source, decomp_dir, abl_config, tmp_path / "ablation", figures=False)
    for directory in (lambda_dir, tmp_path / "calibration", decomp_dir, tmp_path / "ablation"):
        assert_report(directory, figures=False)
        assert not (directory / "source").exists()
        assert not (directory / "tracked_diff.patch").exists()
        assert "git_head" in json.loads((directory / "runtime_metadata.json").read_text())
        summary = json.loads((directory / "summary.json").read_text())
        assert summary["optimizer_steps"] == 0 and summary["policy_unchanged"]
    from eco_planner.analysis.runner import analyze

    for experiment, directory in (
        ("lambda-identifiability", lambda_dir),
        ("reward-calibration", tmp_path / "calibration"),
        ("objective-decomposition", decomp_dir),
        ("critic-gae-ablation", tmp_path / "ablation"),
    ):
        report_dir = tmp_path / (experiment + "-report")
        analyze(experiment, directory, report_dir, figures=False)
        assert_report(report_dir, figures=False)
    assert before == {
        p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()
    }
    # Calibration must compare identical lambda axes, not silently use another diagnostic.
    calibration["lambdas"] = [0.0, 4.0]
    OmegaConf.save(OmegaConf.create(calibration), cal_config)
    with pytest.raises(ValueError, match="lambda axes"):
        run_calibration(source, lambda_dir, cal_config, tmp_path / "bad-calibration", figures=False)
    assert not (tmp_path / "bad-calibration").exists()


@pytest.mark.parametrize("mismatch", ["source", "policy", "samples"])
def test_reference_mismatch_is_rejected(fixed_source, tmp_path, mismatch):
    source, _, samples = fixed_source
    batch = load_fixed_batch(source)
    reference = tmp_path / "reference"
    reference.mkdir()
    summary = {
        "source_batch": str(source),
        "initial_policy_hash": batch.summary["initial_policy_hash"],
    }
    if mismatch == "source":
        summary["source_batch"] = str(tmp_path / "another-batch")
    if mismatch == "policy":
        summary["initial_policy_hash"] = "0" * 64
    write_json(reference / "summary.json", summary)
    write_json(
        reference / "sample_index.json",
        {"samples": samples[::-1] if mismatch == "samples" else samples},
    )
    with pytest.raises(ValueError, match="different|order"):
        verify_reference(reference, source, batch)
