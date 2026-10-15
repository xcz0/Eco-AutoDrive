"""Synthetic storage and offline diagnostic chain, without simulation or research artifacts."""

import numpy as np
import pytest
import torch

from eco_planner.artifacts import write_json
from eco_planner.planning.policy import (
    policy_state_hash,
)
from eco_planner.rl.optimization.diagnostic_runtime import restore_runtime
from eco_planner.rl.reward import rescore, reweight
from eco_planner.rl.rollout.fixed_batch import (
    load_batch,
    load_fixed_batch,
)
from tests.training.artifact_helpers import diagnostic_inputs, persist_diagnostic_batch
from tests.training.helpers import _behavior_policy_episode
from tests.training.reward_helpers import _no_energy_config


@pytest.fixture
def fixed_source(tmp_path):
    config, policy = diagnostic_inputs(_no_energy_config())
    slots = [[], []]
    for index, action in enumerate([(-0.5, 0.2), (0.3, -0.7), (-0.1, -0.4), (0.6, 0.8)]):
        episode = _behavior_policy_episode(policy, torch.tensor([action]), reward=0.5)
        episode.audit["reward_substep_route_progress_delta_m"].fill_(0.2 * (index + 1))
        episode.audit["reward_substep_component_energy"].fill_([0.9, 0.5, 0.7, 0.3][index])
        episode.audit["reward_substep_step_distance_m"].fill_(1.0)
        episode.audit["reward_substep_executed_fuel_proxy_step_energy_ml"].fill_(
            [46.0, 48.0, 47.0, 49.0][index] / 1000.0
        )
        episode.audit["reward_substep_energy_distance_valid"].fill_(True)
        episode.audit["reward_substep_safety_gate"].fill_([1.0, 0.5, 1.0, 0.25][index])
        episode.audit["map_seed"].fill_(config.scenarios[index // 2].seed)
        slots[index // 2].append(rescore(reweight(episode, config.reward), config.reward))
    source = tmp_path / "batch"
    episodes, samples = persist_diagnostic_batch(source, config, policy, slots)
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


def test_fixed_batch_preserves_sample_order_and_restores_runtime(fixed_source):
    source, episodes, samples = fixed_source
    restored = load_fixed_batch(source)
    assert restored.samples == samples
    np.testing.assert_array_equal(restored.scenario_ids, [0, 0, 1, 1])
    assert len(restored.episodes) == len(episodes)
    assert [e.training["next", "reward"].item() for e in restored.episodes] == [
        e.training["next", "reward"].item() for e in episodes
    ]
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


def test_credit_preserves_ppo_gradients_and_objective_identities(fixed_source, monkeypatch):
    from eco_planner.experiments.credit.config import CreditStudyConfig
    from eco_planner.experiments.credit.runner import measure
    from eco_planner.rl.optimization import build_ppo_batch, normalize_full_batch_advantage
    from eco_planner.rl.optimization.gradients import actor_gradients

    source, episodes, _ = fixed_source
    batch = load_fixed_batch(source)
    runtime = restore_runtime(source, batch)
    updater = runtime.updater
    config = CreditStudyConfig.model_validate(
        {
            "arms": [
                {"label": "r0", "weight": 0.0},
                {"label": "lambda_2", "weight": 2.0},
                {"label": "energy_only", "weight": "energy_only"},
            ],
            "advantage_forms": ["raw", "center", "z"],
            "credit_forms": ["standard_gae", "reward_only_gae", "discounted_return"],
            "value_target_ddof": 1,
            "quantiles": [0.0, 0.5, 1.0],
            "calibration": None,
            "energy_band": None,
            "frozen_energy_band": None,
            "objective_gate": None,
            "attribution_gate": None,
        }
    )
    monkeypatch.setattr(updater.optimizer, "step", lambda: pytest.fail("optimizer step"))
    monkeypatch.setattr(updater.scheduler, "step", lambda: pytest.fail("scheduler step"))
    snapshots = [(e.training.clone(), e.audit.clone()) for e in episodes]
    _, arrays = measure(updater, episodes, batch.config.reward, config, batch.scenario_ids)
    runtime.verify_unchanged()
    for arm in ("r0", "lambda_2", "energy_only"):
        for credit in config.credit_forms:
            prefix = f"{arm}__{credit}__"
            raw = arrays[prefix + "raw_advantage"]
            center = arrays[prefix + "center_advantage"]
            np.testing.assert_allclose(center, raw - raw.mean(), rtol=1e-6, atol=1e-8)
            np.testing.assert_allclose(
                arrays[prefix + "normalized_advantage"],
                center / raw.std(ddof=1),
                rtol=1e-5,
                atol=1e-7,
            )
            np.testing.assert_allclose(
                arrays[prefix + "gradient_z_actor_head"],
                arrays[prefix + "gradient_center_actor_head"] / raw.std(ddof=1),
                rtol=1e-4,
                atol=1e-8,
            )
    weight = 16 / 18
    for credit in config.credit_forms:
        for key in ("raw_advantage", "gradient_raw_actor_head", "gradient_center_actor_head"):
            np.testing.assert_allclose(
                arrays[f"lambda_2__{credit}__{key}"],
                weight * arrays[f"r0__{credit}__{key}"]
                + (1 - weight) * arrays[f"energy_only__{credit}__{key}"],
                rtol=1e-4,
                atol=1e-7,
            )
    reference = build_ppo_batch(
        [reweight(e, batch.config.reward) for e in episodes], updater.config
    )
    normalize_full_batch_advantage(reference)
    updater.loss_module(reference)["loss_objective"].backward()
    gradients, _ = actor_gradients(updater.policy)
    for group, values in gradients.items():
        np.testing.assert_array_equal(values, arrays[f"r0__standard_gae__gradient_z_{group}"])
    for episode, (training, audit) in zip(episodes, snapshots, strict=True):
        assert (episode.training == training).all() and (episode.audit == audit).all()


def test_missing_gradient_and_sample_mismatch_are_errors(fixed_source, tmp_path):
    from eco_planner._repository import CONFIG_ROOT
    from eco_planner.experiments.credit.runner import run
    from eco_planner.experiments.fixed_batch import fixed

    source, _, _ = fixed_source
    output = tmp_path / "credit"
    run(source, CONFIG_ROOT / "experiments/credit/sensitivity.yaml", output, figures=False)
    with np.load(output / "diagnostics.npz") as archive:
        arrays = {
            key: archive[key] for key in archive.files if not key.endswith("gradient_z_actor_head")
        }
    np.savez(output / "diagnostics.npz", **arrays)
    with pytest.raises((ValueError, KeyError)):
        fixed(output)
