"""Synthetic offline critic-attribution pipeline, without simulation or research artifacts."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from eco_planner._repository import CONFIG_ROOT
from eco_planner.artifacts import write_json
from eco_planner.configuration import load_resolved_yaml_mapping
from eco_planner.experiments.training.critic_attribution.diagnostics import (
    CriticAttributionConfig,
    MaterialityThresholds,
    evaluate_materiality,
)
from eco_planner.jobs import compose_job_config
from eco_planner.planning.policy import (
    ExplorationPolicy,
    policy_state_hash,
    save_exploration_policy_checkpoint,
)
from eco_planner.rl import (
    build_ppo_batch,
    parse_training_config,
    read_rollout_episode,
    write_rollout_episode,
)
from eco_planner.rl.rollout.fixed_batch import write_batch
from tests.training.test_ppo import _behavior_policy_episode, _policy_config, _ppo_config
from tests.training.test_reward import _no_energy_config

_STUDY = CONFIG_ROOT / "experiments/training/critic-attribution.yaml"
_RUN_YAML = """\
runs:
- label: r0-seed-0
  arm: r0
  training_seed: 0
  path: run
  reward_profile: plannerrft_no_energy_calibrated_v1
baseline_credit_form: standard_gae
comparison_credit_forms:
- reward_only_gae
advantage_form: z
update_indices: [0]
provenance_rtol: 0.0001
required_run_count: 1
thresholds:
  min_actor_head_cosine: 0.99
  min_longitudinal_cosine: 0.99
  min_lateral_cosine: 0.99
  max_advantage_sign_flip_fraction: 0.05
"""


def study() -> CriticAttributionConfig:
    return CriticAttributionConfig.model_validate(load_resolved_yaml_mapping(_STUDY))


@pytest.fixture
def synthetic_source(tmp_path):
    raw = compose_job_config(
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
        .model_copy(
            update={"batch_size": 4, "minibatch_size": 4, "scheduler_total_optimizer_steps": 2}
        )
        .model_dump()
    )
    raw.reward = (
        _no_energy_config()
        .model_copy(update={"name": "plannerrft_no_energy_calibrated_v1"})
        .model_dump()
    )
    config = parse_training_config(raw)
    with torch.random.fork_rng():
        torch.manual_seed(0)
        policy = ExplorationPolicy(config.policy)
    actions = [(-0.5, 0.2), (0.3, -0.7), (-0.1, -0.4), (0.6, 0.8)]
    rewards = [0.2, 0.5, 0.8, 1.1]
    slots = [[], []]
    for index, (action, reward) in enumerate(zip(actions, rewards, strict=True)):
        episode = _behavior_policy_episode(policy, torch.tensor([action]), reward=reward)
        slots[index // 2].append(episode)
    run_dir = tmp_path / "study" / "run"
    run_dir.mkdir(parents=True)
    episodes, _samples = write_batch(run_dir, slots, config.scenarios)
    OmegaConf.save(raw, run_dir / "resolved_config.yaml", resolve=True)
    save_exploration_policy_checkpoint(run_dir / "policy-initial.pt", policy)
    batch = build_ppo_batch(episodes, config.ppo)
    write_json(run_dir / "runtime_metadata.json", {"fixture": "synthetic CPU"})
    write_json(
        run_dir / "summary.json",
        {
            "status": "completed",
            "reward_profile": "plannerrft_no_energy_calibrated_v1",
            "training_seed": 0,
            "initial_policy_hash": policy_state_hash(policy),
            "updates": [
                {
                    "update_index": 0,
                    "raw_advantage_mean": float(batch["advantage"].mean()),
                    "raw_advantage_std": float(batch["advantage"].std(unbiased=True)),
                    "mean_explained_variance": -0.01,
                    "mean_value_loss": 3.0,
                }
            ],
        },
    )
    source = tmp_path / "study"
    config_path = source / "critic-attribution.yaml"
    config_path.write_text(_RUN_YAML, encoding="utf-8")
    return source, config_path


def test_config_validation():
    config = study()
    assert [run.label for run in config.runs] == [
        "r0-seed-0",
        "r0-seed-1",
        "rstress-seed-0",
        "rstress-seed-1",
    ]
    assert config.update_indices[0] == 0
    assert config.update_indices[-1] == 49
    assert config.comparison_credit_forms == ["reward_only_gae"]
    cases = []
    duplicate = config.model_dump()
    duplicate["runs"][1]["label"] = "r0-seed-0"
    cases.append(duplicate)
    mismatch = config.model_dump()
    mismatch["runs"][0]["reward_profile"] = "plannerrft_energy_band_lam64_v1"
    cases.append(mismatch)
    unsorted = config.model_dump()
    unsorted["update_indices"] = [1, 0]
    cases.append(unsorted)
    repeated = config.model_dump()
    repeated["update_indices"] = [0, 0]
    cases.append(repeated)
    baseline = config.model_dump()
    baseline["comparison_credit_forms"] = ["standard_gae"]
    cases.append(baseline)
    wrong_count = config.model_dump()
    wrong_count["required_run_count"] = 3
    cases.append(wrong_count)
    bad_threshold = config.model_dump()
    bad_threshold["thresholds"]["min_lateral_cosine"] = 2.0
    cases.append(bad_threshold)
    for data in cases:
        with pytest.raises(ValueError):
            CriticAttributionConfig.model_validate(data)


def _record(sign_flip, actor_head, lateral, longitudinal):
    return {
        "run": "x",
        "update_index": 0,
        "advantage": {"sign_flip_fraction": sign_flip},
        "gradients": {
            "actor_head": {"cosine": actor_head},
            "lateral": {"cosine": lateral},
            "longitudinal": {"cosine": longitudinal},
        },
    }


def test_materiality_verdicts():
    thresholds = MaterialityThresholds(
        min_actor_head_cosine=0.99,
        min_longitudinal_cosine=0.99,
        min_lateral_cosine=0.99,
        max_advantage_sign_flip_fraction=0.05,
    )
    passing = evaluate_materiality({"comparison": [_record(0.0, 1.0, 0.995, 0.999)]}, thresholds)
    assert passing["verdict"] == "critic_not_material_to_actor_direction"
    assert passing["comparisons"]["comparison"]["passed"] is True
    sign = evaluate_materiality({"comparison": [_record(0.2, 1.0, 1.0, 1.0)]}, thresholds)
    assert sign["verdict"] == "critic_material_candidate"
    direction = evaluate_materiality({"comparison": [_record(0.0, 1.0, 0.5, 1.0)]}, thresholds)
    assert direction["comparisons"]["comparison"]["failures"][0]["check"] == "lateral_cosine"
    undefined = evaluate_materiality({"comparison": [_record(0.0, None, None, None)]}, thresholds)
    assert undefined["verdict"] == "critic_not_material_to_actor_direction"
    assert len(undefined["comparisons"]["comparison"]["excluded_undefined"]) == 3


def test_read_rollout_episode_roundtrip(tmp_path):
    with torch.random.fork_rng():
        torch.manual_seed(1)
        policy = ExplorationPolicy(_policy_config())
    episode = _behavior_policy_episode(policy, torch.tensor([[0.1, -0.2]]), reward=0.4)
    path = tmp_path / "episode.npz"
    write_rollout_episode(path, episode)
    restored = read_rollout_episode(path)
    assert restored.reward_profile == episode.reward_profile
    assert restored.tail_kind == episode.tail_kind
    torch.testing.assert_close(restored.tail_bootstrap_value, episode.tail_bootstrap_value)
    for key in episode.training.keys(include_nested=True, leaves_only=True):
        torch.testing.assert_close(restored.training[key], episode.training[key], rtol=0, atol=0)
    for key in episode.audit.keys():
        torch.testing.assert_close(restored.audit[key], episode.audit[key], rtol=0, atol=0)


@pytest.mark.parametrize("figures", [False, True])
def test_runner_and_offline_recompute(synthetic_source, tmp_path, figures):
    from eco_planner.analysis.runner import analyze
    from eco_planner.experiments.training.critic_attribution.runner import run
    from tests.analysis.test_reports import assert_report

    source, config_path = synthetic_source
    original = {
        path.name: path.read_bytes() for path in (source / "run").iterdir() if path.is_file()
    }
    output = tmp_path / "out"
    result = run(source, config_path, output, figures=figures)
    assert result["status"] == "completed"
    assert result["optimizer_steps"] == 0
    assert result["verdict"] in (
        "critic_not_material_to_actor_direction",
        "critic_material_candidate",
    )
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["kind"] == "training-critic-attribution"
    update = summary["runs"][0]["updates"][0]
    assert update["policy_checkpoint"] == "policy-initial.pt"
    assert update["critic"]["explained_variance"] == -0.01
    assert {
        path.name: path.read_bytes() for path in (source / "run").iterdir() if path.is_file()
    } == original

    analysis_out = tmp_path / "analysis"
    returned = analyze("training-critic-attribution", output, analysis_out, figures=figures)
    assert returned["verdict"] == summary["gate"]["verdict"]
    evidence = json.loads((analysis_out / "analysis.json").read_text(encoding="utf-8"))["evidence"]
    assert len(evidence["checkpoints"]) == 1
    checkpoint = evidence["checkpoints"][0]
    assert (
        checkpoint["advantage"]["sign_flip_fraction"]
        == update["comparisons"]["standard_gae_vs_reward_only_gae"]["advantage"][
            "sign_flip_fraction"
        ]
    )
    assert_report(analysis_out, figures=figures)


def test_recompute_detects_tampering(synthetic_source, tmp_path):
    from eco_planner.analysis.critic_attribution import recompute
    from eco_planner.experiments.training.critic_attribution.runner import run

    source, config_path = synthetic_source
    output = tmp_path / "out"
    run(source, config_path, output, figures=False)
    with np.load(output / "diagnostics.npz", allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}
    key = next(key for key in arrays if key.endswith("__raw_advantage"))
    arrays[key] = arrays[key] * 0.0
    np.savez(output / "diagnostics.npz", **arrays)
    with pytest.raises(ValueError, match="recomputed advantage differs"):
        recompute(output)


def test_critic_attribution_cli_routes(monkeypatch):
    from eco_planner.experiments.training.critic_attribution import runner
    from scripts import experiments as cli

    calls = []
    monkeypatch.setattr(runner, "run", lambda *a, **kw: calls.append((a, kw)))
    args = cli.build_parser().parse_args(
        [
            "training",
            "critic-attribution",
            "run",
            "--source-dir",
            "source",
            "--output-dir",
            "out",
            "--no-figures",
        ]
    )
    assert args.config.is_file()
    cli.dispatch(args)
    assert calls == [((args.source_dir, args.config, args.output_dir), {"figures": False})]
    analyze_args = cli.build_parser().parse_args(
        [
            "training",
            "critic-attribution",
            "analyze",
            "--source-dir",
            "source",
            "--output-dir",
            "out",
        ]
    )
    assert analyze_args.action == "analyze"
