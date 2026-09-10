"""Artifact-to-report regression tests; synthetic fixtures are not research evidence."""

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import optuna
import pytest
import torch
from omegaconf import OmegaConf
from optuna.distributions import FloatDistribution
from optuna.trial import TrialState

from eco_planner.analysis.fixed_batch import recompute
from eco_planner.analysis.io import read_json, write_json
from eco_planner.analysis.runner import analyze
from eco_planner.analysis.statistics import (
    advantage_comparison,
    cosine,
    paired_difference,
    statistics,
)
from eco_planner.evaluation.artifacts.models import (
    CheckpointSummary,
    EvaluationWorkload,
    InferenceRuntimeSummary,
    JobSummary,
    PolicyCheckpointProvenance,
    WorkloadScenario,
)
from tests.characterization.test_experiment_outputs import _episode, _training_summary


def assert_report(output: Path, *, figures: bool) -> None:
    report = (output / "report.md").read_text(encoding="utf-8")
    assert (output / "analysis.json").is_file()
    links = re.findall(r"\]\(<?([^)>]+)>?\)", report)
    for link in links:
        assert (output / link).exists(), link
    files = read_json(output / "analysis.json")["figures"]
    assert bool(files) == figures
    for relative in files:
        path = output / relative
        assert path.stat().st_size > 100
        if relative.endswith(".svg"):
            assert "<svg" in path.read_text(encoding="utf-8")
        else:
            assert path.read_bytes().startswith(b"\x89PNG")


@pytest.fixture(scope="module")
def batches():
    from eco_planner.experiments.critic_gae_ablation.diagnostics import analyze_critic_gae_ablation
    from eco_planner.experiments.lambda_identifiability.diagnostics import analyze as diagnose
    from eco_planner.experiments.objective_decomposition.diagnostics import analyze_decomposition
    from eco_planner.rl.optimization import PPOUpdater
    from eco_planner.rl.policy import ExplorationPolicy
    from tests.training.test_critic_gae_ablation import _study as ablation_study
    from tests.training.test_objective_decomposition import _study
    from tests.training.test_ppo import _behavior_policy_episode, _policy_config, _ppo_config
    from tests.training.test_reward import _no_energy_config

    with torch.random.fork_rng():
        torch.manual_seed(0)
        policy = ExplorationPolicy(_policy_config())
        updater = PPOUpdater(policy, _ppo_config().model_copy(update={"batch_size": 4}))
        episodes = [
            _behavior_policy_episode(policy, torch.tensor([action]), reward=0.5)
            for action in [(-0.5, 0.2), (0.3, -0.7), (-0.1, -0.4), (0.6, 0.8)]
        ]
        for i, episode in enumerate(episodes):
            episode.audit["reward_component_progress"].fill_([0.2, 0.4, 0.6, 0.8][i])
            episode.audit["reward_component_energy"].fill_([0.9, 0.5, 0.7, 0.3][i])
            episode.audit["reward_safety_gate"].fill_([1.0, 0.5, 1.0, 0.25][i])
        base, ids, q = _no_energy_config(), np.array([0, 0, 1, 1]), [0.0, 0.5, 1.0]
        return {
            "lambda-identifiability": diagnose(updater, episodes, base, [0.0, 2.0, 8.0], q, ids),
            "objective-decomposition": analyze_decomposition(
                updater, episodes, base, [2.0, 8.0], q, ids, _study().gate
            ),
            "critic-gae-ablation": analyze_critic_gae_ablation(
                updater, episodes, base, q, ids, ablation_study().gate
            ),
        }


def write_batch(path, batch):
    path.mkdir(parents=True)
    summary, arrays = batch
    write_json(path / "summary.json", summary)
    np.savez(path / "diagnostics.npz", **arrays)
    write_json(
        path / "sample_index.json",
        {
            "samples": [
                {"scenario_index": i // 2, "episode_index": i % 2, "planning_cycle_index": 0}
                for i in range(4)
            ]
        },
    )


@pytest.mark.parametrize(
    "kind", ["lambda-identifiability", "objective-decomposition", "critic-gae-ablation"]
)
def test_fixed_batch_recomputes_original_statistics_and_renders(tmp_path, batches, kind):
    source, output = tmp_path / "source", tmp_path / "report"
    write_batch(source, batches[kind])
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    result = analyze(kind, source, output)
    original = batches[kind][0]
    assert result["evidence"]["arms"] == original["arms"]
    for key in ("gate", "attribution"):
        if key in original:
            assert result["evidence"][key] == original[key]
    assert_report(output, figures=True)
    assert before == {p.name: p.read_bytes() for p in source.iterdir()}
    for i, pair in enumerate(original["pairs"]):
        for key, value in pair.items():
            assert result["evidence"]["pairs"][i][key] == value
    # Summary statistics must not serve as a fallback for absent arrays.
    arrays = dict(batches[kind][1])
    del arrays[next(k for k in arrays if k.endswith("raw_advantage"))]
    np.savez(source / "diagnostics.npz", **arrays)
    with pytest.raises(KeyError):
        recompute(source)


def test_fixed_batch_rejects_wrong_pairing_and_no_figures(tmp_path, batches):
    source = tmp_path / "source"
    write_batch(source, batches["lambda-identifiability"])
    analyze("lambda-identifiability", source, tmp_path / "report", figures=False)
    assert_report(tmp_path / "report", figures=False)
    samples = read_json(source / "sample_index.json")
    samples["samples"][0] = samples["samples"][1]
    write_json(source / "sample_index.json", samples)
    with pytest.raises(ValueError, match="duplicate"):
        recompute(source)
    with pytest.raises(ValueError, match="separate"):
        analyze("lambda-identifiability", source, source)


def test_calibration_recomputes_audit_arrays(tmp_path, batches):
    source = tmp_path / "source"
    for label in ("original", "calibrated"):
        write_batch(source / label, batches["lambda-identifiability"])
    (source / "sample_index.json").write_bytes((source / "original/sample_index.json").read_bytes())
    write_json(source / "summary.json", {"decision": "continuous evidence only"})
    write_json(source / "audit.json", {"interpretation": "recorded audit"})
    np.savez(
        source / "audit.npz",
        scenario_index=np.array([0, 0, 1, 1]),
        planning_cycle_index=np.zeros(4),
        signed_metric=np.array([-2.0, 1.0, 0.0, 3.0]),
    )
    result = analyze("reward-calibration", source, tmp_path / "report")
    assert result["evidence"]["distributions"]["signed_metric"]["all"]["mean"] == 0.5
    assert_report(tmp_path / "report", figures=True)


def test_shared_statistics_preserve_rank_ddof_and_undefined_semantics():
    x, y = np.array([1.0, 2.0, 2.0, 4.0]), np.array([4.0, 2.0, 2.0, 1.0])
    assert statistics(x, [0.0, 0.5, 1.0], ddof=1)["std"] == np.std(x, ddof=1)
    assert statistics(x, [0.0, 0.5, 1.0])["std"] == np.std(x)
    assert advantage_comparison(x, y)["spearman"] == pytest.approx(-1.0)
    assert cosine(np.zeros(4), x) is None
    assert advantage_comparison(np.ones(4), x)["pearson"] is None
    summary, delta = paired_difference(x, y, np.array([0, 0, 1, 1]), [0.0, 0.5, 1.0])
    np.testing.assert_array_equal(delta, y - x)
    assert summary["per_scenario"]["0"]["mean"] == 1.5
    with pytest.raises(ValueError):
        statistics(np.array([1.0]), [0.0, 1.0], ddof=1)


def job(energy=2.0):
    episode = _episode(seed=0, distance_m=100.0, energy_ml=energy)
    return JobSummary(
        status="completed",
        runtime=InferenceRuntimeSummary(
            requested_accelerator="cpu",
            resolved_accelerator="cpu",
            requested_precision="32-true",
            resolved_precision="32-true",
            device="cpu",
            seed=0,
            world_size=1,
        ),
        checkpoint=CheckpointSummary(ema_tensor_count=1, parameter_count=1),
        sampler=episode.sampler,
        guidance=episode.guidance,
        workload=EvaluationWorkload(
            mode="traffic",
            profile="fixture",
            history_warmup_steps=0,
            evaluated_horizon_steps=10,
            scenarios=(WorkloadScenario(name="traffic", map="S", seed=0),),
            video_enabled=False,
        ),
        episodes=(episode,),
    )


def test_energy_matched_comparison_and_protocol_mismatch(tmp_path):
    from eco_planner.analysis.evaluation import paired

    source = tmp_path / "source"
    records = []
    for guidance, energy in (("baseline", 2.0), ("longitudinal_zero", 3.0)):
        path = source / "road" / guidance
        path.mkdir(parents=True)
        write_json(path / "summary.json", job(energy).model_dump(mode="json"))
        records.append({"job": "road", "guidance": guidance, "status": "completed"})
    write_json(source / "matrix_summary.json", {"runs": records})
    result = analyze("energy-sweep", source, tmp_path / "report")
    assert (
        result["evidence"]["comparisons"]["road/longitudinal_zero"]["statistics"]["energy_ml"][
            "mean"
        ]
        == 1.0
    )
    assert_report(tmp_path / "report", figures=True)
    altered = job().model_copy(
        update={"workload": job().workload.model_copy(update={"video_enabled": True})}
    )
    with pytest.raises(ValueError, match="workloads"):
        paired(job(), altered)
    altered = job().model_copy(
        update={"episodes": (job().episodes[0].model_copy(update={"noise_seed": 1}),)}
    )
    with pytest.raises(ValueError, match="pairs"):
        paired(job(), altered)


def test_failed_episode_is_visible_and_never_zero_filled(tmp_path):
    from eco_planner.analysis.evaluation import paired
    from eco_planner.analysis.reporting.experiments import experiment_figures
    from eco_planner.evaluation.artifacts.models import FailedEpisodeSummary

    original = job()
    episode = original.episodes[0].model_dump(mode="json")
    fields = {
        key: value for key, value in episode.items() if key in FailedEpisodeSummary.model_fields
    }
    fields.update(
        status="failed",
        trace_status="empty",
        termination={"type": "runtime_error", "detail": "fixture failure"},
        failure={
            "phase": "inference",
            "exception_type": "ValueError",
            "message": "fixture failure",
            "traceback": "fixture",
        },
    )
    failed = FailedEpisodeSummary.model_validate_json(json.dumps(fields))
    comparison = original.model_copy(update={"status": "failed", "episodes": (failed,)})
    result = paired(original, comparison)
    assert result["available_pair_count"] == 0
    assert result["unavailable_pair_count"] == 1
    assert result["statistics"]["energy_ml"] is None
    assert result["pairs"][0]["energy_ml_delta"] is None
    assert result["pairs"][0]["comparison"]["failure"]["message"] == "fixture failure"
    assert experiment_figures("energy-sweep", {"comparisons": {"failed": result}}, tmp_path)


@pytest.fixture
def training_summary():
    from tests.training.test_tracking import summary

    update = summary.__wrapped__()
    return _training_summary(0, 0).model_copy(update={"updates": (update,)})


def test_scalar_explicit_grouping_and_checkpoint_validation(tmp_path, training_summary):
    from eco_planner.experiments.scalar_reward.config import DEFAULT_PROTOCOL

    source = tmp_path / "source"
    source.mkdir()
    protocol = OmegaConf.load(DEFAULT_PROTOCOL)
    protocol.evaluation.maps = ["S"]
    protocol.evaluation.map_seeds = [0]
    protocol.evaluation.seed = 0
    protocol.evaluation.horizon_steps = 10
    protocol.training.maps = ["C"]
    protocol.training.map_seeds = [1]
    protocol.training.seeds = [0]
    OmegaConf.save(protocol, source / "protocol.yaml")
    for name in ("a0", "a2"):
        (source / name).mkdir()
    write_json(source / "a0/summary.json", job().model_dump(mode="json"))
    policy = job(1.0).model_copy(
        update={
            "policy_checkpoint": PolicyCheckpointProvenance(
                label="final", path="policy.pt", policy_hash=training_summary.final_policy_hash
            )
        }
    )
    write_json(source / "a2/summary.json", policy.model_dump(mode="json"))
    write_json(source / "training.json", training_summary.model_dump(mode="json"))
    config = {
        "protocol": "protocol.yaml",
        "a0_evaluation_dir": "a0",
        "runs": [
            {
                "arm": "a2",
                "training_summary": "training.json",
                "checkpoint_label": "final",
                "evaluation_dir": "a2",
            }
        ],
    }
    OmegaConf.save(OmegaConf.create(config), source / "comparison.yaml")
    result = analyze(
        "scalar-reward", source, tmp_path / "report", comparison_config=source / "comparison.yaml"
    )
    assert result["evidence"]["runs"][0]["comparison"]["statistics"]["energy_ml"]["mean"] == -1.0
    assert_report(tmp_path / "report", figures=True)
    config["runs"] *= 2
    OmegaConf.save(OmegaConf.create(config), source / "comparison.yaml")
    with pytest.raises(ValueError, match="duplicate"):
        analyze(
            "scalar-reward",
            source,
            tmp_path / "report",
            comparison_config=source / "comparison.yaml",
        )


def test_reproducibility_and_execution_reports(tmp_path, training_summary):
    source = tmp_path / "replay"
    path = source / "seed-0-replay-0"
    path.mkdir(parents=True)
    write_json(path / "summary.json", training_summary.model_dump(mode="json"))
    write_json(
        source / "training_report.json",
        {
            "status": "passed",
            "source_dir": str(source),
            "runs": [{"training_seed": 0, "replay_id": 0}],
        },
    )
    analyze("ppo-reproducibility", source, tmp_path / "replay-report")
    assert_report(tmp_path / "replay-report", figures=True)
    execution = tmp_path / "execution"
    execution.mkdir()
    write_json(
        execution / "evaluation_modes.json",
        {
            "evaluation_modes": {
                "serial": {
                    "outer_wall_s": {
                        "samples": [3.0],
                        "median": 3.0,
                        "minimum": 3.0,
                        "maximum": 3.0,
                    },
                    "jobs": [
                        {"metadata": {"elapsed_seconds": 1.0}},
                        {"metadata": {"elapsed_seconds": 2.0}},
                    ],
                }
            }
        },
    )
    result = analyze("execution-backend", execution, tmp_path / "execution-report")
    assert result["evidence"]["modes"]["serial"]["statistics"]["mean"] == 1.5
    assert_report(tmp_path / "execution-report", figures=True)


def test_sanity_keeps_failed_checks(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    write_json(
        source / "sanity_report.json",
        {
            "status": "failed",
            "cases": {"case": {"components": {"energy": 0.5, "comfort": 1.0}}},
            "checks": [{"name": "expected", "passed": False}],
        },
    )
    result = analyze("reward-sanity", source, tmp_path / "report")
    assert result["evidence"]["status"] == "failed"
    assert_report(tmp_path / "report", figures=True)


@pytest.mark.parametrize("count", [0, 4])
def test_optuna_read_only_native_plots_and_unavailable_trials(tmp_path, count, training_summary):
    source = tmp_path / "source"
    source.mkdir()
    study = optuna.create_study(
        study_name="fixture", storage=f"sqlite:///{(source / 'study.db').as_posix()}"
    )
    for i in range(count):
        study.add_trial(
            optuna.trial.create_trial(
                value=float(i),
                params={"x": float(i), "y": float(i % 2)},
                distributions={"x": FloatDistribution(0.0, 4.0), "y": FloatDistribution(0.0, 1.0)},
            )
        )
    study.add_trial(optuna.trial.create_trial(state=TrialState.FAIL))
    if count:
        training = source / "stage-b/config-0000/seed-0"
        training.mkdir(parents=True)
        write_json(training / "summary.json", training_summary.model_dump(mode="json"))
        write_json(
            source / "stage-b/summary.json",
            {
                "records": [
                    {
                        "config_id": 0,
                        "training_seed": 0,
                        "minimum_episode_length_retention": 0.9,
                        "state": "failed",
                        "reason": "fixture failure",
                        "evaluation": None,
                    }
                ],
                "ranked_config_ids": [],
            },
        )
    OmegaConf.save(
        OmegaConf.create({"study_name": "fixture", "sampler_seed": 3}),
        source / "study_manifest.yaml",
    )
    before = (source / "study.db").read_bytes()
    result = analyze("ppo-stability", source, tmp_path / "report")
    assert (source / "study.db").read_bytes() == before
    assert result["evidence"]["trials"][-1]["state"] == "FAIL"
    if count:
        assert_report(tmp_path / "report", figures=True)
        assert result["evidence"]["training_curves"]
        assert result["evidence"]["stages"]["b"]["ranked_config_ids"] == []
        again = analyze("ppo-stability", source, tmp_path / "again", figures=False)
        assert (
            again["evidence"]["parameter_importances"]
            == result["evidence"]["parameter_importances"]
        )
    else:
        assert "parameter-importance" in result["evidence"]["unavailable_figures"]
    with pytest.raises(FileNotFoundError):
        from eco_planner.analysis.stability import load_study

        load_study(tmp_path / "missing", "fixture")
    assert not (tmp_path / "missing").exists()


@pytest.mark.parametrize("training", [False, True])
def test_scalar_run_uses_common_report_writer(tmp_path, training_summary, training):
    from eco_planner.analysis.runner import publish_scalar_run

    summary = training_summary if training else job()
    write_json(tmp_path / "summary.json", summary.model_dump(mode="json"))
    publish_scalar_run(tmp_path, training=training)
    assert_report(tmp_path, figures=True)


def test_offline_imports_do_not_load_execution_modules():
    script = """
import sys
import scripts.experiments.__main__
import eco_planner.analysis.evaluation
import eco_planner.analysis.simple
import eco_planner.analysis.stability
from eco_planner.rl.artifacts import TrainingRunSummary
for root in ('torch', 'metadrive', 'panda3d', 'eco_planner.rl.trainer', 'eco_planner.models'):
    assert not any(k == root or k.startswith(root + '.') for k in sys.modules), root
"""
    subprocess.run([sys.executable, "-c", script], check=True)


@pytest.mark.parametrize(
    "module,function,args",
    [
        ("lambda_identifiability", "run", ["--source-dir", "in", "--output-dir", "out"]),
        ("objective_decomposition", "run", ["--source-dir", "in", "--output-dir", "out"]),
        (
            "critic_gae_ablation",
            "run",
            ["--source-dir", "in", "--reference-dir", "ref", "--output-dir", "out"],
        ),
        (
            "reward_calibration",
            "run",
            ["--source-dir", "in", "--reference-dir", "ref", "--output-dir", "out"],
        ),
        ("energy_sweep", "run_study", ["--output-dir", "out"]),
        ("reward_sanity", "run_sanity", ["--output-dir", "out"]),
        ("scalar_reward", "run_command", ["evaluate-a0", "--output-dir", "out"]),
        ("ppo_stability", "run_command", ["summarize", "--output-dir", "out"]),
        (
            "ppo_reproducibility",
            "summarize_and_write_training_runs",
            ["--source-dir", "in", "--output-dir", "out"],
        ),
        (
            "execution_backend",
            "write_report",
            [
                "--serial-dir",
                "serial",
                "--job-level-dir",
                "job",
                "--vector-dir",
                "vector",
                "--output-dir",
                "out",
                "--serial-wall-s",
                "1",
                "--job-level-wall-s",
                "1",
                "--vector-wall-s",
                "1",
            ],
        ),
    ],
)
def test_unified_cli_forwards_no_figures(monkeypatch, module, function, args):
    from importlib import import_module

    adapter = import_module("scripts.experiments.__main__")
    suffix = (
        ""
        if module == "ppo_reproducibility"
        else (".report" if module == "execution_backend" else ".runner")
    )
    implementation = import_module("eco_planner.experiments." + module + suffix)
    captured = {}

    def fake(*args, **kwargs):
        captured.update(kwargs)
        return 0 if module in ("energy_sweep", "reward_sanity") else {}

    monkeypatch.setattr(implementation, function, fake)
    monkeypatch.setattr(adapter, "bootstrap", lambda *_: None)
    command = (
        []
        if module in ("scalar_reward", "ppo_stability")
        else ["report" if module in ("ppo_reproducibility", "execution_backend") else "run"]
    )
    monkeypatch.setattr(
        sys, "argv", ["experiment", module.replace("_", "-"), *command, *args, "--no-figures"]
    )
    with pytest.raises(SystemExit) as exc:
        adapter.main()
    assert exc.value.code == 0
    assert captured["figures"] is False
