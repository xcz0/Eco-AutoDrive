"""Workflow evidence and matched evaluation contracts, with synthetic measurements."""

import json

import pytest
from omegaconf import OmegaConf

from eco_planner.analysis.runner import analyze
from eco_planner.artifacts import write_json
from eco_planner.evaluation.artifacts.models import PolicyCheckpointProvenance
from eco_planner.experiments.protocol.config import DEFAULT_PROTOCOL
from eco_planner.experiments.training import grid, runner
from tests.analysis.test_reports import job
from tests.evaluation.test_artifacts import _training_summary
from tests.training.test_effective_update import _metrics, _study
from tests.training.test_tracking import summary as update_fixture


@pytest.fixture
def training_source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    training = _training_summary(0, 0).model_copy(
        update={"updates": (update_fixture.__wrapped__(),)}
    )
    write_json(source / "summary.json", training.model_dump(mode="json"))
    OmegaConf.save(
        OmegaConf.create({"ppo": {"max_gradient_norm": 0.5}}), source / "resolved_config.yaml"
    )
    protocol = OmegaConf.load(DEFAULT_PROTOCOL)
    protocol.evaluation.maps = ["S"]
    protocol.evaluation.map_seeds = [0]
    protocol.evaluation.seed = 0
    protocol.evaluation.horizon_steps = 10
    protocol.training.maps = ["C"]
    protocol.training.map_seeds = [1]
    OmegaConf.save(protocol, source / "protocol.yaml")
    return source, training


def test_diagnostics_persist_measurements_missing_seeds_and_offline_report(
    training_source, tmp_path, monkeypatch
):
    source, _ = training_source
    calls = []
    monkeypatch.setattr(runner, "extract_arm_metrics", lambda *a, **kw: {"parameter_delta": 0.25})

    def measured(*args, **kwargs):
        calls.append(kwargs)
        return {"post_update_kl": [0.002]}

    monkeypatch.setattr(runner, "post_update_kl_series", measured)
    config = source / "diagnose.yaml"
    OmegaConf.save(
        OmegaConf.create(
            {
                "training_summaries": ["summary.json"],
                "training_seeds": [0, 1],
                "mc_draws": 128,
                "mc_seed": 7,
            }
        ),
        config,
    )
    output = tmp_path / "diagnostics"
    runner.diagnose(config, output, figures=False)
    assert calls == [{"update_count": 1, "mc_draws": 128, "mc_seed": 7}]
    snapshot = {p: p.read_bytes() for p in output.rglob("*") if p.is_file()}
    result = analyze("training", output, tmp_path / "report", figures=True)
    assert result["evidence"]["missing_seeds"] == [1]
    assert result["evidence"]["runs"][0]["metrics"]["parameter_delta"] == 0.25
    assert len(calls) == 1
    assert all(p.read_bytes() == value for p, value in snapshot.items())


@pytest.mark.parametrize("mismatch", [None, "hash", "mode", "noise"])
def test_policy_evaluation_reuses_only_matched_deterministic_evidence(
    training_source, tmp_path, monkeypatch, mismatch
):
    source, training = training_source
    baseline = source / "deterministic"
    baseline.mkdir()

    def evaluation(directory, seed):
        value = job().model_copy(
            update={
                "policy_checkpoint": PolicyCheckpointProvenance(
                    label="final", path="policy.pt", policy_hash=training.final_policy_hash
                )
            }
        )
        if seed is None and mismatch == "hash":
            value = value.model_copy(
                update={
                    "policy_checkpoint": value.policy_checkpoint.model_copy(
                        update={"policy_hash": "wrong"}
                    )
                }
            )
        if seed is None and mismatch == "noise":
            value = value.model_copy(
                update={"episodes": (value.episodes[0].model_copy(update={"noise_seed": 3}),)}
            )
        directory.mkdir(parents=True, exist_ok=True)
        write_json(directory / "summary.json", value.model_dump(mode="json"))
        OmegaConf.save(
            OmegaConf.create(
                {
                    "evaluation": {
                        "policy_checkpoint": {
                            "action_mode": "sample"
                            if seed is not None or mismatch == "mode"
                            else "mean",
                            "policy_action_seed": seed,
                        }
                    }
                }
            ),
            directory / "resolved_config.yaml",
        )
        return value

    evaluation(baseline, None)
    calls = []

    def compose(*args):
        seed = int(args[-1][-1].split("=")[-1])
        return seed, None

    def execute(seed, directory):
        calls.append(seed)
        return evaluation(directory, seed)

    monkeypatch.setattr(runner, "compose_policy_evaluation_config", compose)
    monkeypatch.setattr(runner, "run_evaluation_job", execute)
    config = source / "evaluation.yaml"
    OmegaConf.save(
        OmegaConf.create(
            {
                "protocol": "protocol.yaml",
                "records": [
                    {
                        "arm": "a2",
                        "training_summary": "summary.json",
                        "checkpoint_label": "final",
                        "checkpoint_path": "policy.pt",
                        "deterministic_evaluation_dir": "deterministic",
                    }
                ],
                "policy_action_seeds": [2, 5],
            }
        ),
        config,
    )
    original = {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}
    if mismatch:
        with pytest.raises(ValueError):
            runner.evaluate(config, tmp_path / "evaluations", figures=False)
    else:
        result = runner.evaluate(config, tmp_path / "evaluations", figures=False)
        assert calls == [2, 5]
        record = result["evidence"]["runs"][0]
        assert record["stochastic"]["2"]["comparison"]["available_pair_count"] == 1
        assert record["stochastic"]["5"]["comparison"]["statistics"]["energy_ml"]["mean"] == 0
    assert all(p.read_bytes() == value for p, value in original.items())


@pytest.mark.parametrize("passed", [False, True])
def test_grid_reports_no_candidate_or_selects_lexicographically(tmp_path, monkeypatch, passed):
    study = _study()
    config = tmp_path / "grid.yaml"
    OmegaConf.save(OmegaConf.create(study.model_dump()), config)
    monkeypatch.setattr(grid, "load_training_grid", lambda p: study)
    monkeypatch.setattr(grid, "load_protocol", lambda p: object())

    def train(study, protocol, directory, *args):
        directory.mkdir()
        write_json(directory / "summary.json", {"initial_policy_hash": "same"})
        return "completed"

    monkeypatch.setattr(grid, "_ensure_training_run", train)
    monkeypatch.setattr(
        grid,
        "extract_arm_metrics",
        lambda *a, **kw: {
            **_metrics(policy_ratio_change=0.001 if passed else 0),
            "initial_policy_hash": "same",
        },
    )
    monkeypatch.setattr(grid, "post_update_kl_series", lambda *a, **kw: {})
    heldout_calls = []

    def heldout(*args):
        heldout_calls.append(args[3])
        return {key: 1.0 if args[3] == "initial" else 2.0 for key in study.gate.heldout_metrics}

    monkeypatch.setattr(grid, "_heldout_values", heldout)
    result = grid.run(config, tmp_path / "grid", figures=False)
    assert result["update_gate_passed"] is passed
    if passed:
        assert result["selected_config"]["label"] == grid.arm_label(1e-5, 1, 0.5)
        assert heldout_calls == ["initial", *(["final"] * 8)]
    else:
        assert result["selected_config"] is None
        assert heldout_calls == []


def test_grid_preserves_failure_and_propagates_original_error(tmp_path, monkeypatch):
    config = tmp_path / "grid.yaml"
    OmegaConf.save(OmegaConf.create(_study().model_dump()), config)
    monkeypatch.setattr(grid, "load_training_grid", lambda p: _study())
    monkeypatch.setattr(grid, "load_protocol", lambda p: object())

    def fail(*args):
        raise RuntimeError("original training failure")

    monkeypatch.setattr(grid, "_ensure_training_run", fail)
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="original training failure"):
        grid.run(config, output, figures=False)
    data = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert data["status"] == "incomplete"
    assert data["selected_config"] is None
    assert data["arms"][0]["status"] == "failed"
    assert "original training failure" in data["arms"][0]["failure"]["traceback"]
    analyze("training", output, tmp_path / "report", figures=False)
