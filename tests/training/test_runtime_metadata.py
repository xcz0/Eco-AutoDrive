"""Strict runtime provenance with synthetic execution reports, without a simulation."""

import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from eco_planner.evaluation import engine
from eco_planner.evaluation.artifacts.io import load_runtime_metadata
from eco_planner.jobs import compose_job_config
from eco_planner.planning.diffusion import CheckpointLoadReport, sampler_report
from eco_planner.rl.artifacts import io
from eco_planner.rl.artifacts.metadata import TrainingRuntimeMetadata
from eco_planner.rl.config import parse_training_config
from eco_planner.runtime.fabric import InferenceRuntimeReport


def test_training_metadata_uses_actual_reports_and_strict_roundtrip(tmp_path, monkeypatch):
    config = parse_training_config(
        compose_job_config(
            "jobs/training/ppo",
            ["components/resources=rtx3050_laptop", "runtime.seed=0", "training.replay_id=0"],
        )
    )
    runtime = SimpleNamespace(
        report=InferenceRuntimeReport("auto", "cpu", "auto", "32-true", "cpu", 37, 1),
        checkpoint_report=CheckpointLoadReport(4, 100),
        sampler_report=sampler_report(config.sampler),
        guidance_config=config.guidance,
    )
    repository = {
        key: "synthetic"
        for key in (
            "git_head",
            "git_branch",
            "platform",
            "python",
            "torch",
            "lightning",
            "metadrive",
            "pydantic",
        )
    }
    repository["git_status_short"] = (" M synthetic.py",)
    monkeypatch.setattr(io, "collect_repository_metadata", lambda *_: repository)
    path = tmp_path / "runtime_metadata.json"
    io.write_training_runtime_metadata(path, runtime, config.resources)
    metadata = TrainingRuntimeMetadata.model_validate_json(path.read_text(encoding="utf-8"))
    assert metadata.inference_runtime.seed == 37
    assert metadata.inference_runtime.requested_accelerator == "auto"
    assert metadata.inference_runtime.resolved_accelerator == "cpu"
    assert metadata.sampler.model_dump() == vars(runtime.sampler_report)
    payload = metadata.model_dump(mode="json")
    payload["unowned"] = 1
    with pytest.raises(ValidationError, match="extra_forbidden"):
        TrainingRuntimeMetadata.model_validate_json(json.dumps(payload))
    payload.pop("unowned")
    payload["guidance"]["trajectory_dt_s"] = float("nan")
    with pytest.raises(ValidationError):
        TrainingRuntimeMetadata.model_validate_json(json.dumps(payload))

    evaluation_dir = tmp_path / "evaluation"
    evaluation_dir.mkdir()
    monkeypatch.setattr(engine, "collect_repository_metadata", lambda *_: repository)
    execution = engine.ExecutionReport("serial", "basic", 1, None, None, True, "cpu", 1, 4, None)
    engine.write_runtime_metadata(
        evaluation_dir, runtime.report, runtime.sampler_report, config.guidance, execution, 1.0
    )
    evaluation = load_runtime_metadata(evaluation_dir / "runtime_metadata.json")
    assert evaluation.inference_runtime.model_dump() == metadata.inference_runtime.model_dump()
    assert evaluation.cadence.model_dump() == config.cadence.model_dump()
    assert evaluation.cuda_memory is None
