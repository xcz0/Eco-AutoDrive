from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from eco_planner import workflows


def test_compose_job_config_uses_the_shared_hydra_boundary(monkeypatch) -> None:
    monkeypatch.setenv("MACHINE_NAME", "rtx3050_laptop")

    config = workflows.compose_job_config(
        "jobs/training/ppo/smoke",
        ("runtime.seed=17", "training.replay_id=3"),
    )

    assert config.runtime.seed == 17
    assert config.training.replay_id == 3


def test_typed_job_runners_parse_before_invoking_domain_execution(
    monkeypatch, tmp_path: Path
) -> None:
    evaluation_config = OmegaConf.create({"evaluation": "raw"})
    training_config = OmegaConf.create({"training": "raw"})
    evaluation_summary = object()
    training_summary = object()
    seen: dict[str, object] = {}

    monkeypatch.setattr(workflows, "parse_evaluation_config", lambda config: evaluation_summary)
    monkeypatch.setattr(
        workflows,
        "run_evaluation",
        lambda config, output_dir: (
            seen.update(evaluation=(config, output_dir)) or evaluation_summary
        ),
    )
    monkeypatch.setattr(workflows, "parse_training_config", lambda config: training_summary)
    monkeypatch.setattr(
        workflows,
        "train",
        lambda config, output_dir, update_observer=None: (
            seen.update(training=(config, output_dir, update_observer)) or training_summary
        ),
    )

    assert (
        workflows.run_evaluation_job(evaluation_config, tmp_path / "evaluation")
        is evaluation_summary
    )
    assert workflows.run_training_job(training_config, tmp_path / "training") is training_summary
    assert seen["evaluation"] == (evaluation_summary, tmp_path / "evaluation")
    assert seen["training"] == (training_summary, tmp_path / "training", None)
    assert (tmp_path / "training" / "resolved_config.yaml").is_file()
