from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from hydra.errors import MissingConfigException
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
    assert config.resources.name == "rtx3050_laptop"
    assert config.resources.rollout_worker_count == 4
    assert config.resources.evaluation_job_worker_count == 2
    assert config.resources.evaluation_vector_env_slots == 4
    assert config.resources.torch_threads_per_worker == 8


def test_compose_job_config_preserves_an_explicit_resource_override(monkeypatch) -> None:
    monkeypatch.setenv("MACHINE_NAME", "rtx3050_laptop")

    config = workflows.compose_job_config(
        "jobs/training/ppo/smoke",
        (
            "components/resources=rtx_a4000",
            "runtime.seed=17",
            "training.replay_id=3",
        ),
    )

    assert config.resources.name == "rtx_a4000"


def test_compose_job_config_without_a_machine_profile(monkeypatch) -> None:
    monkeypatch.delenv("MACHINE_NAME", raising=False)

    config = workflows.compose_job_config(
        "jobs/training/ppo/smoke",
        ("runtime.seed=17", "training.replay_id=3"),
    )

    assert "resources" not in config


def test_unknown_machine_profile_is_reported_by_hydra(monkeypatch) -> None:
    monkeypatch.setenv("MACHINE_NAME", "unknown-machine")

    with pytest.raises(MissingConfigException, match="components/resources/unknown-machine"):
        workflows.compose_job_config(
            "jobs/training/ppo/smoke",
            ("runtime.seed=17", "training.replay_id=3"),
        )


def test_typed_job_runners_parse_before_invoking_domain_execution(
    monkeypatch, tmp_path: Path
) -> None:
    evaluation_config = OmegaConf.create({"evaluation": "raw"})
    training_config = OmegaConf.create({"training": "raw"})
    resource_profile = object()
    evaluation_summary = SimpleNamespace(resources=resource_profile)
    training_summary = SimpleNamespace(resources=resource_profile)
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


@pytest.mark.parametrize("runner_name", ["run_evaluation_job", "run_training_job"])
def test_job_execution_requires_a_resource_profile(
    monkeypatch, tmp_path: Path, runner_name: str
) -> None:
    config = OmegaConf.create({"job": "raw"})
    parsed = SimpleNamespace(resources=None)
    parse_name = (
        "parse_evaluation_config"
        if runner_name == "run_evaluation_job"
        else "parse_training_config"
    )
    monkeypatch.setattr(workflows, parse_name, lambda raw: parsed)

    with pytest.raises(ValueError, match="execution requires a resource profile"):
        getattr(workflows, runner_name)(config, tmp_path / runner_name)
