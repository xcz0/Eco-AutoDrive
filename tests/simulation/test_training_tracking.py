"""Checkpoint-backed MetaDrive training through the shared tracking boundary."""

import json

import pytest
from mlflow import MlflowClient

from eco_planner.jobs import compose_job_config, run_training_job


@pytest.mark.simulator
@pytest.mark.gpu
def test_real_training_observer_and_mlflow_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    config = compose_job_config(
        "jobs/training/ppo_energy_smoke",
        [
            "components/resources=rtx3050_laptop",
            "runtime.seed=0",
            "training.replay_id=0",
            "training.transitions_per_environment=1",
            "ppo.batch_size=2",
            "ppo.minibatch_size=2",
        ],
    )
    config.tracking.tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    config.tracking.artifact_location = (tmp_path / "artifacts").as_posix()
    observed = []
    summary = run_training_job(config, tmp_path / "training", update_observer=observed.append)
    assert observed == list(summary.updates)
    assert summary.frozen_planner_hash_before == summary.frozen_planner_hash_after
    metadata = json.loads((tmp_path / "training/runtime_metadata.json").read_text(encoding="utf-8"))
    identity = metadata["tracking"]
    client = MlflowClient(identity["tracking_uri"])
    run = client.get_run(identity["run_id"])
    assert run.info.status == "FINISHED"
    assert run.data.metrics["behavior/sample_count"] == 2
    assert run.data.tags["git.commit"] == metadata["git_head"]
    invocation = client.list_artifacts(run.info.run_id, "invocations")[0].path
    assert {
        item.path.rsplit("/", 1)[-1] for item in client.list_artifacts(run.info.run_id, invocation)
    } >= {
        "summary.json",
        "runtime_metadata.json",
        "tracked_diff.patch",
        "policy-initial.pt",
        "policy-final.pt",
        "policy-update-000.pt",
        "training-state.ckpt",
        "resolved_config.yaml",
    }
