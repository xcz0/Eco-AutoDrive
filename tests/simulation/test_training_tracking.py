"""Checkpoint-backed MetaDrive training through the shared tracking boundary."""

import json

import numpy as np
import pytest
from mlflow import MlflowClient

from eco_planner.jobs import compose_job_config, run_training_job


@pytest.mark.simulator
@pytest.mark.gpu
def test_update_boundary_resume_matches_continuous_training(tmp_path, monkeypatch):
    import torch

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    overrides = [
        "components/resources=rtx3050_laptop",
        "runtime.seed=0",
        "training.replay_id=0",
        "training.transitions_per_environment=2",
        "ppo.batch_size=4",
        "ppo.minibatch_size=2",
        "ppo.epochs=2",
        "training.update_count=2",
        "tracking.enabled=false",
    ]
    config = compose_job_config("jobs/training/ppo_energy_smoke", overrides)
    continuous = run_training_job(config, tmp_path / "continuous")
    first_config = compose_job_config(
        "jobs/training/ppo_energy_smoke", [*overrides, "training.update_count=1"]
    )
    run_training_job(first_config, tmp_path / "first")
    config.training.resume_checkpoint_path = (tmp_path / "first/training-state.ckpt").as_posix()
    resumed = run_training_job(config, tmp_path / "resumed")
    assert resumed == continuous
    for expected in (tmp_path / "continuous/updates/update-001").glob("*.npz"):
        with (
            np.load(expected) as left,
            np.load(tmp_path / "resumed/updates/update-001" / expected.name) as right,
        ):
            assert set(left.files) == set(right.files)
            for key in left.files:
                np.testing.assert_array_equal(left[key], right[key], err_msg=key)
    left = torch.load(
        tmp_path / "continuous/training-state.ckpt", weights_only=False, map_location="cpu"
    )
    right = torch.load(
        tmp_path / "resumed/training-state.ckpt", weights_only=False, map_location="cpu"
    )

    def assert_equal(a, b):
        if isinstance(a, torch.Tensor):
            assert torch.equal(a, b)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for key in a:
                assert_equal(a[key], b[key])
        elif isinstance(a, (tuple, list)):
            assert len(a) == len(b)
            for x, y in zip(a, b, strict=True):
                assert_equal(x, y)
        else:
            assert a == b

    assert_equal(left, right)


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
    assert not (tmp_path / "training/tracked_diff.patch").exists()
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
        "policy-initial.pt",
        "policy-final.pt",
        "policy-update-000.pt",
        "training-state.ckpt",
        "resolved_config.yaml",
    }


@pytest.mark.simulator
@pytest.mark.gpu
def test_real_training_persists_the_task_g_reward_profile(tmp_path, monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    config = compose_job_config(
        "jobs/training/ppo_energy_smoke",
        [
            "components/resources=rtx3050_laptop",
            "components/reward=plannerrft_energy_band_lam64_v1",
            "runtime.seed=0",
            "training.replay_id=0",
            "training.transitions_per_environment=1",
            "ppo.batch_size=2",
            "ppo.minibatch_size=2",
        ],
    )
    config.tracking.tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    config.tracking.artifact_location = (tmp_path / "artifacts").as_posix()

    summary = run_training_job(config, tmp_path / "training")

    profile = "plannerrft_energy_band_lam64_v1"
    assert summary.reward_profile == profile
    assert all(update.reward_profile == profile for update in summary.updates)
    episodes = sorted((tmp_path / "training" / "updates").glob("update-000/*.npz"))
    assert episodes
    with np.load(episodes[0], allow_pickle=False) as data:
        assert str(data["reward_profile"]) == profile
