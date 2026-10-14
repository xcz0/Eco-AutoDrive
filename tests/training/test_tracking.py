from __future__ import annotations

from pathlib import Path

import pytest
import torch
from lightning.fabric import Fabric
from mlflow import MlflowClient
from omegaconf import OmegaConf

from eco_planner.planning.policy import ExplorationPolicy
from eco_planner.rl.tracking import TrackingIdentity, TrainingTracking, update_metrics
from tests.training.helpers import _config, _policy_config


def _output(tmp_path, config, name):
    path = tmp_path / name
    path.mkdir()
    OmegaConf.save(OmegaConf.create(config.model_dump(mode="json")), path / "resolved_config.yaml")
    for artifact in ("policy-update-000.pt", "policy-update-001.pt", "training-state.ckpt"):
        (path / artifact).write_bytes(b"artifact transport fixture")
    return path


def test_adapter_preserves_units_denominators_and_optional_metrics(summary):
    summary = summary.model_copy(
        update={"collision_count": 1, "out_of_road_count": 1, "executed_fuel_proxy_ml_per_km": None}
    )
    metrics = update_metrics(summary)
    assert metrics["reward/total_mean"] == summary.total_reward / 2
    assert metrics["reward/total_sum"] == summary.total_reward
    assert metrics["behavior/collision_transition_fraction"] == 0.5
    assert metrics["behavior/out_of_road_transition_count"] == 1
    assert metrics["policy/action_mean/dim_1"] == summary.action_mean[1]
    assert "energy/executed_fuel_proxy_ml_per_km" not in metrics
    assert "ppo/kl_early_stop_trigger" not in metrics
    assert "gradient/actor_head_policy" not in metrics


def test_real_mlflow_runs_metrics_artifacts_and_same_run_resume(tmp_path, summary):
    config = _config(tmp_path)
    output = _output(tmp_path, config, "first")
    with torch.random.fork_rng():
        policy = ExplorationPolicy(_policy_config())
    policy_before = {key: value.clone() for key, value in policy.state_dict().items()}
    rng_before = torch.random.get_rng_state().clone()
    with TrainingTracking(config, output) as tracking:
        tracking.start_new()
        tracking.attach(Fabric(accelerator="cpu"), [], None)
        tracking.update(summary)
        identity = tracking.identity
        client = MlflowClient(identity.tracking_uri)
        assert client.get_run(identity.run_id).info.status == "RUNNING"
        assert client.get_metric_history(identity.run_id, "reward/total_mean")[0].step == 0
    assert torch.equal(rng_before, torch.random.get_rng_state())
    for key, value in policy.state_dict().items():
        assert torch.equal(value, policy_before[key])
    assert client.get_run(identity.run_id).info.status == "FINISHED"
    assert client.get_run(identity.run_id).data.params["config.runtime.seed"] == "0"
    artifacts = client.list_artifacts(identity.run_id, tracking.invocation)
    assert {Path(item.path).name for item in artifacts} >= {
        "resolved_config.yaml",
        "policy-update-000.pt",
        "training-state.ckpt",
    }
    resumed = config.model_copy(
        update={
            "training": config.training.model_copy(
                update={
                    "resume_checkpoint_path": str(output / "training-state.ckpt"),
                    "update_count": 5,
                }
            )
        }
    )
    with TrainingTracking(resumed, _output(tmp_path, resumed, "resumed")) as tracking:
        tracking.attach(Fabric(accelerator="cpu"), [summary], identity)
        tracking.update(summary.model_copy(update={"update_index": 1}))
        assert tracking.identity == identity
    assert [m.step for m in client.get_metric_history(identity.run_id, "reward/total_mean")] == [
        0,
        1,
    ]
    other = _config(tmp_path, seed=1)
    with TrainingTracking(other, _output(tmp_path, other, "seed1")) as tracking:
        tracking.start_new()
        tracking.attach(Fabric(accelerator="cpu"), [], None)
        assert tracking.identity.run_id != identity.run_id


def test_resume_repairs_missing_metrics_but_rejects_conflicts_and_stale_checkpoint(
    tmp_path, summary
):
    config = _config(tmp_path)
    first = _output(tmp_path, config, "first")
    with TrainingTracking(config, first) as tracking:
        tracking.start_new()
        fabric = Fabric(accelerator="cpu")
        tracking.attach(fabric, [], None)
        fabric.log_dict({"reward/total_mean": summary.total_reward / summary.sample_count}, step=0)
        identity = tracking.identity
    resumed = _config(tmp_path, resume=str(first / "training-state.ckpt"))
    with TrainingTracking(resumed, _output(tmp_path, resumed, "repair")) as tracking:
        tracking.attach(Fabric(accelerator="cpu"), [summary], identity)
    client = MlflowClient(identity.tracking_uri)
    assert len(client.get_metric_history(identity.run_id, "reward/total_mean")) == 1
    assert len(client.get_metric_history(identity.run_id, "ppo/mean_total_loss")) == 1
    for name, history, match in (
        ("stale", [], "ahead"),
        ("conflict", [summary.model_copy(update={"total_reward": -200.0})], "conflicts"),
    ):
        with pytest.raises(ValueError, match=match):
            with TrainingTracking(resumed, _output(tmp_path, resumed, name)) as tracking:
                tracking.attach(Fabric(accelerator="cpu"), history, identity)
        assert client.get_run(identity.run_id).info.status == "FINISHED"
    changed = _config(tmp_path, seed=1, resume=str(first / "training-state.ckpt"))
    with pytest.raises(ValueError, match="immutable"):
        with TrainingTracking(changed, _output(tmp_path, changed, "changed")) as tracking:
            tracking.attach(Fabric(accelerator="cpu"), [summary], identity)


@pytest.mark.parametrize("original_config", [True, False])
def test_legacy_checkpoint_history_provenance(tmp_path, summary, original_config):
    source = tmp_path / "old"
    source.mkdir()
    config = _config(tmp_path, resume=str(source / "training-state.ckpt"))
    if original_config:
        payload = config.model_dump(mode="json", exclude={"tracking"})
        OmegaConf.save(OmegaConf.create(payload), source / "resolved_config.yaml")
    with TrainingTracking(config, _output(tmp_path, config, "new")) as tracking:
        tracking.attach(Fabric(accelerator="cpu"), [summary], None)
        client = MlflowClient(tracking.identity.tracking_uri)
        run = client.get_run(tracking.identity.run_id)
        prefix = "config." if original_config else "continuation_config."
        assert run.data.params[f"{prefix}runtime.seed"] == "0"
        assert run.data.tags["history.parameter_provenance"] == (
            "original_resolved_config" if original_config else "historical_parameters_unrecorded"
        )
        assert len(client.get_metric_history(run.info.run_id, "reward/total_mean")) == 1


@pytest.mark.parametrize(
    "error,status", [(RuntimeError("failure"), "FAILED"), (KeyboardInterrupt(), "KILLED")]
)
def test_failure_status_and_original_exception(tmp_path, error, status):
    config = _config(tmp_path)
    with pytest.raises(type(error)) as caught:
        with TrainingTracking(config, _output(tmp_path, config, "failure")) as tracking:
            tracking.start_new()
            raise error
    assert caught.value is error
    assert (
        MlflowClient(tracking.identity.tracking_uri).get_run(tracking.identity.run_id).info.status
        == status
    )


def test_disabled_tracking_does_not_create_store(tmp_path, summary):
    config = _config(tmp_path, enabled=False)
    identity = TrackingIdentity(run_id="saved-run", tracking_uri="https://example.org")
    with TrainingTracking(config, _output(tmp_path, config, "disabled")) as tracking:
        tracking.start_new()
        tracking.attach(Fabric(accelerator="cpu"), [summary], identity)
        tracking.update(summary)
        assert tracking.identity == identity
    assert not (tmp_path / "mlflow.db").exists()
    assert not (tmp_path / "artifacts").exists()
