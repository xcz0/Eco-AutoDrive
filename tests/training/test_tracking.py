from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from lightning.fabric import Fabric
from mlflow import MlflowClient
from omegaconf import OmegaConf
from pydantic import ValidationError
from tensordict import TensorDict

from eco_planner.jobs import compose_job_config
from eco_planner.rl.artifacts import PolicyProbeSummary, build_update_summary
from eco_planner.rl.artifacts.metrics import RolloutMetrics
from eco_planner.rl.config import parse_training_config
from eco_planner.rl.optimization import (
    PPOUpdater,
    load_training_checkpoint,
    save_training_checkpoint,
)
from eco_planner.rl.optimization.metrics import PPOMetrics
from eco_planner.rl.optimization.ppo import PPOUpdateReport
from eco_planner.rl.policy import ExplorationPolicy
from eco_planner.rl.tracking import TrackingIdentity, TrainingTracking, update_metrics
from eco_planner.rl.training_state import TrainingLoopState, resume_training_state
from tests.training.test_ppo import _context, _episode, _policy_config, _ppo_config


def _config(tmp_path, *, seed=0, enabled=True, resume=None):
    raw = compose_job_config(
        "jobs/training/ppo",
        [
            "components/resources=rtx3050_laptop",
            f"runtime.seed={seed}",
            "training.replay_id=0",
        ],
    )
    raw.tracking.tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    raw.tracking.artifact_location = (tmp_path / "artifacts").as_posix()
    raw.tracking.enabled = enabled
    raw.training.resume_checkpoint_path = resume
    return parse_training_config(raw)


def _output(tmp_path, config, name):
    path = tmp_path / name
    path.mkdir()
    OmegaConf.save(OmegaConf.create(config.model_dump(mode="json")), path / "resolved_config.yaml")
    for artifact in ("policy-update-000.pt", "policy-update-001.pt", "training-state.ckpt"):
        (path / artifact).write_bytes(b"artifact transport fixture")
    return path


@pytest.fixture(scope="module")
def summary():
    with torch.random.fork_rng():
        torch.manual_seed(0)
        policy = ExplorationPolicy(_policy_config())
        episodes = tuple(
            _episode(reward=r, terminated=True, truncated=False, bootstrap=0.0) for r in (0.25, 2.0)
        )
        return build_update_summary(0, episodes, PPOUpdater(policy, _ppo_config()).update(episodes))


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


def test_rollout_aggregations_match_original_tensor_reductions():
    trajectory = TensorDict(
        {"value": torch.tensor([0.25, 1.0, 8.0]), "stopped": torch.tensor([True, False, False])},
        batch_size=[3],
    )
    metrics = RolloutMetrics(trajectory)
    assert metrics.sum("value") == float(trajectory["value"].sum())
    assert metrics.mean("value") == float(trajectory["value"].mean())
    assert metrics.maximum("value") == 8.0
    assert metrics.stopped_fraction() == float(trajectory["stopped"].float().mean())
    trajectory["value"][0] = float("nan")
    with pytest.raises(RuntimeError, match="nan"):
        metrics.mean("value")


def test_unequal_episode_summary_uses_transition_weights_and_ratio_of_totals(summary):
    short = _episode(reward=0.25, terminated=True, truncated=False, bootstrap=0.0)
    training = torch.cat([short.training, short.training], dim=0)
    training["next", "terminated"][0] = False
    training["next", "done"][0] = False
    audit = torch.cat([short.audit, short.audit], dim=0)
    audit["speed_mps"][:] = 10.0
    audit["step_distance_m"][:] = 1.0
    audit["executed_fuel_proxy_step_energy_ml"][:] = 1.0
    long = replace(short, training=training, audit=audit)
    short.audit["speed_mps"][:] = 1.0
    short.audit["step_distance_m"][:] = 10.0
    short.audit["executed_fuel_proxy_step_energy_ml"][:] = 2.0
    report = PPOUpdateReport(
        **{field.name: getattr(summary, field.name) for field in fields(PPOUpdateReport)}
    )
    report = replace(report, sample_count=3)
    result = build_update_summary(0, (short, long), report)
    assert result.mean_episode_length == 1.5
    assert result.mean_speed_mps == 7.0
    assert result.executed_fuel_proxy_ml_per_km == pytest.approx(4000.0 / 12.0)
    short.audit["step_distance_m"][:] = 0.0
    long.audit["step_distance_m"][:] = 0.0
    assert build_update_summary(0, (short, long), report).executed_fuel_proxy_ml_per_km is None


def test_ppo_metrics_include_evaluated_minibatches_and_only_executed_gradients():
    metrics = PPOMetrics(2, torch.device("cpu"))
    values = [torch.tensor([1.0, 3.0]), torch.tensor([5.0, 7.0])]
    for value in values:
        metrics.update(tuple(value.unbind()))
    metrics.gradient(torch.tensor(2.5))
    expected = torch.stack(values).double().mean(dim=0)
    torch.testing.assert_close(
        metrics.compute(), torch.cat((expected, torch.tensor([2.5]).double())), rtol=0, atol=0
    )
    empty = PPOMetrics(1, torch.device("cpu"))
    empty.update((torch.tensor(0.1),))
    assert empty.compute()[-1] == 0


def test_config_defaults_disable_remote_and_invalid_values(tmp_path):
    config = _config(tmp_path)
    assert config.tracking.enabled
    assert config.tracking.experiment_name == "eco-autodrive-ppo"
    payload = config.tracking.model_dump(mode="python")
    payload["tracking_uri"] = "https://mlflow.example.org"
    payload["artifact_location"] = None
    assert config.tracking.model_validate(payload).tracking_uri == "https://mlflow.example.org"
    for changes in ({"checkpoint_interval": 0}, {"enabled": "yes"}, {"tracking_uri": "bad"}):
        raw = config.tracking.model_dump(mode="python")
        raw.update(changes)
        with pytest.raises(ValidationError):
            config.tracking.model_validate(raw)


def test_real_mlflow_runs_metrics_artifacts_and_same_run_resume(tmp_path, summary):
    config = _config(tmp_path)
    output = _output(tmp_path, config, "first")
    with TrainingTracking(config, output) as tracking:
        tracking.start_new()
        tracking.attach(Fabric(accelerator="cpu"), [], None)
        tracking.update(summary)
        identity = tracking.identity
        client = MlflowClient(identity.tracking_uri)
        assert client.get_run(identity.run_id).info.status == "RUNNING"
        assert client.get_metric_history(identity.run_id, "reward/total_mean")[0].step == 0
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
    with TrainingTracking(config, _output(tmp_path, config, "disabled")) as tracking:
        tracking.start_new()
        tracking.attach(Fabric(accelerator="cpu"), [], None)
        tracking.update(summary)
    assert not (tmp_path / "mlflow.db").exists()
    assert not (tmp_path / "artifacts").exists()


def test_logging_and_checkpoint_resume_preserve_optimizer_and_rng(tmp_path):
    fabric = Fabric(accelerator="cpu")
    with torch.random.fork_rng():
        torch.manual_seed(12)
        policy = ExplorationPolicy(_policy_config())
        ppo = _ppo_config().model_copy(update={"scheduler_total_optimizer_steps": 2})
        updater = PPOUpdater(policy, ppo)
        episodes = tuple(
            _episode(reward=r, terminated=True, truncated=False, bootstrap=0.0) for r in (0.25, 2.0)
        )
        first = updater.update(episodes)
        config = _config(tmp_path)
        output = _output(tmp_path, config, "logged")
        with TrainingTracking(config, output) as tracking:
            tracking.start_new()
            tracking.attach(fabric, [], None)
            rng_before = torch.random.get_rng_state().clone()
            policy_before = {k: v.clone() for k, v in policy.state_dict().items()}
            tracking.update(build_update_summary(0, episodes, first))
            assert torch.equal(rng_before, torch.random.get_rng_state())
            for key, value in policy.state_dict().items():
                assert torch.equal(value, policy_before[key])
            save_training_checkpoint(
                output / "training-state.ckpt",
                fabric,
                policy,
                updater,
                {"completed_updates": 1, "tracking": tracking.identity.model_dump()},
            )
            expected = updater.update(episodes)
            final_policy = {k: v.clone() for k, v in policy.state_dict().items()}
            final_rng = torch.random.get_rng_state().clone()
            restored_policy = ExplorationPolicy(_policy_config())
            restored = PPOUpdater(restored_policy, ppo)
            _, loop = load_training_checkpoint(
                output / "training-state.ckpt", fabric, restored_policy, restored
            )
            assert loop["tracking"] == tracking.identity.model_dump()
            assert restored.update(episodes) == expected
            assert restored.scheduler.state_dict() == updater.scheduler.state_dict()
            assert torch.equal(final_rng, torch.random.get_rng_state())
            for key, value in restored_policy.state_dict().items():
                assert torch.equal(value, final_policy[key])


@pytest.mark.parametrize("tracked", [False, True])
def test_training_loop_restores_json_mode_summary_and_probe_from_checkpoint(
    tmp_path, summary, tracked
):
    fabric = Fabric(accelerator="cpu")
    policy = ExplorationPolicy(_policy_config())
    updater = PPOUpdater(policy, _ppo_config())
    probe = PolicyProbeSummary(
        alpha=((2.0, 2.0),),
        beta=((2.0, 2.0),),
        guidance_mean=((0.0, 0.0),),
        boundary_mass=((0.1, 0.1),),
    )
    identity = (
        TrackingIdentity(run_id="saved-run", tracking_uri="https://example.org")
        if tracked
        else None
    )
    loop = TrainingLoopState(
        completed_updates=1,
        total_transitions=summary.sample_count,
        update_summaries=[summary],
        probe_before=probe,
        probe_contexts=(_context(),),
        initial_policy_hash="a" * 64,
        tracking=identity,
    ).checkpoint_payload()
    assert isinstance(loop["update_summaries"][0]["action_mean"], list)
    checkpoint = tmp_path / "training-state.ckpt"
    save_training_checkpoint(checkpoint, fabric, policy, updater, loop)
    config = _config(tmp_path, enabled=False, resume=str(checkpoint))
    restored = resume_training_state(config, SimpleNamespace(fabric=fabric, policy=policy), updater)
    assert restored.completed_updates == 1
    assert restored.total_transitions == summary.sample_count
    assert restored.initial_policy_hash == "a" * 64
    assert restored.update_summaries == [summary]
    assert restored.probe_before == probe
    assert restored.tracking == identity
    assert torch.equal(
        restored.probe_contexts[0].reference_trajectory, _context().reference_trajectory
    )
    with TrainingTracking(config, tmp_path) as tracking:
        tracking.attach(fabric, restored.update_summaries, restored.tracking)
        assert tracking.identity == identity
