from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from eco_planner.rl import trainer
from eco_planner.rl.artifacts import PolicyProbeSummary
from eco_planner.rl.probing import capture_probe_contexts
from eco_planner.rl.tracking import TrackingIdentity
from eco_planner.rl.training_state import TrainingLoopState, resume_training_state
from tests.training.test_ppo import _context, _episode, _policy_config
from tests.training.test_tracking import _config
from tests.training.test_tracking import summary as summary


def test_new_training_state_has_independent_history(tmp_path):
    config = _config(tmp_path, enabled=False)
    runtime = Mock()
    updater = Mock()
    first = resume_training_state(config, runtime, updater)
    second = resume_training_state(config, runtime, updater)
    assert first == TrainingLoopState()
    assert first.update_summaries is not second.update_summaries
    runtime.assert_not_called()
    updater.assert_not_called()


@pytest.mark.parametrize("start_update", [0, 1, 2])
def test_training_order_fixed_probes_and_completed_resume(
    tmp_path, monkeypatch, summary, start_update
):
    config = _config(tmp_path, enabled=False)
    config = config.model_copy(
        update={
            "training": config.training.model_copy(update={"update_count": 2}),
            "ppo": config.ppo.model_copy(update={"batch_size": 2}),
        }
    )
    probe = PolicyProbeSummary(
        alpha=((2.0, 2.0),) * 2,
        beta=((2.0, 2.0),) * 2,
        guidance_mean=((0.0, 0.0),) * 2,
        boundary_mass=((0.1, 0.1),) * 2,
    )
    contexts = (_context(), _context()) if start_update else None
    state = TrainingLoopState(
        completed_updates=start_update,
        total_transitions=2 * start_update,
        update_summaries=[
            summary.model_copy(update={"update_index": i}) for i in range(start_update)
        ],
        probe_before=probe if start_update else None,
        probe_contexts=contexts,
        initial_policy_hash="a" * 64 if start_update else None,
    )
    events = []
    runtime = SimpleNamespace(
        policy=trainer_policy(),
        fabric=object(),
        new_noise_generator=lambda seed: torch.Generator().manual_seed(seed),
        new_policy_generator=lambda seed: torch.Generator().manual_seed(seed),
        frozen_planner_hash=lambda: "f" * 64,
    )
    monkeypatch.setattr(trainer, "create_fabric_rollout_runtime", lambda *a, **k: runtime)
    monkeypatch.setattr(trainer, "resume_training_state", lambda *a: state)
    monkeypatch.setattr(trainer, "write_training_runtime_metadata", lambda *a: None)
    monkeypatch.setattr(trainer, "write_rollout_episode", lambda *a: None)
    monkeypatch.setattr(
        trainer,
        "build_update_summary",
        lambda i, *a: summary.model_copy(update={"update_index": i}),
    )
    updater = Mock()
    updater.update.side_effect = lambda *a: events.append("ppo")
    monkeypatch.setattr(trainer, "PPOUpdater", lambda *a: updater)

    class Collector:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            events.append("close")

        def collect(self, **kwargs):
            events.append("collect")
            return tuple(
                (_episode(reward=1.0, terminated=True, truncated=False, bootstrap=0.0),)
                for _ in range(2)
            )

    monkeypatch.setattr(trainer, "VectorRolloutCollector", Collector)
    capture = Mock(wraps=capture_probe_contexts)
    monkeypatch.setattr(trainer, "capture_probe_contexts", capture)
    probe_call = Mock(return_value=probe)
    monkeypatch.setattr(trainer, "probe_policy", probe_call)
    exports = []
    monkeypatch.setattr(
        trainer, "save_exploration_policy_checkpoint", lambda path, *a: exports.append(path.name)
    )
    payloads = []

    def save_checkpoint(path, fabric, policy, updater, payload):
        events.append("checkpoint")
        payloads.append(payload)

    monkeypatch.setattr(trainer, "save_training_checkpoint", save_checkpoint)
    identity = TrackingIdentity(run_id="test", tracking_uri="https://example.org")
    tracking = Mock(identity=identity)
    tracking.update.side_effect = lambda *a: events.append("tracking")
    original_precision = torch.get_float32_matmul_precision()
    original_deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        result = trainer._train(config, tmp_path, tracking, lambda *a: events.append("observer"))
    finally:
        torch.set_float32_matmul_precision(original_precision)
        torch.use_deterministic_algorithms(original_deterministic)
    expected = ["collect", "ppo", "checkpoint", "tracking", "observer"] * (2 - start_update)
    assert events == expected + ["close"] + (["checkpoint"] if start_update == 2 else [])
    assert capture.call_count == (1 if start_update == 0 else 0)
    assert probe_call.call_count == (2 if start_update == 0 else 1)
    assert all(call.args[1] is state.probe_contexts for call in probe_call.call_args_list)
    if contexts is not None:
        assert state.probe_contexts is contexts
    assert state.completed_updates == 2
    assert result.total_transitions == 4
    assert len(result.updates) == 2
    assert payloads[-1]["completed_updates"] == 2
    assert payloads[-1]["tracking"] == identity.model_dump()
    assert payloads[-1]["initial_policy_hash"] == result.initial_policy_hash
    assert exports == (["policy-initial.pt"] if start_update == 0 else []) + [
        f"policy-update-{i:03d}.pt" for i in range(start_update, 2)
    ] + ["policy-final.pt"]
    assert (tmp_path / "summary.json").is_file()


def trainer_policy():
    from eco_planner.rl.policy import ExplorationPolicy

    return ExplorationPolicy(_policy_config())
