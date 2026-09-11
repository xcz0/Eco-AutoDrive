from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from eco_planner.rl import trainer
from eco_planner.rl.artifacts import PolicyProbeSummary
from eco_planner.rl.policy import ExplorationPolicy
from eco_planner.rl.policy.distribution import AffineBeta
from eco_planner.rl.probing import capture_probe_contexts, probe_policy
from eco_planner.rl.tracking import TrackingIdentity
from eco_planner.rl.training_state import TrainingLoopState
from tests.training.test_ppo import _context, _episode, _policy_config
from tests.training.test_tracking import _config
from tests.training.test_tracking import summary as summary


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
    return ExplorationPolicy(_policy_config())


def test_capture_uses_first_transition_of_first_episode_in_scenario_order():
    episodes = tuple(
        _episode(reward=1.0, terminated=True, truncated=False, bootstrap=0.0) for _ in range(3)
    )
    for index, episode in enumerate(episodes):
        episode.training["reference_trajectory"].fill_(index)
    contexts = capture_probe_contexts(((episodes[0], episodes[1]), (episodes[2],)), 2)
    assert len(contexts) == 2
    for context, index in zip(contexts, (0, 2), strict=True):
        assert context.reference_trajectory.shape == (1, 80, 4)
        assert torch.equal(
            context.reference_trajectory, episodes[index].training["reference_trajectory"]
        )
    with pytest.raises(RuntimeError, match="one fixed probe context per scenario"):
        capture_probe_contexts(((episodes[0],), ()), 2)


def test_probe_preserves_formula_scenario_order_and_training_random_streams():
    with torch.random.fork_rng():
        torch.manual_seed(11)
        policy = ExplorationPolicy(_policy_config()).eval()
        # Break symmetric initialization so different contexts exercise output ordering.
        torch.nn.init.normal_(policy.actor_head.weight, std=0.1)
        contexts = (_context(), replace(_context(), reference_trajectory=torch.ones(1, 80, 4)))
        noise_generator = torch.Generator().manual_seed(101)
        action_generator = torch.Generator().manual_seed(102)
        noise_before = noise_generator.get_state().clone()
        action_before = action_generator.get_state().clone()
        seeds = []

        def new_generator(seed):
            seeds.append(seed)
            return torch.Generator().manual_seed(seed)

        runtime = SimpleNamespace(
            policy=policy, device=torch.device("cpu"), new_policy_generator=new_generator
        )
        global_before = torch.random.get_rng_state().clone()
        actual = probe_policy(runtime, contexts, 128, 0.1, 40)
        assert seeds == [40, 41]
        assert probe_policy(runtime, contexts, 128, 0.1, 40) == actual
        assert torch.equal(torch.random.get_rng_state(), global_before)
        assert torch.equal(noise_generator.get_state(), noise_before)
        assert torch.equal(action_generator.get_state(), action_before)
        assert actual.alpha[0] != actual.alpha[1]
        for index, context in enumerate(contexts):
            with torch.no_grad():
                distribution = policy(context).distribution
            alpha = distribution.parameters.alpha
            beta = distribution.parameters.beta
            samples = (
                AffineBeta(alpha.expand(128, -1), beta.expand(128, -1))
                .sample(torch.Generator().manual_seed(40 + index))
                .base_action
            )
            boundary = ((samples <= 0.1) | (samples >= 0.9)).float().mean(dim=0)
            assert actual.alpha[index] == tuple(alpha[0].tolist())
            assert actual.beta[index] == tuple(beta[0].tolist())
            assert actual.guidance_mean[index] == tuple(distribution.mean[0].tolist())
            assert actual.boundary_mass[index] == tuple(boundary.tolist())
