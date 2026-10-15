from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock
from weakref import ref

import pytest
import torch
from tensordict import cat

from eco_planner.planning.policy import (
    ExplorationPolicy,
    ExplorationPolicyContext,
    policy_context_tensordict,
)
from eco_planner.planning.policy.distribution import AffineBeta
from eco_planner.rl import trainer
from eco_planner.rl.artifacts import PolicyProbeSummary
from eco_planner.rl.probing import capture_probe_contexts, probe_policy
from eco_planner.rl.tracking import TrackingIdentity
from eco_planner.rl.training_state import TrainingLoopState

from .helpers import _config, _context, _episode, _policy_config


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
    contexts = (
        tuple(
            replace(_context(), reference_trajectory=torch.full((1, 80, 4), float(i + 1)))
            for i in range(2)
        )
        if start_update
        else None
    )
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
    update_refs = []
    runtime = SimpleNamespace(
        policy=trainer_policy(),
        fabric=object(),
        new_noise_generator=lambda seed: torch.Generator().manual_seed(seed),
        new_policy_generator=lambda seed: torch.Generator().manual_seed(seed),
        frozen_planner_hash=lambda: "f" * 64,
    )
    monkeypatch.setattr(trainer, "create_fabric_rollout_runtime", lambda *a, **k: runtime)
    monkeypatch.setattr(trainer, "resume_training_state", lambda *a: state)
    monkeypatch.setattr(trainer, "write_training_runtime_metadata", lambda *a, **k: None)
    monkeypatch.setattr(
        trainer,
        "write_rollout_episode",
        lambda path, episode: events.append(f"episode:{path.name}"),
    )

    def build_summary(index, episodes, report):
        events.append("summary")
        return summary.model_copy(update={"update_index": index})

    monkeypatch.setattr(trainer, "build_update_summary", build_summary)

    def update(storage):
        assert storage.batch_size == (2,)
        assert storage["state_value"].shape == (2, 1)
        events.append("ppo")

    updater = SimpleNamespace(update=update)
    monkeypatch.setattr(trainer, "PPOUpdater", lambda *a: updater)

    class Collector:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def collect(self, **kwargs):
            assert all(item() is None for item in update_refs)
            events.append("collect")
            episodes = [
                _episode(reward=1.0, terminated=True, truncated=False, bootstrap=0.0)
                for _ in range(2)
            ]
            storage = kwargs["training_storage"]
            storage.update(cat([episode.training for episode in episodes]))
            storage["reference_trajectory"][0].fill_(10 * (events.count("collect") - 1) + 1)
            storage["reference_trajectory"][1].fill_(10 * (events.count("collect") - 1) + 2)
            slots = tuple(
                (replace(episode, training=storage[index : index + 1]),)
                for index, episode in enumerate(episodes)
            )
            update_refs.extend([ref(storage), ref(storage["reference_trajectory"])])
            for slot in slots:
                for episode in slot:
                    update_refs.extend([ref(episode), ref(episode.training)])
            return slots

    monkeypatch.setattr(trainer, "VectorRolloutCollector", Collector)

    def compute_probe(*args):
        events.append("probe")
        return probe

    probe_call = Mock(side_effect=compute_probe)
    monkeypatch.setattr(trainer, "probe_policy", probe_call)
    exports = []

    def export(path, *args):
        exports.append(path.name)
        events.append(f"export:{path.name}")

    monkeypatch.setattr(trainer, "save_exploration_policy_checkpoint", export)
    payloads = []

    def save_checkpoint(path, fabric, policy, updater, payload):
        assert payload["completed_updates"] == start_update + events.count("ppo")
        events.append("checkpoint")
        payloads.append(payload)

    monkeypatch.setattr(trainer, "save_training_checkpoint", save_checkpoint)
    identity = TrackingIdentity(run_id="test", tracking_uri="https://example.org")
    tracking = Mock(identity=identity)
    tracking.probe.side_effect = lambda label, *a: events.append(f"tracking:probe:{label}")
    tracking.update.side_effect = lambda *a: events.append("tracking:update")
    tracking.artifact.side_effect = lambda name: events.append(f"artifact:{name}")
    observed_updates = []

    def observe(update_summary):
        events.append("observer")
        observed_updates.append(update_summary)

    original_precision = torch.get_float32_matmul_precision()
    original_deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        result = trainer._train(config, tmp_path, tracking, observe)
    finally:
        torch.set_float32_matmul_precision(original_precision)
        torch.use_deterministic_algorithms(original_deterministic)
    tracking.attach.assert_called_once_with(runtime.fabric, None)
    expected_events = (
        ["export:policy-initial.pt", "artifact:policy-initial.pt"] if start_update == 0 else []
    )
    for index in range(start_update, 2):
        expected_events.extend(
            ["collect", "episode:slot-0-episode-0.npz", "episode:slot-1-episode-0.npz"]
        )
        if index == 0:
            expected_events.extend(["probe", "tracking:probe:before"])
        expected_events.extend(
            [
                "ppo",
                "summary",
                f"export:policy-update-{index:03d}.pt",
                "checkpoint",
                "tracking:update",
                "observer",
            ]
        )
    expected_events.extend(["probe", "export:policy-final.pt"])
    if start_update == 2:
        expected_events.append("checkpoint")
    expected_events.extend(
        [
            "tracking:probe:after",
            "artifact:summary.json",
            "artifact:policy-final.pt",
            "artifact:training-state.ckpt",
        ]
    )
    assert events == expected_events
    assert all(item() is None for item in update_refs)
    assert observed_updates == list(result.updates[start_update:])
    assert [call.args[0] for call in tracking.update.call_args_list] == observed_updates
    assert probe_call.call_count == (2 if start_update == 0 else 1)
    expected_contexts = contexts if contexts is not None else state.probe_contexts
    for index, context in enumerate(expected_contexts):
        assert torch.all(context.reference_trajectory == index + 1)
    for call in probe_call.call_args_list:
        for actual, expected in zip(call.args[1], expected_contexts, strict=True):
            assert (policy_context_tensordict(actual) == policy_context_tensordict(expected)).all()
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
        episodes[index].training["reference_trajectory"].fill_(-1.0)
        assert torch.all(context.reference_trajectory == index)
    with pytest.raises(RuntimeError, match="one fixed probe context per scenario"):
        capture_probe_contexts(((episodes[0],), ()), 2)


def test_probe_preserves_formula_scenario_order_and_training_random_streams():
    with torch.random.fork_rng():
        torch.manual_seed(11)
        policy = ExplorationPolicy(_policy_config()).eval()
        # Break symmetric initialization so different contexts exercise output ordering.
        torch.nn.init.normal_(policy.actor_head.weight, std=0.1)
        contexts = (_context(), replace(_context(), reference_trajectory=torch.ones(1, 80, 4)))
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


def _sampling_context() -> ExplorationPolicyContext:
    return ExplorationPolicyContext(
        scene_tokens=torch.zeros((2, 2, 12)),
        scene_padding_mask=torch.zeros((2, 2), dtype=torch.bool),
        navigation_tokens=torch.zeros((2, 1, 12)),
        navigation_padding_mask=torch.zeros((2, 1), dtype=torch.bool),
        reference_trajectory=torch.zeros((2, 80, 4)),
    )


def test_beta_policy_action_and_log_prob_are_positive_and_finite() -> None:
    with torch.random.fork_rng():
        torch.manual_seed(0)
        policy = ExplorationPolicy(_policy_config())
    output, action = policy.act(_sampling_context(), "sample", torch.Generator().manual_seed(23))

    assert output.distribution.parameters.alpha.shape == (2, 2)
    assert output.distribution.parameters.beta.shape == (2, 2)
    assert torch.all(output.distribution.parameters.alpha > 0.0)
    assert torch.all(output.distribution.parameters.beta > 0.0)
    assert torch.all((action.base_action > 0.0) & (action.base_action < 1.0))
    assert torch.all((action.guidance_action > -1.0) & (action.guidance_action < 1.0))
    assert all(
        torch.isfinite(value).all()
        for value in (
            output.value,
            action.joint_guidance_log_prob,
            action.joint_guidance_entropy,
        )
    )

    generator = torch.Generator().manual_seed(23)
    replay_state = generator.get_state().clone()
    global_state = torch.random.get_rng_state().clone()
    _, first = policy.act(_sampling_context(), "sample", generator)
    consumed_state = generator.get_state().clone()
    replay_generator = torch.Generator()
    replay_generator.set_state(replay_state)
    _, replay = policy.act(_sampling_context(), "sample", replay_generator)

    torch.testing.assert_close(first.base_action, replay.base_action, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(first.guidance_action, replay.guidance_action, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(
        first.joint_guidance_log_prob,
        replay.joint_guidance_log_prob,
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        first.joint_guidance_entropy,
        replay.joint_guidance_entropy,
        rtol=1e-6,
        atol=1e-6,
    )
    assert torch.equal(consumed_state, replay_generator.get_state())
    assert torch.equal(global_state, torch.random.get_rng_state())


def test_deterministic_policy_mean_does_not_consume_any_rng_state() -> None:
    policy = ExplorationPolicy(_policy_config())
    global_state = torch.random.get_rng_state().clone()

    output, action = policy.act(_sampling_context(), "mean")

    torch.testing.assert_close(action.guidance_action, output.distribution.mean)
    assert torch.equal(global_state, torch.random.get_rng_state())
