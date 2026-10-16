from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict, cat

from eco_planner.planning.policy import ExplorationPolicy
from eco_planner.reward import (
    RewardResult,
    aggregate_scalar_rewards,
)
from eco_planner.rl.artifacts.metrics import build_update_summary
from eco_planner.rl.optimization.batch import build_ppo_batch, compute_gae
from eco_planner.rl.optimization.ppo import PPOUpdater
from eco_planner.rl.rollout.assembly import build_training_decision

from .helpers import (
    _behavior_policy_episode,
    _context,
    _episode,
    _policy_config,
    _ppo_config,
    _training_episode,
)


@pytest.mark.parametrize("truncated", [False, True])
def test_gae_uses_terminal_and_nonterminal_tail_bootstrap_semantics(truncated) -> None:
    config = _ppo_config()
    terminal = compute_gae(
        _training_episode(reward=0.25, terminated=True, truncated=False, bootstrap=0.0).training,
        config,
    )
    bootstrapped = compute_gae(
        _training_episode(
            reward=0.25, terminated=False, truncated=truncated, bootstrap=2.0
        ).training,
        config,
    )

    torch.testing.assert_close(terminal["advantage"], torch.tensor([[-0.75]]))
    torch.testing.assert_close(terminal["value_target"], torch.tensor([[0.25]]))
    torch.testing.assert_close(bootstrapped["advantage"], torch.tensor([[1.23]]))
    torch.testing.assert_close(bootstrapped["value_target"], torch.tensor([[2.23]]))


def test_gae_treats_simultaneous_termination_and_truncation_as_terminal() -> None:
    episode = _training_episode(reward=0.25, terminated=True, truncated=True, bootstrap=0.0)

    assert bool(episode.training["next", "terminated"][-1].item())
    assert bool(episode.training["next", "truncated"][-1].item())
    torch.testing.assert_close(episode.training["next", "state_value"], torch.zeros(1, 1))

    trajectory = compute_gae(episode.training, _ppo_config())
    torch.testing.assert_close(trajectory["advantage"], torch.tensor([[-0.75]]))
    torch.testing.assert_close(trajectory["value_target"], torch.tensor([[0.25]]))


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_combined_gae_respects_episode_boundaries_without_mutation(device) -> None:
    episodes = []
    expected_advantages = []
    for length, (terminated, truncated) in enumerate(
        [(True, False), (False, True), (False, False), (True, True)], start=1
    ):
        single = _training_episode(
            reward=100.0 * length,
            terminated=terminated,
            truncated=truncated,
            bootstrap=0.0 if terminated else 2.0,
        )
        training = cat([single.training] * length)
        training["next", "done"][:-1] = False
        training["next", "terminated"][:-1] = False
        training["next", "truncated"][:-1] = False
        training["next", "state_value"][:-1] = training["state_value"][1:]
        episode = SimpleNamespace(training=training)
        episodes.append(episode)
        # Constant V=1: internal delta=r+gamma-1; only the tail may bootstrap V=2.
        advantage = 100.0 * length - 1.0 + (0.0 if terminated else 0.99 * 2.0)
        episode_advantages = [advantage]
        for _ in range(length - 1):
            advantage = 100.0 * length + 0.99 - 1.0 + 0.99 * 0.95 * advantage
            episode_advantages.insert(0, advantage)
        expected_advantages.extend(episode_advantages)
    snapshots = [e.training.clone() for e in episodes]
    compact = cat([episode.training for episode in episodes]).to(device)
    compact_snapshot = compact.clone()
    batch = build_ppo_batch(compact, _ppo_config(), device=torch.device(device))
    assert (compact == compact_snapshot).all()
    expected = torch.tensor(expected_advantages, device=device).unsqueeze(-1)
    torch.testing.assert_close(batch["advantage"], expected)
    torch.testing.assert_close(batch["value_target"], expected + 1.0)
    assert all(value.device.type == device for value in batch.values())
    for episode, training in zip(episodes, snapshots, strict=True):
        assert (episode.training == training).all()


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_minibatches_cover_each_update_and_use_independent_reproducible_rng(device) -> None:
    config = _ppo_config().model_copy(
        update={"batch_size": 4, "epochs": 2, "scheduler_total_optimizer_steps": 8}
    )
    policy = ExplorationPolicy(_policy_config()).to(device)
    replica = ExplorationPolicy(_policy_config()).to(device)
    replica.load_state_dict(policy.state_dict())
    updaters = [PPOUpdater(model, config) for model in (policy, replica)]
    episodes = [
        tuple(
            _training_episode(reward=float(i + 1), terminated=True, truncated=False, bootstrap=0.0)
            for i in range(start, start + 4)
        )
        for start in (0, 4)
    ]
    sequences = []
    for updater in updaters:
        seen = []

        def observe(module, args, updater=updater, seen=seen):
            minibatch = args[0]
            assert minibatch["value_target"].device == updater.device
            seen.extend(minibatch["value_target"].flatten().tolist())

        hook = updater.loss_module.register_forward_pre_hook(observe)
        cpu_rng = torch.random.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state().clone() if device == "cuda" else None
        generator_before = updater.checkpoint_state()["minibatch_generator_state"].clone()
        for batch in episodes:
            updater.update(cat([item.training for item in batch]))
        hook.remove()
        assert torch.equal(torch.random.get_rng_state(), cpu_rng)
        if cuda_rng is not None:
            assert torch.equal(torch.cuda.get_rng_state(), cuda_rng)
        assert not torch.equal(
            updater.checkpoint_state()["minibatch_generator_state"], generator_before
        )
        for offset, expected in ((0, [1, 2, 3, 4]), (8, [5, 6, 7, 8])):
            for epoch in range(2):
                begin = offset + epoch * 4
                assert sorted(seen[begin : begin + 4]) == expected
        sequences.append(seen)
    assert sequences[0] == sequences[1]


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_update_preserves_fixed_ppo_targets_and_source(device) -> None:
    config = _ppo_config().model_copy(
        update={"batch_size": 4, "epochs": 2, "scheduler_total_optimizer_steps": 4}
    )
    episodes = tuple(
        _training_episode(reward=float(i), terminated=True, truncated=False, bootstrap=0.0)
        for i in range(4)
    )
    snapshots = [episode.training.clone() for episode in episodes]
    expected = build_ppo_batch(
        cat([item.training for item in episodes]), config, device=torch.device(device)
    )
    advantage = expected["advantage"]
    expected["advantage"] = (advantage - advantage.mean()) / advantage.std(correction=1)
    updater = PPOUpdater(ExplorationPolicy(_policy_config()).to(device), config)
    minibatches = []

    def capture_minibatch(module, args):
        minibatch = args[0].select(
            "guidance_action", "old_joint_guidance_log_prob", "advantage", "value_target"
        )
        # Unique returns identify each sample regardless of minibatch order.
        indices = minibatch["value_target"].flatten().long()
        for key in minibatch.keys():
            torch.testing.assert_close(minibatch[key], expected[key][indices])
        minibatches.append((minibatch, minibatch.clone()))

    handle = updater.loss_module.register_forward_pre_hook(capture_minibatch)
    updater.update(cat([item.training for item in episodes]))
    handle.remove()
    assert minibatches
    for actual, snapshot in minibatches:
        assert (actual == snapshot).all()
    for episode, training in zip(episodes, snapshots, strict=True):
        assert (episode.training == training).all()


def test_ratio_diagnostics_cover_full_batch_without_mutation() -> None:
    policy = ExplorationPolicy(_policy_config())
    updater = PPOUpdater(policy, _ppo_config())
    episodes = tuple(
        _behavior_policy_episode(policy, torch.tensor([[i / 10, -i / 10]]), float(i))
        for i in range(5)
    )
    batch = build_ppo_batch(cat([item.training for item in episodes]), _ppo_config())
    ratios = torch.tensor([0.5, 1.0, 2.0, 4.0, 8.0])
    batch["old_joint_guidance_log_prob"] -= ratios.log()
    snapshot = batch.clone()
    statistics = updater._policy_ratio_statistics(batch)

    assert statistics == pytest.approx(
        (ratios.mean(), ratios.std(correction=0), torch.quantile(ratios, 0.95), ratios.max())
    )
    assert (batch == snapshot).all()


@pytest.mark.smoke
def test_ppo_update_changes_policy_and_reports_finite_training_summary() -> None:
    with torch.random.fork_rng():
        torch.manual_seed(0)
        policy = ExplorationPolicy(_policy_config())
    policy_before = {name: value.detach().clone() for name, value in policy.state_dict().items()}
    episodes = (
        _episode(reward=0.25, terminated=True, truncated=False, bootstrap=0.0),
        _episode(reward=2.0, terminated=True, truncated=False, bootstrap=0.0),
    )
    report = PPOUpdater(policy, _ppo_config()).update(cat([item.training for item in episodes]))
    summary = build_update_summary(0, episodes, report)

    assert any(
        not torch.equal(value, policy_before[name]) for name, value in policy.state_dict().items()
    )
    assert report.sample_count == 2
    assert report.evaluated_minibatch_count == 1
    assert report.optimizer_step_count == 1
    assert not report.kl_early_stopped
    assert report.kl_early_stop_trigger is None
    assert report.cumulative_kl_early_stop_count == 0
    assert report.policy_ratio_mean > 0.0
    assert report.policy_ratio_std >= 0.0
    assert report.policy_ratio_p95 > 0.0
    assert report.policy_ratio_max > 0.0
    metrics = (value for value in report.__dict__.values() if isinstance(value, float))
    assert all(math.isfinite(value) for value in metrics)
    assert math.isfinite(report.mean_value_target)
    assert math.isfinite(report.std_value_target)
    assert report.std_value_target >= 0.0
    assert summary.update_index == 0
    assert summary.episode_count == 2
    assert summary.sample_count == report.sample_count
    assert summary.total_reward == 2.25
    assert summary.reward_profile == "plannerrft_energy_v1"


def test_ppo_pairs_each_action_with_its_behavior_log_probability() -> None:
    with torch.random.fork_rng():
        torch.manual_seed(0)
        policy = ExplorationPolicy(_policy_config())
    episodes = (
        _behavior_policy_episode(policy, torch.tensor([[0.0, 0.0]]), reward=0.25),
        _behavior_policy_episode(policy, torch.tensor([[0.98, -0.98]]), reward=2.0),
    )

    report = PPOUpdater(policy, _ppo_config()).update(cat([item.training for item in episodes]))

    assert report.mean_clip_fraction == 0.0
    assert report.mean_approximate_kl == pytest.approx(0.0, abs=1e-7)


def test_target_kl_stops_before_triggering_minibatch_optimizer_step() -> None:
    config = _ppo_config().model_copy(
        update={
            "target_kl": 1e-12,
            "epochs": 4,
            "batch_size": 4,
            "minibatch_size": 2,
            "scheduler_total_optimizer_steps": 8,
        }
    )
    episodes = tuple(
        _training_episode(
            reward=float(index + 1),
            terminated=True,
            truncated=False,
            bootstrap=0.0,
        )
        for index in range(4)
    )
    updater = PPOUpdater(ExplorationPolicy(_policy_config()), config)

    report = updater.update(cat([item.training for item in episodes]))

    assert report.kl_early_stopped
    assert report.kl_early_stop_trigger is not None
    assert report.kl_early_stop_trigger > 1.5 * config.target_kl
    assert report.evaluated_minibatch_count == report.optimizer_step_count + 1
    assert 0 <= report.optimizer_step_count < config.optimizer_steps_per_update
    assert updater.completed_optimizer_steps == report.optimizer_step_count
    assert report.cumulative_kl_early_stop_count == 1


def test_optional_gradient_diagnostics_report_finite_parameter_group_norms() -> None:
    config = _ppo_config().model_copy(update={"gradient_diagnostics": True})
    episodes = (
        _training_episode(reward=0.25, terminated=True, truncated=False, bootstrap=0.0),
        _training_episode(reward=2.0, terminated=True, truncated=False, bootstrap=0.0),
    )

    report = PPOUpdater(ExplorationPolicy(_policy_config()), config).update(
        cat([item.training for item in episodes])
    )

    assert report.gradient_diagnostics is not None
    values = report.gradient_diagnostics.__dict__.values()
    assert all(math.isfinite(value) and value >= 0.0 for value in values)
    assert any(value > 0.0 for value in values)


def test_scheduler_horizon_covers_every_epoch_and_minibatch_across_updates() -> None:
    config = _ppo_config().model_copy(
        update={
            "epochs": 2,
            "batch_size": 4,
            "minibatch_size": 2,
            "scheduler_total_optimizer_steps": 8,
        }
    )
    episodes = tuple(
        _training_episode(
            reward=float(index + 1),
            terminated=True,
            truncated=False,
            bootstrap=0.0,
        )
        for index in range(4)
    )
    updater = PPOUpdater(ExplorationPolicy(_policy_config()), config)

    reports = [updater.update(cat([item.training for item in episodes])) for _ in range(2)]

    assert [report.optimizer_step_count for report in reports] == [4, 4]
    assert updater.completed_optimizer_steps == 8


def test_ppo_accepts_non_gated_scalar_results_without_plannerrft_audit() -> None:
    policy = ExplorationPolicy(_policy_config())
    with torch.no_grad():
        output = policy(_context())
    action = torch.tensor([[-0.5, 0.5]])
    decision = build_training_decision(
        _context(), action, output.distribution.log_prob(action), output.value
    )
    training = cat([decision.clone(), decision.clone()])
    rewards = [
        aggregate_scalar_rewards([RewardResult(-0.25), RewardResult(-0.5)]),
        RewardResult(2.0),
    ]
    training["next"] = TensorDict(
        {
            "reward": torch.tensor([[result.total] for result in rewards]),
            "state_value": torch.zeros(2, 1),
            "done": torch.ones(2, 1, dtype=torch.bool),
            "terminated": torch.ones(2, 1, dtype=torch.bool),
            "truncated": torch.zeros(2, 1, dtype=torch.bool),
        },
        batch_size=[2],
    )
    episode = SimpleNamespace(training=training)
    assert vars(episode).keys() == {"training"}
    assert all(
        not hasattr(result, "safety_gate") and not hasattr(result, "components")
        for result in rewards
    )
    gae = compute_gae(episode.training, _ppo_config())
    torch.testing.assert_close(gae["value_target"], training["next", "reward"])
    before = {name: value.clone() for name, value in policy.state_dict().items()}
    report = PPOUpdater(policy, _ppo_config()).update(episode.training)
    assert report.optimizer_step_count == 1
    assert any(not torch.equal(before[name], value) for name, value in policy.state_dict().items())
