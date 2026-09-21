from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch
from tensordict import TensorDictBase, cat

from eco_planner.planning.policy import (
    ExplorationPolicy,
    ExplorationPolicyConfig,
    ExplorationPolicyContext,
    policy_context_tensordict,
)
from eco_planner.reward import RewardComponents, RewardDiagnostics, RewardResult
from eco_planner.reward.result import RewardProfileName
from eco_planner.rl.artifacts import TrainingUpdateSummary, build_update_summary
from eco_planner.rl.optimization import PPOConfig, PPOUpdater, compute_episode_gae
from eco_planner.rl.optimization.ppo import build_ppo_batch
from eco_planner.rl.rollout import (
    ExecutionTransitionAudit,
    RolloutEpisodeBuilder,
    RolloutProvenance,
    build_training_decision,
)


def _policy_config() -> ExplorationPolicyConfig:
    return ExplorationPolicyConfig(
        hidden_dim=12,
        reference_mixer_depth=2,
        reference_token_mlp_hidden_dim=16,
        reference_channel_mlp_hidden_dim=24,
        cross_attention_heads=3,
        cross_attention_dropout=0.0,
        fusion_mlp_depth=2,
        fusion_hidden_dim=16,
        initial_concentration=2.0,
        minimum_concentration=1e-4,
    )


def _ppo_config() -> PPOConfig:
    return PPOConfig(
        name="test",
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        target_kl=None,
        value_coefficient=0.5,
        entropy_coefficient=0.01,
        gradient_diagnostics=False,
        learning_rate=0.00025,
        adam_epsilon=1e-5,
        weight_decay=0.0,
        max_gradient_norm=0.5,
        epochs=1,
        batch_size=2,
        minibatch_size=2,
        minibatch_seed=7,
        scheduler_total_optimizer_steps=1,
        scheduler_minimum_learning_rate=0.0,
    )


def _context() -> ExplorationPolicyContext:
    return ExplorationPolicyContext(
        scene_tokens=torch.zeros((1, 2, 12)),
        scene_padding_mask=torch.zeros((1, 2), dtype=torch.bool),
        navigation_tokens=torch.zeros((1, 1, 12)),
        navigation_padding_mask=torch.zeros((1, 1), dtype=torch.bool),
        reference_trajectory=torch.zeros((1, 80, 4)),
    )


def _decision_audit() -> TensorDictBase:
    context = _context()
    return policy_context_tensordict(context).update(
        dict(
            prediction=torch.zeros((1, 11, 80, 4)),
            initial_noise=torch.zeros((1, 11, 80, 4)),
            base_action=torch.tensor([[0.25, 0.75]]),
            guidance_action=torch.tensor([[-0.5, 0.5]]),
            old_joint_guidance_log_prob=torch.tensor([[0.5]]),
            state_value=torch.tensor([[1.0]]),
            beta_alpha=torch.full((1, 2), 2.0),
            beta_beta=torch.full((1, 2), 2.0),
            diffusion_rng_state=torch.ones((1, 5), dtype=torch.uint8),
            policy_rng_state=torch.ones((1, 5), dtype=torch.uint8),
        )
    )


def _execution_audit(
    reward: float,
    *,
    terminated: bool,
    truncated: bool,
    profile_name: RewardProfileName = "plannerrft_energy_v1",
) -> ExecutionTransitionAudit:
    return ExecutionTransitionAudit(
        reward_result=RewardResult(
            profile_name=profile_name,
            total=reward,
            base_total=reward,
            safety_gate=1.0,
            components=RewardComponents(1.0, 0.5, 1.0, 1.0, 0.5),
            diagnostics=RewardDiagnostics(
                collision_score=1.0,
                drivable_score=1.0,
                wrong_direction_score=1.0,
                has_ttc_candidate=False,
                min_ttc_s=10.0,
                route_progress_delta_m=1.0,
                speed_mps=2.0,
                speed_limit_mps=10.0,
                overspeed_mps=0.0,
                longitudinal_acceleration_mps2=0.0,
                lateral_acceleration_mps2=0.0,
                jerk_mps3=0.0,
                yaw_rate_radps=0.0,
                step_distance_m=1.0,
                native_step_energy_ml=0.0,
                native_episode_energy_ml=0.0,
                executed_fuel_proxy_step_energy_ml=0.05,
                executed_fuel_proxy_ml_per_km=50.0,
                energy_distance_valid=True,
            ),
        ),
        route_completion_delta=0.1,
        distance_m=1.0,
        speed_mps=2.0,
        stopped=False,
        position_error_m=0.0,
        heading_error_rad=0.0,
        arrive_dest=False,
        out_of_road=False,
        crash_vehicle=False,
        crash_object=False,
        crash_building=False,
        crash_human=False,
        crash_sidewalk=False,
        terminated=terminated,
        truncated=truncated,
    )


def _episode(
    *,
    reward: float,
    terminated: bool,
    truncated: bool,
    bootstrap: float,
    profile_name: RewardProfileName = "plannerrft_energy_v1",
):
    context = _context()
    decision = build_training_decision(
        context,
        torch.tensor([[-0.5, 0.5]]),
        torch.tensor([0.5]),
        torch.tensor([1.0]),
    )
    builder = RolloutEpisodeBuilder()
    builder.append(
        decision,
        _decision_audit(),
        _execution_audit(
            reward, terminated=terminated, truncated=truncated, profile_name=profile_name
        ),
        RolloutProvenance(0, 1, 2, 0),
    )
    tail_kind = "terminated" if terminated else "truncated" if truncated else "rollout_limit"
    return builder.finish(tail_kind, torch.tensor([bootstrap]))


def _behavior_policy_episode(
    policy: ExplorationPolicy, guidance_action: torch.Tensor, reward: float
):
    context = _context()
    with torch.no_grad():
        outputs = policy.forward_tensordict(
            policy_context_tensordict(context).to(next(policy.parameters()).device)
        ).cpu()
        distribution = policy.output_from_tensordict(outputs).distribution
        old_log_prob = distribution.log_prob(guidance_action)
    decision = build_training_decision(
        context,
        guidance_action,
        old_log_prob,
        outputs["state_value"],
    )
    decision_audit = _decision_audit().update(
        dict(
            base_action=(guidance_action + 1.0) / 2.0,
            guidance_action=guidance_action,
            old_joint_guidance_log_prob=old_log_prob.reshape(-1, 1),
            state_value=outputs["state_value"],
            beta_alpha=outputs["alpha"],
            beta_beta=outputs["beta"],
        )
    )
    builder = RolloutEpisodeBuilder()
    builder.append(
        decision,
        decision_audit,
        _execution_audit(reward, terminated=True, truncated=False),
        RolloutProvenance(0, 1, 2, 0),
    )
    return builder.finish("terminated", torch.zeros(1))


@pytest.mark.parametrize("truncated", [False, True])
def test_gae_uses_terminal_and_nonterminal_tail_bootstrap_semantics(truncated) -> None:
    config = _ppo_config()
    terminal = compute_episode_gae(
        _episode(reward=0.25, terminated=True, truncated=False, bootstrap=0.0), config
    )
    bootstrapped = compute_episode_gae(
        _episode(reward=0.25, terminated=False, truncated=truncated, bootstrap=2.0), config
    )

    torch.testing.assert_close(terminal["advantage"], torch.tensor([[-0.75]]))
    torch.testing.assert_close(terminal["value_target"], torch.tensor([[0.25]]))
    torch.testing.assert_close(bootstrapped["advantage"], torch.tensor([[1.23]]))
    torch.testing.assert_close(bootstrapped["value_target"], torch.tensor([[2.23]]))


def test_gae_treats_simultaneous_termination_and_truncation_as_terminal() -> None:
    episode = _episode(reward=0.25, terminated=True, truncated=True, bootstrap=0.0)

    assert episode.tail_kind == "terminated"
    assert bool(episode.training["next", "terminated"][-1].item())
    assert bool(episode.training["next", "truncated"][-1].item())
    torch.testing.assert_close(episode.tail_bootstrap_value, torch.zeros(1))

    trajectory = compute_episode_gae(episode, _ppo_config())
    torch.testing.assert_close(trajectory["advantage"], torch.tensor([[-0.75]]))
    torch.testing.assert_close(trajectory["value_target"], torch.tensor([[0.25]]))


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_combined_gae_respects_episode_boundaries_without_mutation(device) -> None:
    episodes = []
    expected_advantages = []
    for length, (terminated, truncated) in enumerate(
        [(True, False), (False, True), (False, False), (True, True)], start=1
    ):
        single = _episode(
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
        episode = replace(single, training=training, audit=cat([single.audit] * length))
        episodes.append(episode)
        # Constant V=1: internal delta=r+gamma-1; only the tail may bootstrap V=2.
        advantage = 100.0 * length - 1.0 + (0.0 if terminated else 0.99 * 2.0)
        episode_advantages = [advantage]
        for _ in range(length - 1):
            advantage = 100.0 * length + 0.99 - 1.0 + 0.99 * 0.95 * advantage
            episode_advantages.insert(0, advantage)
        expected_advantages.extend(episode_advantages)
    snapshots = [(e.training.clone(), e.audit.clone()) for e in episodes]
    batch = build_ppo_batch(episodes, _ppo_config(), device=torch.device(device))
    expected = torch.tensor(expected_advantages, device=device).unsqueeze(-1)
    torch.testing.assert_close(batch["advantage"], expected)
    torch.testing.assert_close(batch["value_target"], expected + 1.0)
    assert all(value.device.type == device for value in batch.values())
    for episode, (training, audit) in zip(episodes, snapshots, strict=True):
        assert (episode.training == training).all()
        assert (episode.audit == audit).all()


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
            _episode(reward=float(i + 1), terminated=True, truncated=False, bootstrap=0.0)
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
            updater.update(batch)
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
@pytest.mark.parametrize("early_stop", [False, True])
def test_next_update_and_checkpoint_preserve_minibatches(tmp_path, early_stop, device) -> None:
    from lightning.fabric import Fabric

    from eco_planner.rl.optimization import load_training_checkpoint, save_training_checkpoint

    config = _ppo_config().model_copy(
        update={
            "batch_size": 4,
            "epochs": 2,
            "scheduler_total_optimizer_steps": 16,
            "target_kl": 1e-12 if early_stop else None,
        }
    )
    policy = ExplorationPolicy(_policy_config()).to(device)
    updater = PPOUpdater(policy, config)
    episodes = tuple(
        _episode(reward=float(i), terminated=True, truncated=False, bootstrap=0.0) for i in range(4)
    )
    first = updater.update(episodes)
    if early_stop:
        assert first.optimizer_step_count == 0
        assert updater.scheduler.last_epoch == 0
    checkpoint = tmp_path / "state.ckpt"
    fabric = Fabric(accelerator=device)
    save_training_checkpoint(checkpoint, fabric, policy, updater, {"completed_updates": 1})
    # The next batch has unique actions, all paired with the updated behavior policy.
    next_episodes = tuple(
        _behavior_policy_episode(policy, torch.tensor([[i / 10, -i / 10]]), float(i + 1))
        for i in range(4)
    )
    updater.config = config.model_copy(update={"target_kl": None})
    seen = []
    hook = updater.loss_module.register_forward_pre_hook(
        lambda module, args: seen.extend(args[0]["guidance_action"][:, 0].tolist())
    )
    expected = updater.update(next_episodes)
    hook.remove()
    for offset in (0, 4):
        assert sorted(seen[offset : offset + 4]) == pytest.approx([0, 0.1, 0.2, 0.3])
    restored = PPOUpdater(ExplorationPolicy(_policy_config()).to(device), updater.config)
    load_training_checkpoint(checkpoint, fabric, restored.policy, restored)
    replayed = []
    hook = restored.loss_module.register_forward_pre_hook(
        lambda module, args: replayed.extend(args[0]["guidance_action"][:, 0].tolist())
    )
    assert restored.update(next_episodes) == expected
    hook.remove()
    assert seen == replayed
    for name, parameter in policy.state_dict().items():
        assert torch.equal(parameter, restored.policy.state_dict()[name])
    assert updater.scheduler.state_dict() == restored.scheduler.state_dict()
    actual_optimizer = updater.optimizer.state_dict()
    restored_optimizer = restored.optimizer.state_dict()
    assert actual_optimizer["param_groups"] == restored_optimizer["param_groups"]
    for key, state in actual_optimizer["state"].items():
        for field, value in state.items():
            torch.testing.assert_close(
                value, restored_optimizer["state"][key][field], rtol=0, atol=0
            )
    assert torch.equal(
        updater.checkpoint_state()["minibatch_generator_state"],
        restored.checkpoint_state()["minibatch_generator_state"],
    )


def test_checkpoint_rejects_old_sampler_state() -> None:
    updater = PPOUpdater(ExplorationPolicy(_policy_config()), _ppo_config())
    old_state = {**updater.checkpoint_state(), "minibatch_sampler_state": {}}
    with pytest.raises(ValueError, match="unexpected fields"):
        updater.restore_checkpoint_state(old_state)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_update_shares_batch_storage_and_preserves_fixed_ppo_targets(monkeypatch, device) -> None:
    from eco_planner.rl.optimization import ppo

    config = _ppo_config().model_copy(
        update={"batch_size": 4, "epochs": 2, "scheduler_total_optimizer_steps": 4}
    )
    episodes = tuple(
        _episode(reward=float(i), terminated=True, truncated=False, bootstrap=0.0) for i in range(4)
    )
    snapshots = [(episode.training.clone(), episode.audit.clone()) for episode in episodes]
    normalized_batches = []
    normalize = ppo.normalize_full_batch_advantage
    storages = []
    tensor_storage = ppo.TensorStorage

    def capture_storage(batch, **kwargs):
        storage = tensor_storage(batch, **kwargs)
        stored = storage.get(slice(None))
        assert len(storage) == config.batch_size
        for key in batch.keys():
            assert stored[key].data_ptr() == batch[key].data_ptr()
            assert stored[key].device.type == device
        storages.append(storage)
        return storage

    def capture_normalized_batch(batch):
        normalize(batch)
        normalized_batches.append((batch, batch.clone()))

    monkeypatch.setattr(ppo, "normalize_full_batch_advantage", capture_normalized_batch)
    monkeypatch.setattr(ppo, "TensorStorage", capture_storage)
    updater = PPOUpdater(ExplorationPolicy(_policy_config()).to(device), config)
    minibatches = []

    def capture_minibatch(module, args):
        minibatch = args[0].select(
            "guidance_action", "old_joint_guidance_log_prob", "advantage", "value_target"
        )
        minibatches.append((minibatch, minibatch.clone()))

    handle = updater.loss_module.register_forward_pre_hook(capture_minibatch)
    updater.update(episodes)
    handle.remove()
    assert len(normalized_batches) == 1
    assert len(storages) == 1
    assert len(minibatches) == 4
    for actual, expected in normalized_batches + minibatches:
        assert (actual == expected).all()
    for episode, (training, audit) in zip(episodes, snapshots, strict=True):
        assert (episode.training == training).all()
        assert (episode.audit == audit).all()


def test_ratio_diagnostics_cover_full_batch_with_bounded_forward_size(monkeypatch) -> None:
    policy = ExplorationPolicy(_policy_config())
    updater = PPOUpdater(policy, _ppo_config())
    episodes = tuple(
        _behavior_policy_episode(policy, torch.tensor([[i / 10, -i / 10]]), float(i))
        for i in range(5)
    )
    batch = build_ppo_batch(episodes, _ppo_config())
    ratios = torch.tensor([0.5, 1.0, 2.0, 4.0, 8.0])
    batch["old_joint_guidance_log_prob"] -= ratios.log()
    snapshot = batch.clone()
    sizes = []
    forward = policy.forward_tensordict

    def observe(context):
        sizes.append(context.batch_size[0])
        assert not torch.is_grad_enabled()
        return forward(context)

    monkeypatch.setattr(policy, "forward_tensordict", observe)
    statistics = updater._policy_ratio_statistics(batch)

    assert sizes == [2, 2, 1]
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
    report = PPOUpdater(policy, _ppo_config()).update(episodes)
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
    assert isinstance(summary, TrainingUpdateSummary)
    assert summary.update_index == 0
    assert summary.episode_count == 2
    assert summary.sample_count == 2
    assert math.isclose(summary.mean_episode_length, 1.0)
    assert math.isfinite(summary.mean_state_value)
    assert math.isfinite(summary.std_state_value)
    assert summary.std_state_value >= 0.0
    assert math.isfinite(summary.mean_value_target)
    assert math.isfinite(summary.std_value_target)
    assert summary.std_value_target >= 0.0
    for field in (
        summary.beta_alpha_mean,
        summary.beta_alpha_min,
        summary.beta_alpha_max,
        summary.beta_beta_mean,
        summary.beta_beta_min,
        summary.beta_beta_max,
        summary.action_mean,
        summary.action_std,
        summary.action_min,
        summary.action_max,
    ):
        assert len(field) == 2
        assert all(math.isfinite(value) for value in field)
    for dim in range(2):
        assert summary.beta_alpha_min[dim] <= summary.beta_alpha_mean[dim]
        assert summary.beta_alpha_mean[dim] <= summary.beta_alpha_max[dim]
        assert summary.beta_beta_min[dim] <= summary.beta_beta_mean[dim]
        assert summary.beta_beta_mean[dim] <= summary.beta_beta_max[dim]
        assert summary.action_min[dim] <= summary.action_mean[dim]
        assert summary.action_mean[dim] <= summary.action_max[dim]
        assert summary.action_std[dim] >= 0.0
    assert summary.reward_profile == "plannerrft_energy_v1"


@pytest.mark.parametrize(
    "profile_name",
    [
        "plannerrft_energy_v1",
        "plannerrft_energy_band_lam64_v1",
        "plannerrft_no_energy_v1",
        "plannerrft_no_energy_calibrated_v1",
    ],
)
def test_update_summary_persists_every_reward_profile_name(profile_name) -> None:
    episodes = (
        _episode(
            reward=0.25,
            terminated=True,
            truncated=False,
            bootstrap=0.0,
            profile_name=profile_name,
        ),
        _episode(
            reward=2.0,
            terminated=True,
            truncated=False,
            bootstrap=0.0,
            profile_name=profile_name,
        ),
    )

    summary = build_update_summary(0, episodes, _update_report(episodes))

    assert isinstance(summary, TrainingUpdateSummary)
    assert summary.reward_profile == profile_name


def _update_report(episodes):
    with torch.random.fork_rng():
        torch.manual_seed(0)
        policy = ExplorationPolicy(_policy_config())
    return PPOUpdater(policy, _ppo_config()).update(episodes)


def test_ppo_pairs_each_action_with_its_behavior_log_probability() -> None:
    with torch.random.fork_rng():
        torch.manual_seed(0)
        policy = ExplorationPolicy(_policy_config())
    episodes = (
        _behavior_policy_episode(policy, torch.tensor([[0.0, 0.0]]), reward=0.25),
        _behavior_policy_episode(policy, torch.tensor([[0.98, -0.98]]), reward=2.0),
    )

    report = PPOUpdater(policy, _ppo_config()).update(episodes)

    assert report.mean_clip_fraction == 0.0
    assert report.mean_approximate_kl == pytest.approx(0.0, abs=1e-7)


def test_target_kl_stops_before_triggering_minibatch_optimizer_step_and_resumes_state() -> None:
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
        _episode(
            reward=float(index + 1),
            terminated=True,
            truncated=False,
            bootstrap=0.0,
        )
        for index in range(4)
    )
    updater = PPOUpdater(ExplorationPolicy(_policy_config()), config)

    report = updater.update(episodes)

    assert report.kl_early_stopped
    assert report.kl_early_stop_trigger is not None
    assert report.kl_early_stop_trigger > 1.5 * config.target_kl
    assert report.evaluated_minibatch_count == report.optimizer_step_count + 1
    assert 0 <= report.optimizer_step_count < config.optimizer_steps_per_update
    assert updater.completed_optimizer_steps == report.optimizer_step_count
    restored = PPOUpdater(ExplorationPolicy(_policy_config()), config)
    restored.restore_checkpoint_state(updater.checkpoint_state())
    assert restored.checkpoint_state()["completed_optimizer_steps"] == report.optimizer_step_count
    assert restored.checkpoint_state()["kl_early_stop_count"] == 1


def test_optional_gradient_diagnostics_report_finite_parameter_group_norms() -> None:
    config = _ppo_config().model_copy(update={"gradient_diagnostics": True})
    episodes = (
        _episode(reward=0.25, terminated=True, truncated=False, bootstrap=0.0),
        _episode(reward=2.0, terminated=True, truncated=False, bootstrap=0.0),
    )

    report = PPOUpdater(ExplorationPolicy(_policy_config()), config).update(episodes)

    assert report.gradient_diagnostics is not None
    values = report.gradient_diagnostics.__dict__.values()
    assert all(math.isfinite(value) and value >= 0.0 for value in values)
    assert any(value > 0.0 for value in values)


def test_scheduler_horizon_covers_every_epoch_and_minibatch_across_updates() -> None:
    config = _ppo_config().model_copy(
        update={
            "epochs": 4,
            "batch_size": 4,
            "minibatch_size": 2,
            "scheduler_total_optimizer_steps": 32,
        }
    )
    episodes = tuple(
        _episode(
            reward=float(index + 1),
            terminated=True,
            truncated=False,
            bootstrap=0.0,
        )
        for index in range(4)
    )
    updater = PPOUpdater(ExplorationPolicy(_policy_config()), config)

    reports = [updater.update(episodes) for _ in range(4)]

    assert [report.optimizer_step_count for report in reports] == [8, 8, 8, 8]
    assert updater.completed_optimizer_steps == 32
