"""Checkpoint recovery of optimizer, training loop, and random streams."""

from types import SimpleNamespace

import pytest
import torch
from lightning.fabric import Fabric

from eco_planner.planning.policy import ExplorationPolicy
from eco_planner.rl.artifacts import PolicyProbeSummary
from eco_planner.rl.optimization import (
    PPOUpdater,
    load_training_checkpoint,
    save_training_checkpoint,
)
from eco_planner.rl.tracking import TrackingIdentity
from eco_planner.rl.training_state import TrainingLoopState, resume_training_state
from tests.training.helpers import (
    _behavior_policy_episode,
    _config,
    _context,
    _episode,
    _policy_config,
    _ppo_config,
)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("early_stop", [False, True])
def test_next_update_and_checkpoint_preserve_minibatches(tmp_path, early_stop, device) -> None:
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
        assert first.kl_early_stopped
        assert first.cumulative_kl_early_stop_count == 1
        assert first.optimizer_step_count == 0
        assert updater.scheduler.last_epoch == 0
    checkpoint = tmp_path / "state.ckpt"
    fabric = Fabric(accelerator=device)
    saved_rng = torch.random.get_rng_state().clone()
    saved_cuda_rng = (
        [state.clone() for state in torch.cuda.get_rng_state_all()] if device == "cuda" else []
    )
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
    final_rng = torch.random.get_rng_state().clone()
    hook.remove()
    for offset in (0, 4):
        assert sorted(seen[offset : offset + 4]) == pytest.approx([0, 0.1, 0.2, 0.3])
    restored = PPOUpdater(ExplorationPolicy(_policy_config()).to(device), updater.config)
    torch.rand(7)
    if device == "cuda":
        torch.rand(7, device="cuda")
    report, loop = load_training_checkpoint(checkpoint, fabric, restored.policy, restored)
    assert report.completed_updates == loop["completed_updates"] == 1
    assert report.completed_optimizer_steps == first.optimizer_step_count
    assert restored.checkpoint_state()["kl_early_stop_count"] == int(early_stop)
    assert torch.equal(torch.random.get_rng_state(), saved_rng)
    if device == "cuda":
        for actual, saved in zip(torch.cuda.get_rng_state_all(), saved_cuda_rng, strict=True):
            assert torch.equal(actual, saved)
    replayed = []
    hook = restored.loss_module.register_forward_pre_hook(
        lambda module, args: replayed.extend(args[0]["guidance_action"][:, 0].tolist())
    )
    assert restored.update(next_episodes) == expected
    assert torch.equal(torch.random.get_rng_state(), final_rng)
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


@pytest.mark.parametrize(
    "tracked,rng_error",
    [
        pytest.param(False, None, id="untracked-roundtrip"),
        pytest.param(True, None, id="tracked-roundtrip"),
        pytest.param(False, "missing", id="missing-rng-state"),
        pytest.param(False, "count", id="wrong-rng-slot-count"),
        pytest.param(False, "dtype", id="wrong-rng-dtype"),
    ],
)
def test_training_loop_restores_json_mode_summary_and_probe_from_checkpoint(
    tmp_path, summary, tracked, rng_error
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
        diffusion_rng_states=tuple(torch.Generator().manual_seed(i).get_state() for i in (1, 2)),
        policy_rng_states=tuple(torch.Generator().manual_seed(i).get_state() for i in (3, 4)),
    ).checkpoint_payload()
    assert isinstance(loop["update_summaries"][0]["action_mean"], list)
    if rng_error == "missing":
        del loop["diffusion_rng_states"]
    elif rng_error == "count":
        loop["diffusion_rng_states"] = ()
    elif rng_error == "dtype":
        loop["diffusion_rng_states"] = (torch.zeros(3), torch.zeros(3))
    checkpoint = tmp_path / "training-state.ckpt"
    save_training_checkpoint(checkpoint, fabric, policy, updater, loop)
    config = _config(tmp_path, enabled=False, resume=str(checkpoint))
    if rng_error:
        with pytest.raises((TypeError, ValueError), match="diffusion_rng_states"):
            resume_training_state(config, SimpleNamespace(fabric=fabric, policy=policy), updater)
        return
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


def test_rollout_rng_restore_preserves_streams_and_rejects_reordered_slots():
    diffusion = tuple(torch.Generator().manual_seed(i) for i in (10, 11))
    policy = tuple(torch.Generator().manual_seed(i) for i in (20, 21))
    for generator in (*diffusion, *policy):
        torch.rand(7, generator=generator)
    state = TrainingLoopState()
    state.capture_rollout_rng(diffusion, policy)
    expected = [torch.rand(5, generator=g) for g in (*diffusion, *policy)]
    restored_diffusion = tuple(torch.Generator().manual_seed(i) for i in (10, 11))
    restored_policy = tuple(torch.Generator().manual_seed(i) for i in (20, 21))
    global_rng = torch.random.get_rng_state().clone()
    state.restore_rollout_rng(restored_diffusion, restored_policy)
    for generator, values in zip((*restored_diffusion, *restored_policy), expected, strict=True):
        assert torch.equal(torch.rand(5, generator=generator), values)
    assert torch.equal(global_rng, torch.random.get_rng_state())
    with pytest.raises(ValueError, match="logical slot"):
        state.restore_rollout_rng(restored_diffusion[::-1], restored_policy)
