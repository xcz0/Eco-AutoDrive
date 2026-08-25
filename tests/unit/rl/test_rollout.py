from __future__ import annotations

import numpy as np
import pytest
import torch
from tensordict import TensorDictBase

from eco_planner.rl.artifacts import write_rollout_episode
from eco_planner.rl.policy import ExplorationPolicyContext
from eco_planner.rl.rollout import (
    build_rollout_audit,
    build_training_decision,
    build_training_transition,
    finalize_rollout_episode,
    set_training_transition_next_state_value,
)


def _context() -> ExplorationPolicyContext:
    return ExplorationPolicyContext(
        scene_tokens=torch.zeros((1, 2, 4), dtype=torch.float32),
        scene_padding_mask=torch.zeros((1, 2), dtype=torch.bool),
        navigation_tokens=torch.zeros((1, 1, 4), dtype=torch.float32),
        navigation_padding_mask=torch.zeros((1, 1), dtype=torch.bool),
        reference_trajectory=torch.zeros((1, 80, 4), dtype=torch.float32),
    )


def _transitions(
    index: int = 0, *, terminated: bool = False, truncated: bool = False
) -> tuple[TensorDictBase, TensorDictBase]:
    context = _context()
    base_action = torch.tensor([[0.25, 0.75]], dtype=torch.float32)
    guidance_action = 2.0 * base_action - 1.0
    decision = build_training_decision(
        context,
        guidance_action,
        torch.tensor([0.5], dtype=torch.float32),
        torch.tensor([1.0], dtype=torch.float32),
    )
    training = build_training_transition(
        decision, reward=0.25, terminated=terminated, truncated=truncated
    )
    audit = build_rollout_audit(
        policy_context=context,
        base_action=base_action,
        guidance_action=guidance_action,
        old_joint_guidance_log_prob=torch.tensor([0.5], dtype=torch.float32),
        state_value=torch.tensor([1.0], dtype=torch.float32),
        beta_alpha=torch.full((1, 2), 2.0),
        beta_beta=torch.full((1, 2), 2.0),
        initial_noise=torch.zeros((1, 11, 80, 4), dtype=torch.float32),
        diffusion_rng_state=torch.ones(5, dtype=torch.uint8),
        policy_rng_state=torch.ones(5, dtype=torch.uint8),
        reward=0.25,
        dense_reward=0.25,
        terminal_override=0.0,
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
        terminated=terminated,
        truncated=truncated,
        map_seed=0,
        noise_seed=1,
        policy_action_seed=2,
        planning_cycle_index=index,
    )
    return training, audit


def test_rollout_tensor_dicts_preserve_terminal_and_truncation_bootstrap_rules() -> None:
    training, audit = _transitions(terminated=True)
    terminal = finalize_rollout_episode([training], [audit], "terminated", torch.zeros(1))
    assert terminal.transition_count == 1
    assert terminal.training["next", "done"].item()
    assert terminal.training["next", "terminated"].item()

    training, audit = _transitions(truncated=True)
    truncated = finalize_rollout_episode([training], [audit], "truncated", torch.tensor([2.0]))
    assert truncated.training["next", "state_value"].item() == pytest.approx(2.0)


def test_rollout_limit_preserves_audit_and_training_device_ownership() -> None:
    first_training, first_audit = _transitions(0)
    second_training, second_audit = _transitions(1)
    set_training_transition_next_state_value(first_training, second_training["state_value"])
    episode = finalize_rollout_episode(
        [first_training, second_training],
        [first_audit, second_audit],
        "rollout_limit",
        torch.ones(1),
    )
    assert episode.audit["planning_cycle_index"].squeeze(-1).tolist() == [0, 1]
    assert episode.training["scene_tokens"].device.type == "cpu"
    assert episode.audit["scene_tokens"].device.type == "cpu"


def test_rollout_links_interior_next_value_and_injects_only_the_tail_bootstrap() -> None:
    first_training, first_audit = _transitions(0)
    second_training, second_audit = _transitions(1)
    set_training_transition_next_state_value(first_training, torch.tensor([3.0]))

    episode = finalize_rollout_episode(
        [first_training, second_training],
        [first_audit, second_audit],
        "rollout_limit",
        torch.tensor([5.0]),
    )

    assert episode.training["next", "state_value"].squeeze(-1).tolist() == [3.0, 5.0]
    assert episode.training["next", "reward"].squeeze(-1).tolist() == [0.25, 0.25]
    assert episode.training["next", "done"].squeeze(-1).tolist() == [False, True]


def test_complete_rollout_artifact_uses_the_audit_tensor_dict(tmp_path) -> None:
    training, audit = _transitions()
    episode = finalize_rollout_episode([training], [audit], "rollout_limit", torch.zeros(1))
    path = tmp_path / "episode.npz"
    write_rollout_episode(path, episode)

    with np.load(path, allow_pickle=False) as arrays:
        assert arrays["initial_noise"].shape == (1, 11, 80, 4)
        assert arrays["diffusion_rng_state"].dtype == np.uint8
        assert arrays["planning_cycle_index"].item() == 0


def test_rollout_validates_complete_audit_at_the_episode_boundary() -> None:
    training, audit = _transitions()
    with pytest.raises(ValueError, match="strictly inside"):
        finalize_rollout_episode(
            [training],
            [audit.set("base_action", torch.tensor([[0.0, 0.5]]))],
            "rollout_limit",
            torch.zeros(1),
        )


def test_training_transition_owns_detached_policy_features() -> None:
    context = _context()
    decision = build_training_decision(
        context,
        torch.tensor([[-0.5, 0.5]], dtype=torch.float32),
        torch.tensor([0.5], dtype=torch.float32),
        torch.tensor([1.0], dtype=torch.float32),
    )
    expected_tokens = decision["scene_tokens"].clone()
    context.scene_tokens.add_(1.0)
    training = build_training_transition(decision, reward=0.25, terminated=False, truncated=True)

    assert not training["scene_tokens"].requires_grad
    torch.testing.assert_close(training["scene_tokens"], expected_tokens)
    assert "reward" not in training.keys()
    assert set(training["next"].keys()) == {"reward", "done", "terminated", "truncated"}
    assert training["next", "reward"].item() == pytest.approx(0.25)
    assert training["next", "done"].item()
