from __future__ import annotations

import numpy as np
import pytest
import torch

from eco_planner.rl.artifacts import write_partial_rollout, write_rollout_episode
from eco_planner.rl.policy import ExplorationPolicyContext
from eco_planner.rl.rollout import (
    PPOTrainingTrajectory,
    RolloutAuditTrajectory,
    build_rollout_transition,
    finalize_rollout_episode,
)


def _context() -> ExplorationPolicyContext:
    return ExplorationPolicyContext(
        scene_tokens=torch.zeros((1, 2, 4), dtype=torch.float32),
        scene_padding_mask=torch.zeros((1, 2), dtype=torch.bool),
        navigation_tokens=torch.zeros((1, 1, 4), dtype=torch.float32),
        navigation_padding_mask=torch.zeros((1, 1), dtype=torch.bool),
        reference_trajectory=torch.zeros((1, 80, 4), dtype=torch.float32),
    )


def _transition(index: int = 0, *, terminated: bool = False, truncated: bool = False):
    return build_rollout_transition(
        policy_context=_context(),
        base_action=torch.tensor([[0.25, 0.75]], dtype=torch.float32),
        guidance_action=torch.tensor([[-0.5, 0.5]], dtype=torch.float32),
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


def test_tensor_dict_rollout_preserves_terminal_and_truncation_bootstrap_rules() -> None:
    terminal = finalize_rollout_episode(
        [_transition(terminated=True)], "terminated", torch.zeros(1)
    )
    assert terminal.transition_count == 1
    assert terminal.training_trajectory["next", "done"].item()
    assert terminal.training_trajectory["next", "terminated"].item()

    truncated = finalize_rollout_episode(
        [_transition(truncated=True)], "truncated", torch.tensor([2.0])
    )
    assert truncated.training_trajectory["next", "state_value"].item() == pytest.approx(2.0)


def test_rollout_limit_uses_contiguous_tensor_dict_transitions() -> None:
    episode = finalize_rollout_episode(
        [_transition(0), _transition(1)], "rollout_limit", torch.ones(1)
    )
    assert episode.audit_trajectory["planning_cycle_index"].squeeze(-1).tolist() == [0, 1]
    assert episode.policy_context_at(1).reference_trajectory.shape == (1, 80, 4)


def test_partial_rollout_artifact_converts_only_at_the_npz_boundary(tmp_path) -> None:
    path = tmp_path / "trace.npz"
    write_partial_rollout(path, _transition().audit_trajectory)

    with np.load(path, allow_pickle=False) as arrays:
        assert arrays["trace_status"].item() == "partial"
        assert arrays["base_action"].shape == (1, 2)
        assert arrays["initial_noise"].shape == (1, 11, 80, 4)


def test_complete_rollout_artifact_uses_the_audit_trajectory(tmp_path) -> None:
    episode = finalize_rollout_episode([_transition()], "rollout_limit", torch.zeros(1))
    path = tmp_path / "episode.npz"
    write_rollout_episode(path, episode)

    with np.load(path, allow_pickle=False) as arrays:
        assert arrays["initial_noise"].shape == (1, 11, 80, 4)
        assert arrays["diffusion_rng_state"].dtype == np.uint8
        assert arrays["planning_cycle_index"].item() == 0


def test_transition_rejects_non_interior_action_without_clamping() -> None:
    with pytest.raises(ValueError, match="strictly inside"):
        RolloutAuditTrajectory(
            _transition().audit_trajectory.set("base_action", torch.tensor([[0.0, 0.5]]))
        )


def test_training_and_audit_contracts_validate_at_their_own_boundaries() -> None:
    transition = _transition()
    assert "initial_noise" not in transition.training_trajectory.keys()
    assert "initial_noise" in transition.audit_trajectory.keys()

    with pytest.raises(ValueError, match="PPO training trajectory is missing fields"):
        PPOTrainingTrajectory(transition.training_trajectory.exclude("reward"))
    with pytest.raises(ValueError, match="rollout audit trajectory is missing fields"):
        RolloutAuditTrajectory(transition.audit_trajectory.exclude("initial_noise"))
