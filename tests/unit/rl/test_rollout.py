from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from eco_planner.rl import (
    ExplorationPolicyContext,
    MetaDriveRolloutReward,
    MetaDriveTransitionAudit,
    RolloutBuffer,
    RolloutTransition,
)
from eco_planner.rl.training_artifact import write_partial_rollout


def _context() -> ExplorationPolicyContext:
    return ExplorationPolicyContext(
        scene_tokens=torch.zeros((1, 2, 4), dtype=torch.float32),
        scene_padding_mask=torch.zeros((1, 2), dtype=torch.bool),
        navigation_tokens=torch.zeros((1, 1, 4), dtype=torch.float32),
        navigation_padding_mask=torch.zeros((1, 1), dtype=torch.bool),
        reference_trajectory=torch.zeros((1, 80, 4), dtype=torch.float32),
    )


def _transition(
    *,
    index: int = 0,
    terminated: bool = False,
    truncated: bool = False,
) -> RolloutTransition:
    base = torch.tensor([[0.25, 0.75]], dtype=torch.float32)
    return RolloutTransition(
        policy_context=_context(),
        base_action=base,
        guidance_action=2.0 * base - 1.0,
        old_joint_guidance_log_prob=torch.tensor([0.5], dtype=torch.float32),
        old_value=torch.tensor([1.0], dtype=torch.float32),
        beta_alpha=torch.full((1, 2), 2.0),
        beta_beta=torch.full((1, 2), 2.0),
        initial_noise=torch.zeros((1, 11, 80, 4), dtype=torch.float32),
        diffusion_rng_state=torch.Generator().manual_seed(3).get_state(),
        policy_rng_state=torch.Generator().manual_seed(4).get_state(),
        reward=MetaDriveRolloutReward(
            substep_scores=torch.tensor([0.25], dtype=torch.float32),
            total_score=torch.tensor([0.25], dtype=torch.float32),
            dense_step_scores=torch.tensor([0.25], dtype=torch.float32),
            terminal_override_deltas=torch.tensor([0.0], dtype=torch.float32),
        ),
        audit=MetaDriveTransitionAudit(
            route_completion_delta=0.01,
            distance_m=1.0,
            speed_mps=10.0,
            stopped=False,
            position_error_m=0.0,
            heading_error_rad=0.0,
            arrive_dest=False,
            out_of_road=False,
            crash_vehicle=False,
            crash_object=False,
            crash_building=False,
            crash_human=False,
        ),
        terminated=terminated,
        truncated=truncated,
        bootstrap_mask=not terminated,
        scenario_name="straight",
        map_sequence="S",
        map_seed=1,
        noise_seed=2,
        policy_action_seed=3,
        planning_cycle_index=index,
        executed_substep_count=1,
    )


def test_rollout_buffer_preserves_terminal_and_truncation_bootstrap_contract() -> None:
    terminal = _transition(terminated=True)
    terminal_buffer = RolloutBuffer()
    terminal_buffer.append(terminal)
    episode = terminal_buffer.finalize("terminated", torch.zeros(1, dtype=torch.float32))
    assert episode.tail_kind == "terminated"
    assert not episode.transitions[0].bootstrap_mask
    assert torch.equal(episode.tail_bootstrap_value, torch.zeros(1))

    truncated = _transition(truncated=True)
    truncated_buffer = RolloutBuffer()
    truncated_buffer.append(truncated)
    truncated_episode = truncated_buffer.finalize(
        "truncated", torch.tensor([2.0], dtype=torch.float32)
    )
    assert truncated_episode.transitions[0].bootstrap_mask
    assert truncated_episode.tail_bootstrap_value.item() == pytest.approx(2.0)


def test_rollout_limit_bootstraps_without_crossing_episode_boundary() -> None:
    buffer = RolloutBuffer()
    buffer.append(_transition(index=0))
    buffer.append(_transition(index=1))
    episode = buffer.finalize("rollout_limit", torch.tensor([3.0], dtype=torch.float32))
    assert [step.planning_cycle_index for step in episode.transitions] == [0, 1]
    assert all(step.bootstrap_mask for step in episode.transitions)

    with pytest.raises(ValueError, match="episode boundary"):
        buffer.append(_transition(index=2, terminated=True))
        buffer.append(_transition(index=3))


def test_metadrive_reward_preserves_terminal_override_decomposition() -> None:
    reward = MetaDriveRolloutReward(
        substep_scores=torch.tensor([-5.0]),
        total_score=torch.tensor([-5.0]),
        dense_step_scores=torch.tensor([0.25]),
        terminal_override_deltas=torch.tensor([-5.25]),
    )
    assert reward.source == "metadrive_builtin_v1"


def test_partial_rollout_artifact_retains_collected_prefix(tmp_path) -> None:
    path = tmp_path / "trace.npz"
    write_partial_rollout(path, (_transition(),))

    with np.load(path, allow_pickle=False) as arrays:
        assert arrays["trace_status"].item() == "partial"
        assert arrays["base_action"].shape == (1, 2)
        assert arrays["initial_noise"].shape == (1, 11, 80, 4)
        assert arrays["reward"].tolist() == pytest.approx([0.25])


@pytest.mark.parametrize(
    "update",
    [
        {"executed_substep_count": 5},
        {"old_value": torch.tensor([float("nan")], dtype=torch.float32)},
        {"base_action": torch.tensor([[0.0, 0.5]], dtype=torch.float32)},
        {"diffusion_rng_state": torch.empty(0, dtype=torch.uint8)},
        {"policy_context": replace(_context(), scene_tokens=torch.zeros((1, 2, 4), device="meta"))},
    ],
)
def test_rollout_transition_rejects_invalid_or_mixed_device_fields(
    update: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError), match="substep|finite|strictly|RNG|CPU"):
        RolloutTransition(**(_transition().__dict__ | update))
