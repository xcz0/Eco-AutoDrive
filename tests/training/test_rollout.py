from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from eco_planner.envs.metadrive.reward import MetaDriveBuiltinRewardAudit
from eco_planner.rl.artifacts import (
    BUILTIN_ROLLOUT_ARTIFACT_FIELDS,
    write_rollout_episode,
)
from eco_planner.rl.policy import ExplorationPolicyContext
from eco_planner.rl.rollout import (
    DecisionAudit,
    ExecutionTransitionAudit,
    RolloutEpisodeBuilder,
    RolloutProvenance,
    build_training_decision,
)


def _context() -> ExplorationPolicyContext:
    return ExplorationPolicyContext(
        scene_tokens=torch.zeros((1, 2, 4)),
        scene_padding_mask=torch.zeros((1, 2), dtype=torch.bool),
        navigation_tokens=torch.zeros((1, 1, 4)),
        navigation_padding_mask=torch.zeros((1, 1), dtype=torch.bool),
        reference_trajectory=torch.zeros((1, 80, 4)),
    )


def _transition():
    context = _context()
    training_decision = build_training_decision(
        context,
        torch.tensor([[-0.5, 0.5]]),
        torch.tensor([0.5]),
        torch.tensor([1.0]),
    )
    decision_audit = DecisionAudit(
        prediction=np.zeros((1, 11, 80, 4), dtype=np.float32),
        initial_noise=torch.zeros((1, 11, 80, 4)),
        policy_context=context,
        base_action=torch.tensor([[0.25, 0.75]]),
        guidance_action=torch.tensor([[-0.5, 0.5]]),
        old_joint_guidance_log_prob=torch.tensor([0.5]),
        old_value=torch.tensor([1.0]),
        beta_alpha=torch.full((1, 2), 2.0),
        beta_beta=torch.full((1, 2), 2.0),
        diffusion_rng_state=torch.ones(5, dtype=torch.uint8),
        policy_rng_state=torch.ones(5, dtype=torch.uint8),
    )
    reward_audit = MetaDriveBuiltinRewardAudit(
        profile_name="metadrive_builtin_v1",
        reward_total=0.25,
        dense_reward=0.25,
        terminal_override=0.0,
        step_distance_m=1.0,
        native_step_energy_ml=0.0,
        native_episode_energy_ml=0.0,
        executed_fuel_proxy_step_energy_ml=0.05,
        executed_fuel_proxy_ml_per_km=50.0,
        energy_distance_valid=True,
    )
    execution_audit = ExecutionTransitionAudit(
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
        crash_sidewalk=False,
        terminated=False,
        truncated=False,
        reward_audit=reward_audit,
    )
    provenance = RolloutProvenance(0, 1, 2, 0)
    return training_decision, decision_audit, execution_audit, provenance


def test_rollout_links_next_values_and_uses_tail_bootstrap_only_at_boundary() -> None:
    builder = RolloutEpisodeBuilder()
    builder.append(*_transition())
    builder.link_next_state_value(torch.tensor([3.0]))
    builder.append(*_transition())

    episode = builder.finish("rollout_limit", torch.tensor([5.0]))

    assert episode.training["next", "state_value"].squeeze(-1).tolist() == [3.0, 5.0]
    assert episode.training["next", "done"].squeeze(-1).tolist() == [False, True]
    assert episode.transition_count == 2
    assert episode.reward_profile == "metadrive_builtin_v1"


def test_rollout_artifact_uses_the_explicit_reward_profile_schema(tmp_path: Path) -> None:
    builder = RolloutEpisodeBuilder()
    builder.append(*_transition())
    episode = builder.finish("rollout_limit", torch.tensor([5.0]))
    artifact = tmp_path / "episode.npz"

    write_rollout_episode(artifact, episode)

    with np.load(artifact, allow_pickle=False) as arrays:
        assert set(arrays.files) == set(BUILTIN_ROLLOUT_ARTIFACT_FIELDS)
        assert str(arrays["reward_profile"]) == "metadrive_builtin_v1"
