from __future__ import annotations

from pathlib import Path

import pytest
import torch
from lightning.fabric import Fabric

from eco_planner.rl import (
    ExplorationPolicy,
    ExplorationPolicyConfig,
    load_exploration_policy_checkpoint,
    save_exploration_policy_checkpoint,
)
from eco_planner.rl.checkpoint import load_training_checkpoint, save_training_checkpoint
from eco_planner.rl.config import PPOConfig
from eco_planner.rl.ppo import PPOUpdater


def test_policy_checkpoint_round_trip_contains_only_policy_parameters(
    tmp_path: Path, exploration_policy_config: ExplorationPolicyConfig
) -> None:
    source = ExplorationPolicy(exploration_policy_config)
    target = ExplorationPolicy(exploration_policy_config)
    with torch.no_grad():
        source.value_head.bias.fill_(3.0)
    path = tmp_path / "policy.pt"

    saved = save_exploration_policy_checkpoint(path, source)
    loaded = load_exploration_policy_checkpoint(path, target)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)

    assert set(checkpoint) == {"format_version", "policy_state_dict"}
    assert set(checkpoint["policy_state_dict"]) == set(dict(source.named_parameters()))
    assert saved == loaded
    assert saved.parameter_count == sum(parameter.numel() for parameter in source.parameters())
    for source_value, target_value in zip(source.parameters(), target.parameters(), strict=True):
        assert torch.equal(source_value, target_value)


def test_policy_checkpoint_rejects_unexpected_state_key(
    tmp_path: Path, exploration_policy_config: ExplorationPolicyConfig
) -> None:
    policy = ExplorationPolicy(exploration_policy_config)
    path = tmp_path / "invalid.pt"
    torch.save(
        {
            "format_version": 1,
            "policy_state_dict": {**policy.state_dict(), "planner.weight": torch.ones(1)},
        },
        path,
    )

    with pytest.raises(ValueError, match="unexpected"):
        load_exploration_policy_checkpoint(path, policy)


def test_fabric_training_checkpoint_restores_loop_and_ppo_rng_state(
    tmp_path: Path, exploration_policy_config: ExplorationPolicyConfig
) -> None:
    config = PPOConfig(
        name="test",
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coefficient=0.5,
        entropy_coefficient=0.01,
        learning_rate=0.001,
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
    source = ExplorationPolicy(exploration_policy_config)
    source_updater = PPOUpdater(source, config)
    fabric = Fabric(accelerator="cpu", devices=1)
    path = tmp_path / "training.ckpt"
    saved = save_training_checkpoint(
        path,
        fabric,
        source,
        source_updater,
        {"completed_updates": 0, "total_transitions": 0},
    )
    target = ExplorationPolicy(exploration_policy_config)
    target_updater = PPOUpdater(target, config)
    loaded, loop = load_training_checkpoint(path, fabric, target, target_updater)

    assert saved == loaded
    assert loop == {"completed_updates": 0, "total_transitions": 0}
    for source_value, target_value in zip(source.parameters(), target.parameters(), strict=True):
        assert torch.equal(source_value, target_value)
