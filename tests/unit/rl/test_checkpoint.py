from __future__ import annotations

from pathlib import Path

import pytest
import torch

from eco_planner.rl import (
    ExplorationPolicy,
    ExplorationPolicyConfig,
    load_exploration_policy_checkpoint,
    save_exploration_policy_checkpoint,
)


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
