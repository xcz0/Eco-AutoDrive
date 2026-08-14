"""Strict policy-only checkpoint persistence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from eco_planner.rl.policy import ExplorationPolicy

_FORMAT_VERSION = 1


@dataclass(frozen=True)
class PolicyCheckpointReport:
    format_version: int
    tensor_count: int
    parameter_count: int


def save_exploration_policy_checkpoint(
    path: Path, policy: ExplorationPolicy
) -> PolicyCheckpointReport:
    """Save only the trainable Exploration Policy state dict."""

    _validate_checkpoint_target(path, policy)
    state_dict = {name: value.detach().cpu().clone() for name, value in policy.state_dict().items()}
    torch.save({"format_version": _FORMAT_VERSION, "policy_state_dict": state_dict}, path)
    return _report(state_dict)


def load_exploration_policy_checkpoint(
    path: Path, policy: ExplorationPolicy
) -> PolicyCheckpointReport:
    """Strictly load a policy-only checkpoint into the supplied architecture."""

    if not isinstance(path, Path):
        raise TypeError("policy checkpoint path must be pathlib.Path")
    _validate_policy_state(policy)
    checkpoint: Any = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or set(checkpoint) != {
        "format_version",
        "policy_state_dict",
    }:
        raise ValueError(
            "policy checkpoint must contain exactly format_version and policy_state_dict"
        )
    if checkpoint["format_version"] != _FORMAT_VERSION:
        raise ValueError(f"policy checkpoint format_version must equal {_FORMAT_VERSION}")
    state_dict = checkpoint["policy_state_dict"]
    if not isinstance(state_dict, Mapping) or not all(
        isinstance(name, str) and isinstance(value, torch.Tensor)
        for name, value in state_dict.items()
    ):
        raise TypeError("policy_state_dict must map string names to tensors")
    expected = set(policy.state_dict())
    actual = set(state_dict)
    if actual != expected:
        raise ValueError(
            "policy state-dict keys mismatch; "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
        )
    policy.load_state_dict(state_dict, strict=True)
    return _report(state_dict)


def _validate_checkpoint_target(path: Path, policy: ExplorationPolicy) -> None:
    if not isinstance(path, Path):
        raise TypeError("policy checkpoint path must be pathlib.Path")
    _validate_policy_state(policy)


def _validate_policy_state(policy: ExplorationPolicy) -> None:
    if not isinstance(policy, ExplorationPolicy):
        raise TypeError("policy checkpoint requires ExplorationPolicy")
    parameters = dict(policy.named_parameters())
    if not parameters or any(not parameter.requires_grad for parameter in parameters.values()):
        raise ValueError("policy checkpoint requires only trainable policy parameters")
    if set(policy.state_dict()) != set(parameters):
        raise ValueError("policy checkpoint does not support persistent non-parameter buffers")


def _report(state_dict: Mapping[str, torch.Tensor]) -> PolicyCheckpointReport:
    return PolicyCheckpointReport(
        format_version=_FORMAT_VERSION,
        tensor_count=len(state_dict),
        parameter_count=sum(value.numel() for value in state_dict.values()),
    )
