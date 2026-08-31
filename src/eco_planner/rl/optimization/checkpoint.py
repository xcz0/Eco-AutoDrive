"""Strict policy-only checkpoint persistence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from lightning.fabric import Fabric

from eco_planner.rl.optimization.ppo import PPOUpdater
from eco_planner.rl.policy import ExplorationPolicy

_FORMAT_VERSION = 1


@dataclass(frozen=True)
class PolicyCheckpointReport:
    format_version: int
    tensor_count: int
    parameter_count: int


@dataclass(frozen=True)
class TrainingCheckpointReport:
    completed_updates: int
    completed_optimizer_steps: int


def save_exploration_policy_checkpoint(
    path: Path, policy: ExplorationPolicy
) -> PolicyCheckpointReport:
    """Save only the trainable Exploration Policy state dict."""

    _validate_policy_state(policy)
    state_dict = {name: value.detach().cpu().clone() for name, value in policy.state_dict().items()}
    torch.save({"format_version": _FORMAT_VERSION, "policy_state_dict": state_dict}, path)
    return _report(state_dict)


def load_exploration_policy_checkpoint(
    path: Path, policy: ExplorationPolicy
) -> PolicyCheckpointReport:
    """Strictly load a policy-only checkpoint into the supplied architecture."""

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


def save_training_checkpoint(
    path: Path,
    fabric: Fabric,
    policy: ExplorationPolicy,
    updater: PPOUpdater,
    loop_state: Mapping[str, object],
) -> TrainingCheckpointReport:
    """Save Fabric-managed model, optimizer, scheduler, loop, and RNG state."""

    completed_updates = loop_state.get("completed_updates")
    if type(completed_updates) is not int or completed_updates < 0:
        raise ValueError("loop_state.completed_updates must be a non-negative integer")
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "policy": policy,
        "optimizer": updater.optimizer,
        "scheduler": updater.scheduler,
        "trainer": {
            "loop_state": dict(loop_state),
            "ppo": updater.checkpoint_state(),
            "cpu_rng_state": torch.random.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else (),
        },
    }
    fabric.save(path, state)
    return TrainingCheckpointReport(completed_updates, updater.completed_optimizer_steps)


def load_training_checkpoint(
    path: Path,
    fabric: Fabric,
    policy: ExplorationPolicy,
    updater: PPOUpdater,
) -> tuple[TrainingCheckpointReport, dict[str, object]]:
    """Restore a state created by :func:`save_training_checkpoint`."""

    if not path.is_file():
        raise ValueError("training checkpoint path must name an existing file")
    trainer_state: dict[str, object] = {
        "loop_state": {},
        "ppo": updater.checkpoint_state(),
        "cpu_rng_state": torch.random.get_rng_state(),
        "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else (),
    }
    state = {
        "policy": policy,
        "optimizer": updater.optimizer,
        "scheduler": updater.scheduler,
        "trainer": trainer_state,
    }
    fabric.load(path, state)
    loaded_trainer_state = state["trainer"]
    if not isinstance(loaded_trainer_state, dict):
        raise TypeError("training checkpoint has an invalid trainer state")
    loop_state = loaded_trainer_state["loop_state"]
    ppo_state = loaded_trainer_state["ppo"]
    cpu_rng_state = loaded_trainer_state["cpu_rng_state"]
    cuda_rng_states = loaded_trainer_state["cuda_rng_states"]
    if not isinstance(loop_state, dict):
        raise TypeError("training checkpoint has an invalid loop state")
    completed_updates = loop_state.get("completed_updates")
    if type(completed_updates) is not int or completed_updates < 0:
        raise ValueError("training checkpoint has an invalid completed update count")
    if not isinstance(ppo_state, Mapping):
        raise TypeError("training checkpoint has an invalid PPO state")
    if not isinstance(cpu_rng_state, torch.Tensor) or cpu_rng_state.dtype != torch.uint8:
        raise TypeError("training checkpoint has an invalid CPU RNG state")
    if not isinstance(cuda_rng_states, (list, tuple)) or not all(
        isinstance(item, torch.Tensor) and item.dtype == torch.uint8 for item in cuda_rng_states
    ):
        raise TypeError("training checkpoint has invalid CUDA RNG states")
    updater.restore_checkpoint_state(ppo_state)
    torch.random.set_rng_state(cpu_rng_state)
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(list(cuda_rng_states))
    return TrainingCheckpointReport(
        completed_updates, updater.completed_optimizer_steps
    ), loop_state


def _validate_policy_state(policy: ExplorationPolicy) -> None:
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
