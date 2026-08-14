"""Strict CPU-resident transition types for PPO-only closed-loop collection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch

from eco_planner.rl.policy import ExplorationPolicyContext

_TailKind = Literal["terminated", "truncated", "rollout_limit"]
_REWARD_SOURCE = "metadrive_builtin_v1"
_REWARD_UNIT = "dimensionless_score"


@dataclass(frozen=True)
class MetaDriveRolloutReward:
    """The unmodified MetaDrive reward transported for Stage 4, not a parity reward."""

    substep_scores: torch.Tensor
    total_score: torch.Tensor
    source: Literal["metadrive_builtin_v1"] = _REWARD_SOURCE
    unit: Literal["dimensionless_score"] = _REWARD_UNIT

    def __post_init__(self) -> None:
        _cpu_float(self.substep_scores, "reward substep scores", ndim=1)
        _cpu_float(self.total_score, "reward total score", shape=(1,))
        if self.substep_scores.shape != (1,):
            raise ValueError("Stage-4 reward must contain exactly one executed substep score")
        if not torch.equal(self.total_score, self.substep_scores.sum().reshape(1)):
            raise ValueError("reward total score must equal the executed substep score sum")


@dataclass(frozen=True)
class RolloutTransition:
    """All data required to audit one 10 Hz policy action without a denoise chain."""

    policy_context: ExplorationPolicyContext
    base_action: torch.Tensor
    guidance_action: torch.Tensor
    old_joint_guidance_log_prob: torch.Tensor
    old_value: torch.Tensor
    initial_noise: torch.Tensor
    diffusion_rng_state: torch.Tensor
    policy_rng_state: torch.Tensor
    reward: MetaDriveRolloutReward
    terminated: bool
    truncated: bool
    bootstrap_mask: bool
    scenario_name: str
    map_sequence: str
    map_seed: int
    noise_seed: int
    policy_action_seed: int
    planning_cycle_index: int
    executed_substep_count: int

    def __post_init__(self) -> None:
        _validate_context(self.policy_context)
        _cpu_float(self.base_action, "base action", shape=(1, 2))
        _cpu_float(self.guidance_action, "guidance action", shape=(1, 2))
        if torch.any((self.base_action <= 0.0) | (self.base_action >= 1.0)):
            raise ValueError("base action must be strictly inside (0, 1)")
        if torch.any((self.guidance_action <= -1.0) | (self.guidance_action >= 1.0)):
            raise ValueError("guidance action must be strictly inside (-1, 1)")
        torch.testing.assert_close(self.guidance_action, 2.0 * self.base_action - 1.0)
        _cpu_float(self.old_joint_guidance_log_prob, "old guidance log-prob", shape=(1,))
        _cpu_float(self.old_value, "old value", shape=(1,))
        _cpu_float(self.initial_noise, "initial noise", shape=(1, 11, 80, 4))
        _rng_state(self.diffusion_rng_state, "diffusion RNG state")
        _rng_state(self.policy_rng_state, "policy RNG state")
        if type(self.terminated) is not bool or type(self.truncated) is not bool:
            raise TypeError("terminated and truncated must be bool")
        if type(self.bootstrap_mask) is not bool:
            raise TypeError("bootstrap_mask must be bool")
        if self.bootstrap_mask is not (not self.terminated):
            raise ValueError("bootstrap_mask must equal not terminated")
        if not isinstance(self.scenario_name, str) or not self.scenario_name:
            raise ValueError("scenario_name must be non-empty")
        if not isinstance(self.map_sequence, str) or not self.map_sequence:
            raise ValueError("map_sequence must be non-empty")
        for name, value in (
            ("map_seed", self.map_seed),
            ("noise_seed", self.noise_seed),
            ("policy_action_seed", self.policy_action_seed),
            ("planning_cycle_index", self.planning_cycle_index),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.executed_substep_count != 1:
            raise ValueError("Stage-4 rollout transitions must execute exactly one substep")


@dataclass(frozen=True)
class RolloutEpisode:
    """One non-leaking sequence with an explicit terminal, truncation, or collection tail."""

    transitions: tuple[RolloutTransition, ...]
    tail_kind: _TailKind
    tail_bootstrap_value: torch.Tensor

    def __post_init__(self) -> None:
        if not self.transitions:
            raise ValueError("rollout episode must contain at least one transition")
        _cpu_float(self.tail_bootstrap_value, "tail bootstrap value", shape=(1,))
        if self.tail_kind not in {"terminated", "truncated", "rollout_limit"}:
            raise ValueError("rollout episode has an invalid tail kind")
        for index, transition in enumerate(self.transitions):
            if transition.planning_cycle_index != index:
                raise ValueError("planning_cycle_index must be contiguous from zero")
            previous = self.transitions[index - 1] if index else None
            if previous is not None and (previous.terminated or previous.truncated):
                raise ValueError("rollout episode cannot continue after an episode boundary")
        final = self.transitions[-1]
        if self.tail_kind == "terminated":
            if not final.terminated or final.bootstrap_mask:
                raise ValueError("terminated tail must not bootstrap")
            if not torch.equal(
                self.tail_bootstrap_value, torch.zeros_like(self.tail_bootstrap_value)
            ):
                raise ValueError("terminated tail bootstrap value must be zero")
        elif self.tail_kind == "truncated":
            if final.terminated or not final.truncated or not final.bootstrap_mask:
                raise ValueError("truncated tail must bootstrap unless also terminated")
        elif final.terminated or final.truncated or not final.bootstrap_mask:
            raise ValueError("rollout_limit tail must end before a done boundary and bootstrap")


@dataclass
class RolloutBuffer:
    """Append-only validator used by a single Stage-4 collector episode."""

    _transitions: list[RolloutTransition] = field(default_factory=list)

    def append(self, transition: RolloutTransition) -> None:
        if not isinstance(transition, RolloutTransition):
            raise TypeError("rollout buffer accepts RolloutTransition values")
        expected_index = len(self._transitions)
        if transition.planning_cycle_index != expected_index:
            raise ValueError("rollout transition planning_cycle_index is not contiguous")
        if self._transitions and (
            self._transitions[-1].terminated or self._transitions[-1].truncated
        ):
            raise ValueError("rollout buffer cannot append after an episode boundary")
        self._transitions.append(transition)

    def finalize(self, tail_kind: _TailKind, tail_bootstrap_value: torch.Tensor) -> RolloutEpisode:
        return RolloutEpisode(tuple(self._transitions), tail_kind, tail_bootstrap_value)


def _validate_context(context: ExplorationPolicyContext) -> None:
    if not isinstance(context, ExplorationPolicyContext):
        raise TypeError("policy_context must be ExplorationPolicyContext")
    _cpu_float(context.scene_tokens, "scene tokens", ndim=3)
    _cpu_bool(context.scene_padding_mask, "scene padding mask", ndim=2)
    _cpu_float(context.navigation_tokens, "navigation tokens", ndim=3)
    _cpu_bool(context.navigation_padding_mask, "navigation padding mask", ndim=2)
    _cpu_float(context.reference_trajectory, "reference trajectory", shape=(1, 80, 4))
    if context.scene_tokens.shape[0] != 1 or context.navigation_tokens.shape[0] != 1:
        raise ValueError("rollout policy context must have batch size one")
    if context.scene_padding_mask.shape != context.scene_tokens.shape[:2]:
        raise ValueError("scene padding mask shape disagrees with scene tokens")
    if context.navigation_padding_mask.shape != context.navigation_tokens.shape[:2]:
        raise ValueError("navigation padding mask shape disagrees with navigation tokens")
    if context.scene_tokens.dtype != context.navigation_tokens.dtype:
        raise TypeError("policy context floating tensors must share dtype")


def _cpu_float(
    value: torch.Tensor,
    name: str,
    *,
    shape: tuple[int, ...] | None = None,
    ndim: int | None = None,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.device.type != "cpu" or value.dtype != torch.float32:
        raise TypeError(f"{name} must be a CPU float32 tensor")
    if shape is not None and tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")


def _cpu_bool(value: torch.Tensor, name: str, *, ndim: int) -> None:
    if (
        not isinstance(value, torch.Tensor)
        or value.device.type != "cpu"
        or value.dtype != torch.bool
    ):
        raise TypeError(f"{name} must be a CPU bool tensor")
    if value.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")


def _rng_state(value: torch.Tensor, name: str) -> None:
    if (
        not isinstance(value, torch.Tensor)
        or value.device.type != "cpu"
        or value.dtype != torch.uint8
    ):
        raise TypeError(f"{name} must be a CPU uint8 tensor")
    if value.ndim != 1 or value.numel() == 0:
        raise ValueError(f"{name} must be non-empty and one-dimensional")
