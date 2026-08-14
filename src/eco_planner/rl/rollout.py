"""TensorDict-first closed-loop trajectories for PPO collection and artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from tensordict import TensorDict, TensorDictBase

from eco_planner.rl.policy import ExplorationPolicyContext

TailKind = Literal["terminated", "truncated", "rollout_limit"]

_CONTEXT_KEYS = (
    "scene_tokens",
    "scene_padding_mask",
    "navigation_tokens",
    "navigation_padding_mask",
    "reference_trajectory",
)
_REQUIRED_KEYS = frozenset(
    {
        *_CONTEXT_KEYS,
        "base_action",
        "guidance_action",
        "old_joint_guidance_log_prob",
        "state_value",
        "beta_alpha",
        "beta_beta",
        "initial_noise",
        "diffusion_rng_state",
        "policy_rng_state",
        "reward",
        "dense_reward",
        "terminal_override",
        "route_completion_delta",
        "distance_m",
        "speed_mps",
        "stopped",
        "position_error_m",
        "heading_error_rad",
        "arrive_dest",
        "out_of_road",
        "crash_vehicle",
        "crash_object",
        "crash_building",
        "crash_human",
        "terminated",
        "truncated",
        "map_seed",
        "noise_seed",
        "policy_action_seed",
        "planning_cycle_index",
    }
)


@dataclass(frozen=True)
class RolloutEpisode:
    """One CPU trajectory with a validated PPO bootstrap boundary."""

    trajectory: TensorDictBase
    tail_kind: TailKind
    tail_bootstrap_value: torch.Tensor

    def __post_init__(self) -> None:
        _validate_trajectory(self.trajectory)
        _validate_tail(self.trajectory, self.tail_kind, self.tail_bootstrap_value)

    @property
    def transition_count(self) -> int:
        return self.trajectory.batch_size[0]

    def policy_context_at(self, index: int) -> ExplorationPolicyContext:
        if type(index) is not int or not 0 <= index < self.transition_count:
            raise IndexError("policy context index is outside the rollout trajectory")
        item = self.trajectory[index]
        return ExplorationPolicyContext(**{key: item[key].unsqueeze(0) for key in _CONTEXT_KEYS})

    def with_gae(self, advantage: torch.Tensor, value_target: torch.Tensor) -> TensorDictBase:
        expected = (self.transition_count, 1)
        for name, value in (("advantage", advantage), ("value target", value_target)):
            if (
                not isinstance(value, torch.Tensor)
                or value.device.type != "cpu"
                or value.dtype != torch.float32
                or tuple(value.shape) != expected
                or value.requires_grad
                or not torch.isfinite(value).all()
            ):
                raise ValueError(f"GAE {name} must be detached CPU float32 with shape [T, 1]")
        result = self.trajectory.clone()
        result["advantage"] = advantage
        result["value_target"] = value_target
        return result


def build_rollout_transition(
    *,
    policy_context: ExplorationPolicyContext,
    base_action: torch.Tensor,
    guidance_action: torch.Tensor,
    old_joint_guidance_log_prob: torch.Tensor,
    state_value: torch.Tensor,
    beta_alpha: torch.Tensor,
    beta_beta: torch.Tensor,
    initial_noise: torch.Tensor,
    diffusion_rng_state: torch.Tensor,
    policy_rng_state: torch.Tensor,
    reward: float,
    dense_reward: float,
    terminal_override: float,
    route_completion_delta: float,
    distance_m: float,
    speed_mps: float,
    stopped: bool,
    position_error_m: float,
    heading_error_rad: float,
    arrive_dest: bool,
    out_of_road: bool,
    crash_vehicle: bool,
    crash_object: bool,
    crash_building: bool,
    crash_human: bool,
    terminated: bool,
    truncated: bool,
    map_seed: int,
    noise_seed: int,
    policy_action_seed: int,
    planning_cycle_index: int,
) -> TensorDict:
    """Create one auditable, batch-size-one rollout transition."""

    if terminated and truncated:
        raise ValueError("rollout transition cannot be both terminated and truncated")
    for name, value in (
        ("reward", reward),
        ("dense_reward", dense_reward),
        ("terminal_override", terminal_override),
        ("route_completion_delta", route_completion_delta),
        ("distance_m", distance_m),
        ("speed_mps", speed_mps),
        ("position_error_m", position_error_m),
        ("heading_error_rad", heading_error_rad),
    ):
        if type(value) is not float or not torch.isfinite(torch.tensor(value)):
            raise TypeError(f"{name} must be a finite float")
    if distance_m < 0.0 or speed_mps < 0.0 or position_error_m < 0.0 or heading_error_rad < 0.0:
        raise ValueError("rollout distances, speeds, and errors must be non-negative")
    for name, value in (
        ("stopped", stopped),
        ("arrive_dest", arrive_dest),
        ("out_of_road", out_of_road),
        ("crash_vehicle", crash_vehicle),
        ("crash_object", crash_object),
        ("crash_building", crash_building),
        ("crash_human", crash_human),
        ("terminated", terminated),
        ("truncated", truncated),
    ):
        if type(value) is not bool:
            raise TypeError(f"{name} must be bool")
    for name, value in (
        ("map_seed", map_seed),
        ("noise_seed", noise_seed),
        ("policy_action_seed", policy_action_seed),
        ("planning_cycle_index", planning_cycle_index),
    ):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    transition = TensorDict(
        {
            "scene_tokens": policy_context.scene_tokens,
            "scene_padding_mask": policy_context.scene_padding_mask,
            "navigation_tokens": policy_context.navigation_tokens,
            "navigation_padding_mask": policy_context.navigation_padding_mask,
            "reference_trajectory": policy_context.reference_trajectory,
            "base_action": base_action,
            "guidance_action": guidance_action,
            "old_joint_guidance_log_prob": old_joint_guidance_log_prob.reshape(1, 1),
            "state_value": state_value.reshape(1, 1),
            "beta_alpha": beta_alpha,
            "beta_beta": beta_beta,
            "initial_noise": initial_noise,
            "diffusion_rng_state": diffusion_rng_state.unsqueeze(0),
            "policy_rng_state": policy_rng_state.unsqueeze(0),
            "reward": _float(reward),
            "dense_reward": _float(dense_reward),
            "terminal_override": _float(terminal_override),
            "route_completion_delta": _float(route_completion_delta),
            "distance_m": _float(distance_m),
            "speed_mps": _float(speed_mps),
            "stopped": _bool(stopped),
            "position_error_m": _float(position_error_m),
            "heading_error_rad": _float(heading_error_rad),
            "arrive_dest": _bool(arrive_dest),
            "out_of_road": _bool(out_of_road),
            "crash_vehicle": _bool(crash_vehicle),
            "crash_object": _bool(crash_object),
            "crash_building": _bool(crash_building),
            "crash_human": _bool(crash_human),
            "terminated": _bool(terminated),
            "truncated": _bool(truncated),
            "map_seed": _integer(map_seed),
            "noise_seed": _integer(noise_seed),
            "policy_action_seed": _integer(policy_action_seed),
            "planning_cycle_index": _integer(planning_cycle_index),
        },
        batch_size=[1],
    )
    _validate_transition(transition)
    return transition


def finalize_rollout_episode(
    transitions: list[TensorDictBase], tail_kind: TailKind, tail_bootstrap_value: torch.Tensor
) -> RolloutEpisode:
    """Concatenate native TensorDict transitions and attach the GAE ``next`` fields."""

    if not transitions:
        raise ValueError("rollout episode must contain at least one transition")
    for index, transition in enumerate(transitions):
        _validate_transition(transition)
        if transition["planning_cycle_index"].item() != index:
            raise ValueError("planning_cycle_index must be contiguous from zero")
        if index and (
            transitions[index - 1]["terminated"].item()
            or transitions[index - 1]["truncated"].item()
        ):
            raise ValueError("rollout episode cannot continue after an episode boundary")
    trajectory = torch.cat(transitions, dim=0)
    next_values = torch.cat(
        [trajectory["state_value"][1:], tail_bootstrap_value.reshape(1, 1)], dim=0
    )
    done = torch.zeros((trajectory.batch_size[0], 1), dtype=torch.bool)
    done[-1] = True
    trajectory["next"] = TensorDict(
        {
            "state_value": next_values,
            "reward": trajectory["reward"].clone(),
            "done": done,
            "terminated": trajectory["terminated"].clone(),
        },
        batch_size=trajectory.batch_size,
    )
    return RolloutEpisode(trajectory, tail_kind, tail_bootstrap_value)


def _validate_trajectory(trajectory: TensorDictBase) -> None:
    if not isinstance(trajectory, TensorDictBase) or len(trajectory.batch_size) != 1:
        raise TypeError("rollout trajectory must be a one-dimensional TensorDict")
    if trajectory.batch_size[0] <= 0:
        raise ValueError("rollout trajectory must contain at least one transition")
    actual = set(trajectory.keys(include_nested=False))
    missing = _REQUIRED_KEYS - actual
    if missing:
        raise ValueError(f"rollout trajectory is missing fields: {sorted(missing)}")
    for key, value in trajectory.items(include_nested=True, leaves_only=True):
        if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
            raise TypeError(f"rollout field {key!r} must be a CPU tensor")
        if value.dtype.is_floating_point and not torch.isfinite(value).all():
            raise ValueError(f"rollout field {key!r} must be finite")
    _validate_transition(trajectory)


def _validate_transition(transition: TensorDictBase) -> None:
    if transition.batch_size != torch.Size([1]) and len(transition.batch_size) != 1:
        raise ValueError("rollout transition must use one leading batch dimension")
    for key in _REQUIRED_KEYS:
        if key not in transition:
            raise ValueError(f"rollout transition is missing {key!r}")
    context = ExplorationPolicyContext(**{key: transition[key] for key in _CONTEXT_KEYS})
    if context.scene_tokens.shape[0] != transition.batch_size[0]:
        raise ValueError("policy context batch size must match rollout transition")
    for key in ("base_action", "guidance_action", "beta_alpha", "beta_beta"):
        if tuple(transition[key].shape[1:]) != (2,):
            raise ValueError(f"rollout {key} must have shape [T, 2]")
    if torch.any((transition["base_action"] <= 0.0) | (transition["base_action"] >= 1.0)):
        raise ValueError("base action must be strictly inside (0, 1)")
    if torch.any((transition["guidance_action"] <= -1.0) | (transition["guidance_action"] >= 1.0)):
        raise ValueError("guidance action must be strictly inside (-1, 1)")
    torch.testing.assert_close(transition["guidance_action"], 2.0 * transition["base_action"] - 1.0)
    if torch.any(transition["beta_alpha"] <= 0.0) or torch.any(transition["beta_beta"] <= 0.0):
        raise ValueError("Beta parameters must be strictly positive")
    if tuple(transition["initial_noise"].shape[1:]) != (11, 80, 4):
        raise ValueError("initial noise must have shape [T, 11, 80, 4]")
    for key in ("diffusion_rng_state", "policy_rng_state"):
        if transition[key].dtype != torch.uint8 or transition[key].ndim != 2:
            raise TypeError(f"{key} must have shape [T, state_length] and uint8 dtype")
    if not torch.allclose(
        transition["reward"],
        transition["dense_reward"] + transition["terminal_override"],
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError("reward must equal dense_reward plus terminal_override")


def _validate_tail(trajectory: TensorDictBase, tail_kind: TailKind, value: torch.Tensor) -> None:
    if tail_kind not in {"terminated", "truncated", "rollout_limit"}:
        raise ValueError("rollout episode has an invalid tail kind")
    if (
        not isinstance(value, torch.Tensor)
        or value.device.type != "cpu"
        or value.dtype != torch.float32
        or tuple(value.shape) != (1,)
        or value.requires_grad
        or not torch.isfinite(value).all()
    ):
        raise ValueError("tail bootstrap value must be detached CPU float32 with shape [1]")
    final = trajectory[-1]
    terminated = bool(final["terminated"].item())
    truncated = bool(final["truncated"].item())
    if tail_kind == "terminated":
        if not terminated or truncated or not torch.equal(value, torch.zeros_like(value)):
            raise ValueError("terminated tail must have a zero bootstrap value")
    elif tail_kind == "truncated":
        if terminated or not truncated:
            raise ValueError("truncated tail must end in a non-terminal truncation")
    elif terminated or truncated:
        raise ValueError("rollout_limit tail must end before an episode boundary")


def _float(value: float) -> torch.Tensor:
    return torch.tensor([[value]], dtype=torch.float32)


def _bool(value: bool) -> torch.Tensor:
    return torch.tensor([[value]], dtype=torch.bool)


def _integer(value: int) -> torch.Tensor:
    return torch.tensor([[value]], dtype=torch.int64)
