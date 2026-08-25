"""TensorDict contracts for one policy-guided rollout episode."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from tensordict import TensorDict, TensorDictBase

from eco_planner.rl.policy import ExplorationPolicyContext

TailKind = Literal["terminated", "truncated", "rollout_limit"]
_CPU_DEVICE = torch.device("cpu")
_CONTEXT_KEYS = (
    "scene_tokens",
    "scene_padding_mask",
    "navigation_tokens",
    "navigation_padding_mask",
    "reference_trajectory",
)
_TRAINING_KEYS = frozenset(
    {
        *_CONTEXT_KEYS,
        "guidance_action",
        "old_joint_guidance_log_prob",
        "state_value",
        "next",
    }
)
_NEXT_TRAINING_KEYS = frozenset(
    {
        "state_value",
        "reward",
        "done",
        "terminated",
        "truncated",
    }
)
_AUDIT_KEYS = frozenset(
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
    """PPO and CPU audit TensorDicts with a validated GAE boundary."""

    training: TensorDictBase
    audit: TensorDictBase
    tail_kind: TailKind
    tail_bootstrap_value: torch.Tensor

    def __post_init__(self) -> None:
        _validate_training_trajectory(self.training)
        _validate_audit_trajectory(self.audit)
        if self.training.batch_size != self.audit.batch_size:
            raise ValueError("PPO training and audit trajectories must have matching batch sizes")
        _validate_tail(self.training, self.tail_kind, self.tail_bootstrap_value)

    @property
    def transition_count(self) -> int:
        return self.training.batch_size[0]


def build_training_decision(
    policy_context: ExplorationPolicyContext,
    guidance_action: torch.Tensor,
    old_joint_guidance_log_prob: torch.Tensor,
    state_value: torch.Tensor,
) -> TensorDictBase:
    """Detach compact PPO inputs on their collection device."""

    values = {
        "scene_tokens": policy_context.scene_tokens,
        "scene_padding_mask": policy_context.scene_padding_mask,
        "navigation_tokens": policy_context.navigation_tokens,
        "navigation_padding_mask": policy_context.navigation_padding_mask,
        "reference_trajectory": policy_context.reference_trajectory,
        "guidance_action": guidance_action,
        "old_joint_guidance_log_prob": old_joint_guidance_log_prob.reshape(-1, 1),
        "state_value": state_value.reshape(-1, 1),
    }
    batch = policy_context.reference_trajectory.shape[0]
    return TensorDict(
        {key: value.detach().clone() for key, value in values.items()}, batch_size=[batch]
    )


def build_training_transition(
    decision: TensorDictBase, *, reward: float, terminated: bool, truncated: bool
) -> TensorDictBase:
    """Create one root/next PPO transition from a detached policy decision."""

    device = _tensordict_device(decision)
    return TensorDict(
        {
            **{key: decision[key] for key in decision.keys()},
            "next": TensorDict(
                {
                    "reward": _float(reward, device),
                    "done": _bool(terminated or truncated, device),
                    "terminated": _bool(terminated, device),
                    "truncated": _bool(truncated, device),
                },
                batch_size=[1],
            ),
        },
        batch_size=[1],
    )


def set_training_transition_next_state_value(
    transition: TensorDictBase, next_state_value: torch.Tensor
) -> None:
    """Attach the frozen critic value already computed for a transition's next state."""

    if transition.batch_size != torch.Size([1]):
        raise ValueError("PPO training transition must have batch size [1]")
    device = _tensordict_device(transition)
    value = next_state_value.detach().clone().reshape(-1, 1)
    if value.device != device or tuple(value.shape) != (1, 1):
        raise ValueError("next PPO state value must match the transition device with shape [1, 1]")
    transition["next", "state_value"] = value


def build_rollout_audit(
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
) -> TensorDictBase:
    """Build one CPU audit transition; validate it when the episode closes."""

    return TensorDict(
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


def finalize_rollout_episode(
    training_transitions: list[TensorDictBase],
    audit_transitions: list[TensorDictBase],
    tail_kind: TailKind,
    tail_bootstrap_value: torch.Tensor,
) -> RolloutEpisode:
    """Join PPO and CPU audit data after attaching the final GAE boundary value."""

    if not training_transitions or not audit_transitions:
        raise ValueError("rollout episode must contain at least one transition")
    final_transition = training_transitions[-1]
    device = _tensordict_device(final_transition)
    bootstrap = tail_bootstrap_value.detach().to(device)
    set_training_transition_next_state_value(final_transition, bootstrap)
    final_transition["next", "done"] = _bool(True, device)
    training = torch.cat(training_transitions, dim=0)
    return RolloutEpisode(training, torch.cat(audit_transitions, dim=0), tail_kind, bootstrap)


def _validate_training_trajectory(trajectory: TensorDictBase) -> None:
    _validate_trajectory(trajectory, _TRAINING_KEYS, "PPO training")
    _validate_policy_context(trajectory, "PPO training")
    if tuple(trajectory["guidance_action"].shape[1:]) != (2,):
        raise ValueError("PPO training guidance_action must have shape [T, 2]")
    if torch.any((trajectory["guidance_action"] <= -1.0) | (trajectory["guidance_action"] >= 1.0)):
        raise ValueError("PPO training guidance_action must be strictly inside (-1, 1)")
    for key in ("old_joint_guidance_log_prob", "state_value", "reward"):
        value = trajectory["next", key] if key == "reward" else trajectory[key]
        if tuple(value.shape[1:]) != (1,):
            raise ValueError(f"PPO training {key} must have shape [T, 1]")
    next_transition = trajectory["next"]
    missing = _NEXT_TRAINING_KEYS - set(next_transition.keys(include_nested=False))
    if missing:
        raise ValueError(f"PPO training next transition is missing fields: {sorted(missing)}")
    for key in ("done", "terminated", "truncated"):
        value = next_transition[key]
        if value.dtype != torch.bool or tuple(value.shape[1:]) != (1,):
            raise TypeError(f"PPO training next {key} must be bool with shape [T, 1]")
    if torch.any(next_transition["terminated"] & next_transition["truncated"]):
        raise ValueError("PPO training transition cannot be both terminated and truncated")


def _validate_audit_trajectory(trajectory: TensorDictBase) -> None:
    _validate_trajectory(trajectory, _AUDIT_KEYS, "rollout audit")
    if _tensordict_device(trajectory).type != "cpu":
        raise TypeError("rollout audit fields must be CPU tensors")
    _validate_policy_context(trajectory, "rollout audit")
    for key in ("base_action", "guidance_action", "beta_alpha", "beta_beta"):
        if tuple(trajectory[key].shape[1:]) != (2,):
            raise ValueError(f"rollout audit {key} must have shape [T, 2]")
    if torch.any((trajectory["base_action"] <= 0.0) | (trajectory["base_action"] >= 1.0)):
        raise ValueError("rollout audit base_action must be strictly inside (0, 1)")
    if torch.any((trajectory["guidance_action"] <= -1.0) | (trajectory["guidance_action"] >= 1.0)):
        raise ValueError("rollout audit guidance_action must be strictly inside (-1, 1)")
    if torch.any(trajectory["beta_alpha"] <= 0.0) or torch.any(trajectory["beta_beta"] <= 0.0):
        raise ValueError("rollout audit Beta parameters must be strictly positive")
    if tuple(trajectory["initial_noise"].shape[1:]) != (11, 80, 4):
        raise ValueError("rollout audit initial_noise must have shape [T, 11, 80, 4]")
    for key in ("diffusion_rng_state", "policy_rng_state"):
        if trajectory[key].dtype != torch.uint8 or trajectory[key].ndim != 2:
            raise TypeError(f"{key} must have shape [T, state_length] and uint8 dtype")
    if not torch.allclose(
        trajectory["reward"],
        trajectory["dense_reward"] + trajectory["terminal_override"],
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError("rollout audit reward must equal dense_reward plus terminal_override")


def _validate_trajectory(
    trajectory: TensorDictBase, required_keys: frozenset[str], contract: str
) -> None:
    if not isinstance(trajectory, TensorDictBase) or len(trajectory.batch_size) != 1:
        raise TypeError(f"{contract} trajectory must be a one-dimensional TensorDict")
    if trajectory.batch_size[0] <= 0:
        raise ValueError(f"{contract} trajectory must contain at least one transition")
    missing = required_keys - set(trajectory.keys(include_nested=False))
    if missing:
        raise ValueError(f"{contract} trajectory is missing fields: {sorted(missing)}")
    device: torch.device | None = None
    for key, value in trajectory.items(include_nested=True, leaves_only=True):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{contract} field {key!r} must be a tensor")
        if device is None:
            device = value.device
        elif value.device != device:
            raise TypeError(f"{contract} fields must use one device")
        if value.dtype.is_floating_point and not torch.isfinite(value).all():
            raise ValueError(f"{contract} field {key!r} must be finite")


def _validate_policy_context(trajectory: TensorDictBase, contract: str) -> None:
    context = ExplorationPolicyContext(**{key: trajectory[key] for key in _CONTEXT_KEYS})
    if context.scene_tokens.shape[0] != trajectory.batch_size[0]:
        raise ValueError(f"{contract} policy context batch size must match trajectory")


def _validate_tail(trajectory: TensorDictBase, tail_kind: TailKind, value: torch.Tensor) -> None:
    if tail_kind not in {"terminated", "truncated", "rollout_limit"}:
        raise ValueError("rollout episode has an invalid tail kind")
    if (
        value.dtype != torch.float32
        or tuple(value.shape) != (1,)
        or value.requires_grad
        or not torch.isfinite(value).all()
    ):
        raise ValueError("tail bootstrap value must be detached float32 with shape [1]")
    next_transition = trajectory["next"]
    if not bool(next_transition["done"][-1].item()) or torch.any(next_transition["done"][:-1]):
        raise ValueError("PPO training next done must mark only the final GAE boundary")
    terminated = bool(next_transition["terminated"][-1].item())
    truncated = bool(next_transition["truncated"][-1].item())
    if tail_kind == "terminated":
        if not terminated or truncated or not torch.equal(value, torch.zeros_like(value)):
            raise ValueError("terminated tail must have a zero bootstrap value")
    elif tail_kind == "truncated":
        if terminated or not truncated:
            raise ValueError("truncated tail must end in a non-terminal truncation")
    elif terminated or truncated:
        raise ValueError("rollout_limit tail must end before an episode boundary")


def _float(value: float, device: torch.device = _CPU_DEVICE) -> torch.Tensor:
    return torch.tensor([[value]], dtype=torch.float32, device=device)


def _bool(value: bool, device: torch.device = _CPU_DEVICE) -> torch.Tensor:
    return torch.tensor([[value]], dtype=torch.bool, device=device)


def _integer(value: int) -> torch.Tensor:
    return torch.tensor([[value]], dtype=torch.int64)


def _tensordict_device(trajectory: TensorDictBase) -> torch.device:
    for value in trajectory.values(include_nested=True, leaves_only=True):
        if isinstance(value, torch.Tensor):
            return value.device
    raise ValueError("TensorDict must contain at least one tensor")
