"""Separate PPO training and audit/replay trajectories for one rollout."""

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
        "reward",
        "terminated",
        "truncated",
    }
)
_TRAINING_DECISION_KEYS = frozenset(
    {
        *_CONTEXT_KEYS,
        "guidance_action",
        "old_joint_guidance_log_prob",
        "state_value",
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
class PPOTrainingTrajectory:
    """One-device fields required by GAE and PPO, excluding audit-only rollout data."""

    data: TensorDictBase

    def __post_init__(self) -> None:
        _validate_training_trajectory(self.data)

    @property
    def transition_count(self) -> int:
        return self.data.batch_size[0]

    def policy_context_at(self, index: int) -> ExplorationPolicyContext:
        if type(index) is not int or not 0 <= index < self.transition_count:
            raise IndexError("policy context index is outside the PPO training trajectory")
        item = self.data[index]
        return ExplorationPolicyContext(**{key: item[key].unsqueeze(0) for key in _CONTEXT_KEYS})

    def with_gae(self, advantage: torch.Tensor, value_target: torch.Tensor) -> TensorDictBase:
        expected = (self.transition_count, 1)
        for name, value in (("advantage", advantage), ("value target", value_target)):
            if (
                not isinstance(value, torch.Tensor)
                or value.device != _tensordict_device(self.data)
                or value.dtype != torch.float32
                or tuple(value.shape) != expected
                or value.requires_grad
                or not torch.isfinite(value).all()
            ):
                raise ValueError(f"GAE {name} must be detached float32 with shape [T, 1]")
        result = self.data.clone()
        result["advantage"] = advantage
        result["value_target"] = value_target
        return result


@dataclass(frozen=True)
class RolloutAuditTrajectory:
    """CPU artifact/replay fields, including reproducibility and execution diagnostics."""

    data: TensorDictBase

    def __post_init__(self) -> None:
        _validate_audit_trajectory(self.data)

    @property
    def transition_count(self) -> int:
        return self.data.batch_size[0]


@dataclass(frozen=True)
class PPOTrainingDecision:
    """Detached policy data retained until one bounded episode reaches PPO."""

    data: TensorDictBase

    def __post_init__(self) -> None:
        if not isinstance(self.data, TensorDictBase) or self.data.batch_size != torch.Size([1]):
            raise TypeError("PPO training decision must be a batch-size-one TensorDict")
        if set(self.data.keys(include_nested=False)) != _TRAINING_DECISION_KEYS:
            raise ValueError("PPO training decision has unexpected fields")
        devices = set()
        for key, value in self.data.items(include_nested=True, leaves_only=True):
            if not isinstance(value, torch.Tensor) or value.requires_grad:
                raise TypeError(f"PPO training decision field {key!r} must be a detached tensor")
            devices.add(value.device)
        if len(devices) != 1:
            raise ValueError("PPO training decision fields must use one device")

    @classmethod
    def from_policy_output(
        cls,
        policy_context: ExplorationPolicyContext,
        guidance_action: torch.Tensor,
        old_joint_guidance_log_prob: torch.Tensor,
        state_value: torch.Tensor,
    ) -> PPOTrainingDecision:
        """Detach and own the compact PPO inputs without an artifact-device conversion."""

        values = {
            "scene_tokens": policy_context.scene_tokens,
            "scene_padding_mask": policy_context.scene_padding_mask,
            "navigation_tokens": policy_context.navigation_tokens,
            "navigation_padding_mask": policy_context.navigation_padding_mask,
            "reference_trajectory": policy_context.reference_trajectory,
            "guidance_action": guidance_action,
            "old_joint_guidance_log_prob": old_joint_guidance_log_prob.reshape(1, 1),
            "state_value": state_value.reshape(1, 1),
        }
        return cls(
            TensorDict(
                {key: value.detach().clone() for key, value in values.items()}, batch_size=[1]
            )
        )


class RolloutTrainingBuffer:
    """Bounded, single-device staging for PPO data collected during one episode."""

    def __init__(self, capacity: int) -> None:
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("rollout training buffer capacity must be a positive integer")
        self._capacity = capacity
        self._transitions: list[TensorDictBase] = []
        self._device: torch.device | None = None
        self._finalized = False

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def transition_count(self) -> int:
        return len(self._transitions)

    def append(
        self, decision: PPOTrainingDecision, *, reward: float, terminated: bool, truncated: bool
    ) -> None:
        """Append exactly one detached PPO transition without crossing devices for features."""

        if self._finalized:
            raise RuntimeError("cannot append to a finalized rollout training buffer")
        if not isinstance(decision, PPOTrainingDecision):
            raise TypeError("rollout training buffer requires a PPOTrainingDecision")
        if type(reward) is not float or not torch.isfinite(torch.tensor(reward)):
            raise TypeError("rollout training reward must be a finite float")
        if type(terminated) is not bool or type(truncated) is not bool:
            raise TypeError("rollout training terminal flags must be bool")
        if terminated and truncated:
            raise ValueError("rollout training transition cannot be both terminated and truncated")
        if self.transition_count >= self.capacity:
            raise RuntimeError("rollout training buffer capacity is exhausted")
        device = _tensordict_device(decision.data)
        if self._device is None:
            self._device = device
        elif device != self._device:
            raise ValueError("rollout training buffer cannot mix devices")
        if self._transitions and (
            self._transitions[-1]["terminated"].item() or self._transitions[-1]["truncated"].item()
        ):
            raise ValueError("rollout training buffer cannot continue after an episode boundary")
        self._transitions.append(
            TensorDict(
                {
                    **{key: decision.data[key] for key in _TRAINING_DECISION_KEYS},
                    "reward": _float(reward, device),
                    "terminated": _bool(terminated, device),
                    "truncated": _bool(truncated, device),
                },
                batch_size=[1],
            )
        )

    def finalize(
        self, tail_kind: TailKind, tail_bootstrap_value: torch.Tensor
    ) -> PPOTrainingTrajectory:
        """Release staged decisions as one PPO trajectory with its GAE boundary."""

        if self._finalized:
            raise RuntimeError("rollout training buffer is already finalized")
        if not self._transitions or self._device is None:
            raise ValueError("rollout training buffer must contain at least one transition")
        self._finalized = True
        trajectory = torch.cat(self._transitions, dim=0)
        self._transitions.clear()
        bootstrap = _validate_and_move_tail(trajectory, tail_kind, tail_bootstrap_value)
        next_values = torch.cat([trajectory["state_value"][1:], bootstrap.reshape(1, 1)], dim=0)
        done = torch.zeros((trajectory.batch_size[0], 1), dtype=torch.bool, device=self._device)
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
        return PPOTrainingTrajectory(trajectory)

    def clear(self) -> None:
        """Release retained decisions after an aborted rollout."""

        self._transitions.clear()
        self._finalized = True


@dataclass(frozen=True)
class RolloutTransition:
    """One aligned PPO and audit transition before episode concatenation."""

    training: PPOTrainingTrajectory
    audit: RolloutAuditTrajectory

    def __post_init__(self) -> None:
        _validate_shared_fields(self.training.data, self.audit.data)

    @property
    def training_trajectory(self) -> TensorDictBase:
        return self.training.data

    @property
    def audit_trajectory(self) -> TensorDictBase:
        return self.audit.data


@dataclass(frozen=True)
class RolloutEpisode:
    """Aligned PPO and audit trajectories with a validated GAE bootstrap boundary."""

    training: PPOTrainingTrajectory
    audit: RolloutAuditTrajectory
    tail_kind: TailKind
    tail_bootstrap_value: torch.Tensor

    def __post_init__(self) -> None:
        _validate_shared_fields(self.training.data, self.audit.data)
        _validate_tail(self.training.data, self.tail_kind, self.tail_bootstrap_value)

    @property
    def transition_count(self) -> int:
        return self.training.transition_count

    @property
    def training_trajectory(self) -> TensorDictBase:
        return self.training.data

    @property
    def audit_trajectory(self) -> TensorDictBase:
        return self.audit.data

    def policy_context_at(self, index: int) -> ExplorationPolicyContext:
        return self.training.policy_context_at(index)

    def with_gae(self, advantage: torch.Tensor, value_target: torch.Tensor) -> TensorDictBase:
        return self.training.with_gae(advantage, value_target)


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
) -> RolloutTransition:
    """Create aligned, batch-size-one PPO and audit/replay transition contracts."""

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
    shared = {
        "scene_tokens": policy_context.scene_tokens,
        "scene_padding_mask": policy_context.scene_padding_mask,
        "navigation_tokens": policy_context.navigation_tokens,
        "navigation_padding_mask": policy_context.navigation_padding_mask,
        "reference_trajectory": policy_context.reference_trajectory,
        "guidance_action": guidance_action,
        "old_joint_guidance_log_prob": old_joint_guidance_log_prob.reshape(1, 1),
        "state_value": state_value.reshape(1, 1),
        "reward": _float(reward),
        "terminated": _bool(terminated),
        "truncated": _bool(truncated),
    }
    training = PPOTrainingTrajectory(TensorDict(shared, batch_size=[1]))
    audit = RolloutAuditTrajectory(
        TensorDict(
            {
                **shared,
                "base_action": base_action,
                "beta_alpha": beta_alpha,
                "beta_beta": beta_beta,
                "initial_noise": initial_noise,
                "diffusion_rng_state": diffusion_rng_state.unsqueeze(0),
                "policy_rng_state": policy_rng_state.unsqueeze(0),
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
                "map_seed": _integer(map_seed),
                "noise_seed": _integer(noise_seed),
                "policy_action_seed": _integer(policy_action_seed),
                "planning_cycle_index": _integer(planning_cycle_index),
            },
            batch_size=[1],
        )
    )
    return RolloutTransition(training, audit)


def finalize_rollout_episode(
    transitions: list[RolloutTransition], tail_kind: TailKind, tail_bootstrap_value: torch.Tensor
) -> RolloutEpisode:
    """Concatenate aligned transitions and attach GAE ``next`` fields to PPO data only."""

    if not transitions:
        raise ValueError("rollout episode must contain at least one transition")
    for index, transition in enumerate(transitions):
        if not isinstance(transition, RolloutTransition):
            raise TypeError("rollout episode transitions must be RolloutTransition instances")
        if transition.audit_trajectory["planning_cycle_index"].item() != index:
            raise ValueError("planning_cycle_index must be contiguous from zero")
        if index and (
            transitions[index - 1].training_trajectory["terminated"].item()
            or transitions[index - 1].training_trajectory["truncated"].item()
        ):
            raise ValueError("rollout episode cannot continue after an episode boundary")
    training_trajectory = torch.cat([item.training_trajectory for item in transitions], dim=0)
    next_values = torch.cat(
        [training_trajectory["state_value"][1:], tail_bootstrap_value.reshape(1, 1)], dim=0
    )
    done = torch.zeros((training_trajectory.batch_size[0], 1), dtype=torch.bool)
    done[-1] = True
    training_trajectory["next"] = TensorDict(
        {
            "state_value": next_values,
            "reward": training_trajectory["reward"].clone(),
            "done": done,
            "terminated": training_trajectory["terminated"].clone(),
        },
        batch_size=training_trajectory.batch_size,
    )
    return RolloutEpisode(
        PPOTrainingTrajectory(training_trajectory),
        RolloutAuditTrajectory(torch.cat([item.audit_trajectory for item in transitions], dim=0)),
        tail_kind,
        tail_bootstrap_value,
    )


def finalize_buffered_rollout_episode(
    training_buffer: RolloutTrainingBuffer,
    audit_transitions: list[RolloutAuditTrajectory],
    tail_kind: TailKind,
    tail_bootstrap_value: torch.Tensor,
) -> RolloutEpisode:
    """Join a device-resident PPO buffer to its independent CPU audit trajectory."""

    if not isinstance(training_buffer, RolloutTrainingBuffer):
        raise TypeError("buffered rollout requires a RolloutTrainingBuffer")
    if not audit_transitions:
        raise ValueError("buffered rollout must contain at least one audit transition")
    for index, audit in enumerate(audit_transitions):
        if not isinstance(audit, RolloutAuditTrajectory):
            raise TypeError("buffered rollout audits must be RolloutAuditTrajectory instances")
        if audit.data["planning_cycle_index"].item() != index:
            raise ValueError("planning_cycle_index must be contiguous from zero")
        if index and (
            audit_transitions[index - 1].data["terminated"].item()
            or audit_transitions[index - 1].data["truncated"].item()
        ):
            raise ValueError("buffered rollout cannot continue after an episode boundary")
    training = training_buffer.finalize(tail_kind, tail_bootstrap_value)
    audit = RolloutAuditTrajectory(torch.cat([item.data for item in audit_transitions], dim=0))
    if training.transition_count != audit.transition_count:
        raise ValueError("PPO training buffer and audit trajectory lengths disagree")
    return RolloutEpisode(
        training,
        audit,
        tail_kind,
        tail_bootstrap_value.detach().to(_tensordict_device(training.data)),
    )


def _validate_training_trajectory(trajectory: TensorDictBase) -> None:
    _validate_common_trajectory(trajectory, _TRAINING_KEYS, "PPO training")
    _validate_policy_context(trajectory, "PPO training")
    if tuple(trajectory["guidance_action"].shape[1:]) != (2,):
        raise ValueError("PPO training guidance_action must have shape [T, 2]")
    if torch.any((trajectory["guidance_action"] <= -1.0) | (trajectory["guidance_action"] >= 1.0)):
        raise ValueError("PPO training guidance_action must be strictly inside (-1, 1)")
    for key in ("old_joint_guidance_log_prob", "state_value", "reward"):
        if tuple(trajectory[key].shape[1:]) != (1,):
            raise ValueError(f"PPO training {key} must have shape [T, 1]")
    for key in ("terminated", "truncated"):
        if trajectory[key].dtype != torch.bool or tuple(trajectory[key].shape[1:]) != (1,):
            raise TypeError(f"PPO training {key} must be bool with shape [T, 1]")
    if torch.any(trajectory["terminated"] & trajectory["truncated"]):
        raise ValueError("PPO training transition cannot be both terminated and truncated")


def _validate_audit_trajectory(trajectory: TensorDictBase) -> None:
    _validate_common_trajectory(trajectory, _AUDIT_KEYS, "rollout audit")
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
    torch.testing.assert_close(trajectory["guidance_action"], 2.0 * trajectory["base_action"] - 1.0)
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


def _validate_common_trajectory(
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


def _validate_shared_fields(training: TensorDictBase, audit: TensorDictBase) -> None:
    if training.batch_size != audit.batch_size:
        raise ValueError(
            "PPO training and rollout audit trajectories must have matching batch sizes"
        )
    for key in _TRAINING_KEYS:
        if training[key].shape != audit[key].shape or training[key].dtype != audit[key].dtype:
            raise ValueError(f"PPO training and rollout audit fields disagree for {key!r}")
        if training[key].device == audit[key].device and not torch.equal(training[key], audit[key]):
            raise ValueError(f"PPO training and rollout audit fields disagree for {key!r}")


def _validate_tail(trajectory: TensorDictBase, tail_kind: TailKind, value: torch.Tensor) -> None:
    if tail_kind not in {"terminated", "truncated", "rollout_limit"}:
        raise ValueError("rollout episode has an invalid tail kind")
    if (
        not isinstance(value, torch.Tensor)
        or value.device != _tensordict_device(trajectory)
        or value.dtype != torch.float32
        or tuple(value.shape) != (1,)
        or value.requires_grad
        or not torch.isfinite(value).all()
    ):
        raise ValueError("tail bootstrap value must be detached float32 with shape [1]")
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


def _validate_and_move_tail(
    trajectory: TensorDictBase, tail_kind: TailKind, value: torch.Tensor
) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.float32
        or tuple(value.shape) != (1,)
    ):
        raise ValueError("tail bootstrap value must be detached float32 with shape [1]")
    result = value.detach().to(_tensordict_device(trajectory))
    _validate_tail(trajectory, tail_kind, result)
    return result


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
