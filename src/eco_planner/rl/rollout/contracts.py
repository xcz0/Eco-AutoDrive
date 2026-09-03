"""TensorDict contracts for one policy-guided rollout episode."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
import torch
from tensordict import TensorDict, TensorDictBase

from eco_planner.contracts import PLANNER_ACTOR_COUNT, PLANNER_HORIZON, PLANNER_STATE_DIM
from eco_planner.rl.policy import ExplorationPolicyContext
from eco_planner.rl.reward import (
    PlannerRFTEnergyRewardAudit,
    RewardAudit,
)

TailKind = Literal["terminated", "truncated", "rollout_limit"]
RewardProfileName = Literal["metadrive_builtin_v1", "plannerrft_energy_v1"]
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
_BUILTIN_AUDIT_KEYS: tuple[str, ...] = (
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
    "crash_sidewalk",
    "terminated",
    "truncated",
    "map_seed",
    "noise_seed",
    "policy_action_seed",
    "planning_cycle_index",
    "step_distance_m",
    "native_step_energy_ml",
    "native_episode_energy_ml",
    "executed_fuel_proxy_step_energy_ml",
    "executed_fuel_proxy_ml_per_km",
    "energy_distance_valid",
)
_ENERGY_AUDIT_KEYS: tuple[str, ...] = (
    *_BUILTIN_AUDIT_KEYS,
    "reward_gate",
    "collision_score",
    "drivable_score",
    "wrong_direction_score",
    "ttc_score",
    "progress_score",
    "comfort_score",
    "speed_score",
    "energy_score",
    "has_ttc_candidate",
    "min_ttc_s",
    "route_progress_delta_m",
    "speed_limit_mps",
    "overspeed_mps",
    "longitudinal_acceleration_mps2",
    "lateral_acceleration_mps2",
    "jerk_mps3",
    "yaw_rate_radps",
)


@dataclass(frozen=True)
class DecisionAudit:
    """CPU data retained from one policy-guided planner decision."""

    prediction: np.ndarray
    initial_noise: torch.Tensor
    policy_context: ExplorationPolicyContext
    base_action: torch.Tensor
    guidance_action: torch.Tensor
    old_joint_guidance_log_prob: torch.Tensor
    old_value: torch.Tensor
    beta_alpha: torch.Tensor
    beta_beta: torch.Tensor
    diffusion_rng_state: torch.Tensor
    policy_rng_state: torch.Tensor

    @property
    def ego_trajectory(self) -> np.ndarray:
        return self.prediction[0, 0]


@dataclass(frozen=True)
class ExecutionTransitionAudit:
    """Typed environment result for one 10 Hz rollout transition."""

    reward: float
    dense_reward: float
    terminal_override: float
    route_completion_delta: float
    distance_m: float
    speed_mps: float
    stopped: bool
    position_error_m: float
    heading_error_rad: float
    arrive_dest: bool
    out_of_road: bool
    crash_vehicle: bool
    crash_object: bool
    crash_building: bool
    crash_human: bool
    crash_sidewalk: bool
    terminated: bool
    truncated: bool
    reward_audit: RewardAudit


@dataclass(frozen=True)
class RolloutProvenance:
    """Seed namespaces and episode-local index for one transition."""

    map_seed: int
    noise_seed: int
    policy_action_seed: int
    planning_cycle_index: int


@dataclass(frozen=True)
class RolloutEpisode:
    """PPO and CPU audit TensorDicts with a validated GAE boundary."""

    training: TensorDictBase
    audit: TensorDictBase
    tail_kind: TailKind
    tail_bootstrap_value: torch.Tensor
    reward_profile: RewardProfileName

    def __post_init__(self) -> None:
        _validate_training_trajectory(self.training)
        _validate_audit_trajectory(self.audit, self.reward_profile)
        if self.training.batch_size != self.audit.batch_size:
            raise ValueError("PPO training and audit trajectories must have matching batch sizes")
        _validate_tail(self.training, self.tail_kind, self.tail_bootstrap_value)

    @property
    def transition_count(self) -> int:
        return self.training.batch_size[0]


class RolloutEpisodeBuilder:
    """Build matching PPO and audit trajectories for serial or vector collection."""

    def __init__(self) -> None:
        self._training: list[TensorDictBase] = []
        self._audit: list[TensorDictBase] = []
        self._reward_profile: RewardProfileName | None = None

    @property
    def transition_count(self) -> int:
        return len(self._training)

    @property
    def empty(self) -> bool:
        return not self._training and not self._audit

    def link_next_state_value(self, value: torch.Tensor) -> None:
        set_training_transition_next_state_value(self._training[-1], value)

    def append(
        self,
        training_decision: TensorDictBase,
        decision_audit: DecisionAudit,
        execution: ExecutionTransitionAudit,
        provenance: RolloutProvenance,
    ) -> None:
        profile = execution.reward_audit.profile_name
        if self._reward_profile is None:
            self._reward_profile = profile
        elif profile != self._reward_profile:
            raise ValueError("one rollout episode cannot mix reward profiles")
        self._training.append(
            build_training_transition(
                training_decision,
                reward=execution.reward,
                terminated=execution.terminated,
                truncated=execution.truncated,
            )
        )
        self._audit.append(build_rollout_audit(decision_audit, execution, provenance))

    def finish(self, tail_kind: TailKind, tail_bootstrap_value: torch.Tensor) -> RolloutEpisode:
        return finalize_rollout_episode(
            self._training,
            self._audit,
            tail_kind,
            tail_bootstrap_value,
            cast(RewardProfileName, self._reward_profile),
        )


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
    decision: DecisionAudit,
    execution: ExecutionTransitionAudit,
    provenance: RolloutProvenance,
) -> TensorDictBase:
    """Build one CPU audit transition; validate it when the episode closes."""

    payload = {
        "scene_tokens": decision.policy_context.scene_tokens,
        "scene_padding_mask": decision.policy_context.scene_padding_mask,
        "navigation_tokens": decision.policy_context.navigation_tokens,
        "navigation_padding_mask": decision.policy_context.navigation_padding_mask,
        "reference_trajectory": decision.policy_context.reference_trajectory,
        "base_action": decision.base_action,
        "guidance_action": decision.guidance_action,
        "old_joint_guidance_log_prob": decision.old_joint_guidance_log_prob.reshape(1, 1),
        "state_value": decision.old_value.reshape(1, 1),
        "beta_alpha": decision.beta_alpha,
        "beta_beta": decision.beta_beta,
        "initial_noise": decision.initial_noise,
        "diffusion_rng_state": decision.diffusion_rng_state.unsqueeze(0),
        "policy_rng_state": decision.policy_rng_state.unsqueeze(0),
        "reward": _float(execution.reward),
        "dense_reward": _float(execution.dense_reward),
        "terminal_override": _float(execution.terminal_override),
        "route_completion_delta": _float(execution.route_completion_delta),
        "distance_m": _float(execution.distance_m),
        "speed_mps": _float(execution.speed_mps),
        "stopped": _bool(execution.stopped),
        "position_error_m": _float(execution.position_error_m),
        "heading_error_rad": _float(execution.heading_error_rad),
        "arrive_dest": _bool(execution.arrive_dest),
        "out_of_road": _bool(execution.out_of_road),
        "crash_vehicle": _bool(execution.crash_vehicle),
        "crash_object": _bool(execution.crash_object),
        "crash_building": _bool(execution.crash_building),
        "crash_human": _bool(execution.crash_human),
        "crash_sidewalk": _bool(execution.crash_sidewalk),
        "terminated": _bool(execution.terminated),
        "truncated": _bool(execution.truncated),
        "map_seed": _integer(provenance.map_seed),
        "noise_seed": _integer(provenance.noise_seed),
        "policy_action_seed": _integer(provenance.policy_action_seed),
        "planning_cycle_index": _integer(provenance.planning_cycle_index),
        "step_distance_m": _float(execution.reward_audit.step_distance_m),
        "native_step_energy_ml": _float(execution.reward_audit.native_step_energy_ml),
        "native_episode_energy_ml": _float(execution.reward_audit.native_episode_energy_ml),
        "executed_fuel_proxy_step_energy_ml": _float(
            execution.reward_audit.executed_fuel_proxy_step_energy_ml
        ),
        "executed_fuel_proxy_ml_per_km": _float(
            execution.reward_audit.executed_fuel_proxy_ml_per_km
        ),
        "energy_distance_valid": _bool(execution.reward_audit.energy_distance_valid),
    }
    if execution.reward != execution.reward_audit.reward_total:
        raise ValueError("rollout reward must equal its typed environment reward audit")
    reward_audit = execution.reward_audit
    if isinstance(reward_audit, PlannerRFTEnergyRewardAudit):
        payload.update(
            {
                "reward_gate": _float(reward_audit.reward_gate),
                "collision_score": _float(reward_audit.collision_score),
                "drivable_score": _float(reward_audit.drivable_score),
                "wrong_direction_score": _float(reward_audit.wrong_direction_score),
                "ttc_score": _float(reward_audit.ttc_score),
                "progress_score": _float(reward_audit.progress_score),
                "comfort_score": _float(reward_audit.comfort_score),
                "speed_score": _float(reward_audit.speed_score),
                "energy_score": _float(reward_audit.energy_score),
                "has_ttc_candidate": _bool(reward_audit.has_ttc_candidate),
                "min_ttc_s": _float(reward_audit.min_ttc_s),
                "route_progress_delta_m": _float(reward_audit.route_progress_delta_m),
                "speed_limit_mps": _float(reward_audit.speed_limit_mps),
                "overspeed_mps": _float(reward_audit.overspeed_mps),
                "longitudinal_acceleration_mps2": _float(
                    reward_audit.longitudinal_acceleration_mps2
                ),
                "lateral_acceleration_mps2": _float(reward_audit.lateral_acceleration_mps2),
                "jerk_mps3": _float(reward_audit.jerk_mps3),
                "yaw_rate_radps": _float(reward_audit.yaw_rate_radps),
            }
        )
    return TensorDict(payload, batch_size=[1])


def finalize_rollout_episode(
    training_transitions: list[TensorDictBase],
    audit_transitions: list[TensorDictBase],
    tail_kind: TailKind,
    tail_bootstrap_value: torch.Tensor,
    reward_profile: RewardProfileName,
) -> RolloutEpisode:
    """Join PPO and CPU audit data after attaching the final GAE boundary value."""

    if not training_transitions or not audit_transitions:
        raise ValueError("rollout episode must contain at least one transition")
    final_transition = training_transitions[-1]
    device = _tensordict_device(final_transition)
    bootstrap = tail_bootstrap_value.detach().to(device)
    set_training_transition_next_state_value(final_transition, bootstrap)
    final_transition["next", "done"] = _bool(True, device)
    training = concatenate_tensordicts(training_transitions)
    return RolloutEpisode(
        training,
        concatenate_tensordicts(audit_transitions),
        tail_kind,
        bootstrap,
        reward_profile,
    )


def _validate_training_trajectory(trajectory: TensorDictBase) -> None:
    _validate_trajectory(trajectory, _TRAINING_KEYS, "PPO training")
    _validate_policy_context(trajectory, "PPO training")
    guidance_action = _tensor(trajectory, "guidance_action")
    if tuple(guidance_action.shape[1:]) != (2,):
        raise ValueError("PPO training guidance_action must have shape [T, 2]")
    if torch.any((guidance_action <= -1.0) | (guidance_action >= 1.0)):
        raise ValueError("PPO training guidance_action must be strictly inside (-1, 1)")
    for key in ("old_joint_guidance_log_prob", "state_value", "reward"):
        value = _tensor(trajectory, ("next", key) if key == "reward" else key)
        if tuple(value.shape[1:]) != (1,):
            raise ValueError(f"PPO training {key} must have shape [T, 1]")
    next_transition = _tensordict(trajectory, "next")
    missing = _NEXT_TRAINING_KEYS - set(next_transition.keys(include_nested=False))
    if missing:
        raise ValueError(f"PPO training next transition is missing fields: {sorted(missing)}")
    for key in ("done", "terminated", "truncated"):
        value = _tensor(next_transition, key)
        if value.dtype != torch.bool or tuple(value.shape[1:]) != (1,):
            raise TypeError(f"PPO training next {key} must be bool with shape [T, 1]")


def rollout_audit_keys(reward_profile: RewardProfileName) -> tuple[str, ...]:
    """Return the exact in-memory audit keys for one reward profile."""

    if reward_profile == "metadrive_builtin_v1":
        return _BUILTIN_AUDIT_KEYS
    if reward_profile == "plannerrft_energy_v1":
        return _ENERGY_AUDIT_KEYS
    raise ValueError("rollout episode has an invalid reward profile")


def _validate_audit_trajectory(
    trajectory: TensorDictBase, reward_profile: RewardProfileName
) -> None:
    actual_keys = set(trajectory.keys(include_nested=False))
    expected_keys = rollout_audit_keys(reward_profile)
    _validate_trajectory(trajectory, frozenset(expected_keys), "rollout audit")
    unexpected = actual_keys - set(expected_keys)
    if unexpected:
        raise ValueError(f"rollout audit trajectory has unexpected fields: {sorted(unexpected)}")
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
    expected_noise_shape = (PLANNER_ACTOR_COUNT, PLANNER_HORIZON, PLANNER_STATE_DIM)
    if tuple(trajectory["initial_noise"].shape[1:]) != expected_noise_shape:
        raise ValueError(
            "rollout audit initial_noise must have shape "
            f"[T, {PLANNER_ACTOR_COUNT}, {PLANNER_HORIZON}, {PLANNER_STATE_DIM}]"
        )
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
    if reward_profile == "plannerrft_energy_v1":
        for key in (
            "reward_gate",
            "collision_score",
            "drivable_score",
            "wrong_direction_score",
            "ttc_score",
            "progress_score",
            "comfort_score",
            "speed_score",
            "energy_score",
        ):
            if torch.any((trajectory[key] < 0.0) | (trajectory[key] > 1.0)):
                raise ValueError(f"rollout audit {key} must remain in [0, 1]")


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
    context = ExplorationPolicyContext(**{key: _tensor(trajectory, key) for key in _CONTEXT_KEYS})
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
    next_transition = _tensordict(trajectory, "next")
    done = _tensor(next_transition, "done")
    if not bool(done[-1].item()) or torch.any(done[:-1]):
        raise ValueError("PPO training next done must mark only the final GAE boundary")
    terminated = bool(_tensor(next_transition, "terminated")[-1].item())
    truncated = bool(_tensor(next_transition, "truncated")[-1].item())
    if tail_kind == "terminated":
        if not terminated or not torch.equal(value, torch.zeros_like(value)):
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


def concatenate_tensordicts(values: Sequence[TensorDictBase], dim: int = 0) -> TensorDictBase:
    """Concatenate TensorDicts through their registered torch dispatch."""

    # TensorDict registers torch.cat at runtime, but torch's stubs only admit Tensor here.
    result = torch.cat(values, dim=dim)  # pyright: ignore[reportCallIssue, reportArgumentType]
    return cast(TensorDictBase, result)


def _tensor(trajectory: TensorDictBase, key: str | tuple[str, ...]) -> torch.Tensor:
    return cast(torch.Tensor, trajectory[key])


def _tensordict(trajectory: TensorDictBase, key: str | tuple[str, ...]) -> TensorDictBase:
    return cast(TensorDictBase, trajectory[key])
