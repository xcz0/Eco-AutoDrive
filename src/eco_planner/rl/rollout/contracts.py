"""TensorDict contracts for one policy-guided rollout episode."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import torch
from tensordict import TensorDictBase, cat

from eco_planner.contracts import (
    CLOSED_LOOP_EXECUTION_STEPS,
    PLANNER_ACTOR_COUNT,
    PLANNER_HORIZON,
    PLANNER_STATE_DIM,
)
from eco_planner.planning.policy import (
    POLICY_CONTEXT_KEYS,
    ExplorationPolicyContext,
    policy_context_tensordict,
)
from eco_planner.reward import PlannerRFTRewardResult, aggregate_transition_reward
from eco_planner.reward.result import RewardProfileName as RewardProfileName

TailKind = Literal["terminated", "truncated", "rollout_limit"]
_CONTEXT_KEYS = POLICY_CONTEXT_KEYS
PPO_BATCH_KEYS = (
    *POLICY_CONTEXT_KEYS,
    "guidance_action",
    "old_joint_guidance_log_prob",
    "advantage",
    "value_target",
)
TRAINING_KEYS = (
    *POLICY_CONTEXT_KEYS,
    "guidance_action",
    "old_joint_guidance_log_prob",
    "state_value",
    "next",
)
_TRAINING_KEYS = frozenset(TRAINING_KEYS)
_NEXT_TRAINING_KEYS = frozenset(
    {
        "state_value",
        "reward",
        "done",
        "terminated",
        "truncated",
    }
)
_DECISION_AUDIT_KEYS = (
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
)
# Source, artifact prefix, dtype, and the explicitly persisted source fields.
_AUDIT_SCALAR_GROUPS = (
    ("reward", "reward_", torch.float32, ("total", "base_total", "safety_gate")),
    (
        "components",
        "reward_component_",
        torch.float32,
        ("ttc", "progress", "comfort", "speed", "energy"),
    ),
    (
        "execution",
        "",
        torch.float32,
        (
            "route_completion_delta",
            "distance_m",
            "speed_mps",
            "position_error_m",
            "heading_error_rad",
        ),
    ),
    (
        "execution",
        "",
        torch.bool,
        (
            "stopped",
            "collision",
            "wrong_direction",
            "arrive_dest",
            "out_of_road",
            "crash_vehicle",
            "crash_object",
            "crash_building",
            "crash_human",
            "crash_sidewalk",
            "terminated",
            "truncated",
        ),
    ),
    (
        "provenance",
        "",
        torch.int64,
        ("map_seed", "noise_seed", "policy_action_seed", "planning_cycle_index"),
    ),
    (
        "diagnostics",
        "",
        torch.float32,
        (
            "step_distance_m",
            "native_step_energy_ml",
            "native_episode_energy_ml",
            "executed_fuel_proxy_step_energy_ml",
            "executed_fuel_proxy_ml_per_km",
            "min_ttc_s",
            "route_progress_delta_m",
            "speed_limit_mps",
            "overspeed_mps",
            "longitudinal_acceleration_mps2",
            "lateral_acceleration_mps2",
            "jerk_mps3",
            "yaw_rate_radps",
        ),
    ),
    ("diagnostics", "", torch.bool, ("energy_distance_valid", "has_ttc_candidate")),
    (
        "diagnostics",
        "reward_diagnostic_",
        torch.float32,
        ("collision_score", "drivable_score", "wrong_direction_score"),
    ),
)
_SUBSTEP_COMPONENT_NAMES = ("ttc", "progress", "comfort", "speed", "energy")
# Per-substep reward inputs persisted for offline reweighting/rescoring. Every
# float/bool field below has shape [T, CLOSED_LOOP_EXECUTION_STEPS] and is only
# valid up to `reward_substep_count`; padded positions are zero/False.
_SUBSTEP_AUDIT_KEYS = (
    "reward_substep_count",
    "reward_substep_safety_gate",
    *(f"reward_substep_component_{name}" for name in _SUBSTEP_COMPONENT_NAMES),
    "reward_substep_route_progress_delta_m",
    "reward_substep_longitudinal_acceleration_mps2",
    "reward_substep_lateral_acceleration_mps2",
    "reward_substep_jerk_mps3",
    "reward_substep_yaw_rate_radps",
    "reward_substep_executed_fuel_proxy_step_energy_ml",
    "reward_substep_step_distance_m",
    "reward_substep_energy_distance_valid",
)
_AUDIT_KEYS = (
    *_DECISION_AUDIT_KEYS,
    *(prefix + field for _, prefix, _, fields in _AUDIT_SCALAR_GROUPS for field in fields),
    *_SUBSTEP_AUDIT_KEYS,
)


@dataclass(frozen=True)
class ExecutionTransitionAudit:
    """Typed environment result for one closed-loop decision and its execution prefix."""

    reward_result: PlannerRFTRewardResult
    substep_results: tuple[PlannerRFTRewardResult, ...]
    route_completion_delta: float
    distance_m: float
    speed_mps: float
    stopped: bool
    collision: bool
    wrong_direction: bool
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

    def append(
        self,
        training_decision: TensorDictBase,
        decision_audit: TensorDictBase,
        execution: ExecutionTransitionAudit,
        provenance: RolloutProvenance,
    ) -> None:
        profile = execution.reward_result.profile_name
        if self._reward_profile is None:
            self._reward_profile = profile
        elif profile != self._reward_profile:
            raise ValueError("one rollout episode cannot mix reward profiles")
        self._training.append(training_decision)
        self._audit.append(build_rollout_audit(decision_audit, execution, provenance))

    def finish(self, tail_kind: TailKind, tail_bootstrap_value: torch.Tensor) -> RolloutEpisode:
        if self.empty:
            raise ValueError("rollout episode must contain at least one transition")
        training = cat(self._training)
        audit = cat(self._audit)
        device = training["state_value"].device
        bootstrap = tail_bootstrap_value.detach().to(device)
        next_transition = audit.select("reward_total", "terminated", "truncated").clone().to(device)
        next_transition.rename_key_("reward_total", "reward")
        next_transition["state_value"] = torch.cat(
            (training["state_value"][1:], bootstrap.reshape(1, 1))
        )
        done = next_transition["terminated"] | next_transition["truncated"]
        done[-1] = True
        next_transition["done"] = done
        training["next"] = next_transition
        return RolloutEpisode(
            training, audit, tail_kind, bootstrap, cast(RewardProfileName, self._reward_profile)
        )


def build_training_decision(
    policy_context: ExplorationPolicyContext,
    guidance_action: torch.Tensor,
    old_joint_guidance_log_prob: torch.Tensor,
    state_value: torch.Tensor,
) -> TensorDictBase:
    """Detach compact PPO inputs on their collection device."""

    return (
        policy_context_tensordict(policy_context)
        .update(
            {
                "guidance_action": guidance_action,
                "old_joint_guidance_log_prob": old_joint_guidance_log_prob.reshape(-1),
                "state_value": state_value.reshape(-1, 1),
            }
        )
        .detach()
        .clone()
    )


def build_rollout_audit(
    decision: TensorDictBase,
    execution: ExecutionTransitionAudit,
    provenance: RolloutProvenance,
) -> TensorDictBase:
    """Project one CPU decision and its scalar execution audit without changing the inputs."""

    audit = decision.select(*_DECISION_AUDIT_KEYS)
    sources = {
        "execution": execution,
        "reward": execution.reward_result,
        "components": execution.reward_result.components,
        "diagnostics": execution.reward_result.diagnostics,
        "provenance": provenance,
    }
    audit.update(
        {
            prefix + field: torch.tensor([[getattr(sources[source], field)]], dtype=dtype)
            for source, prefix, dtype, fields in _AUDIT_SCALAR_GROUPS
            for field in fields
        }
    )
    audit.update(_substep_audit_fields(execution))
    return audit


def _substep_audit_fields(execution: ExecutionTransitionAudit) -> dict[str, torch.Tensor]:
    """Pad per-substep reward inputs to the canonical prefix length."""

    results = execution.substep_results
    limit = CLOSED_LOOP_EXECUTION_STEPS
    if not 1 <= len(results) <= limit:
        raise ValueError(
            "execution transition must carry between 1 and "
            f"{limit} substep results, got {len(results)}"
        )
    if aggregate_transition_reward(results) != execution.reward_result:
        raise ValueError("transition reward_result must remain the aggregation of its substeps")
    float_fields: dict[str, list[float]] = {
        name: [] for name in _SUBSTEP_AUDIT_KEYS if name != "reward_substep_count"
    }
    for result in results:
        diagnostics = result.diagnostics
        float_fields["reward_substep_safety_gate"].append(result.safety_gate)
        for name in _SUBSTEP_COMPONENT_NAMES:
            float_fields[f"reward_substep_component_{name}"].append(
                float(getattr(result.components, name))
            )
        float_fields["reward_substep_route_progress_delta_m"].append(
            diagnostics.route_progress_delta_m
        )
        float_fields["reward_substep_longitudinal_acceleration_mps2"].append(
            diagnostics.longitudinal_acceleration_mps2
        )
        float_fields["reward_substep_lateral_acceleration_mps2"].append(
            diagnostics.lateral_acceleration_mps2
        )
        float_fields["reward_substep_jerk_mps3"].append(diagnostics.jerk_mps3)
        float_fields["reward_substep_yaw_rate_radps"].append(diagnostics.yaw_rate_radps)
        float_fields["reward_substep_executed_fuel_proxy_step_energy_ml"].append(
            diagnostics.executed_fuel_proxy_step_energy_ml
        )
        float_fields["reward_substep_step_distance_m"].append(diagnostics.step_distance_m)
        float_fields["reward_substep_energy_distance_valid"].append(
            float(diagnostics.energy_distance_valid)
        )
    fields: dict[str, torch.Tensor] = {}
    for name, values in float_fields.items():
        dtype = torch.bool if name == "reward_substep_energy_distance_valid" else torch.float32
        padded = torch.zeros((1, limit), dtype=dtype)
        for index, value in enumerate(values):
            padded[0, index] = bool(value) if dtype == torch.bool else value
        fields[name] = padded
    fields["reward_substep_count"] = torch.tensor([[len(results)]], dtype=torch.int64)
    return fields


def _validate_training_trajectory(trajectory: TensorDictBase) -> None:
    _validate_trajectory(trajectory, _TRAINING_KEYS, "PPO training")
    _validate_policy_context(trajectory, "PPO training")
    guidance_action = _tensor(trajectory, "guidance_action")
    if tuple(guidance_action.shape[1:]) != (2,):
        raise ValueError("PPO training guidance_action must have shape [T, 2]")
    if torch.any((guidance_action <= -1.0) | (guidance_action >= 1.0)):
        raise ValueError("PPO training guidance_action must be strictly inside (-1, 1)")
    old_log_prob = _tensor(trajectory, "old_joint_guidance_log_prob")
    if old_log_prob.ndim != 1:
        raise ValueError("PPO training old_joint_guidance_log_prob must have shape [T]")
    for key in ("state_value", "reward"):
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

    # All PlannerRFT profiles share one audit schema: under the no-energy
    # profiles the energy component stays an audited, unweighted diagnostic.
    if reward_profile in (
        "plannerrft_energy_v1",
        "plannerrft_energy_band_lam1_v1",
        "plannerrft_energy_band_lam2_v1",
        "plannerrft_energy_band_lam4_v1",
        "plannerrft_energy_band_lam8_v1",
        "plannerrft_energy_band_lam64_v1",
        "plannerrft_no_energy_v1",
        "plannerrft_no_energy_calibrated_v1",
    ):
        return _AUDIT_KEYS
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
    for key in (
        "reward_safety_gate",
        "reward_diagnostic_collision_score",
        "reward_diagnostic_drivable_score",
        "reward_diagnostic_wrong_direction_score",
    ):
        if torch.any((trajectory[key] < 0.0) | (trajectory[key] > 1.0)):
            raise ValueError(f"rollout audit {key} must remain in [0, 1]")
    # Components are summed over the transition's substeps, so they are
    # non-negative but no longer bounded by one substep's [0, 1] score.
    for key in (
        "reward_component_ttc",
        "reward_component_progress",
        "reward_component_comfort",
        "reward_component_speed",
        "reward_component_energy",
    ):
        if torch.any(trajectory[key] < 0.0):
            raise ValueError(f"rollout audit {key} must be non-negative")
    _validate_substep_audit(trajectory)


def _validate_substep_audit(trajectory: TensorDictBase) -> None:
    count = trajectory["reward_substep_count"]
    if count.dtype != torch.int64 or tuple(count.shape[1:]) != (1,):
        raise TypeError("rollout audit reward_substep_count must be int64 with shape [T, 1]")
    if torch.any((count < 1) | (count > CLOSED_LOOP_EXECUTION_STEPS)):
        raise ValueError(
            f"rollout audit reward_substep_count must be within [1, {CLOSED_LOOP_EXECUTION_STEPS}]"
        )
    for key in _SUBSTEP_AUDIT_KEYS:
        if key == "reward_substep_count":
            continue
        value = trajectory[key]
        if tuple(value.shape[1:]) != (CLOSED_LOOP_EXECUTION_STEPS,):
            raise ValueError(
                f"rollout audit {key} must have shape [T, {CLOSED_LOOP_EXECUTION_STEPS}]"
            )
    for key in (
        "reward_substep_component_ttc",
        "reward_substep_component_progress",
        "reward_substep_component_comfort",
        "reward_substep_component_speed",
        "reward_substep_component_energy",
        "reward_substep_route_progress_delta_m",
        "reward_substep_step_distance_m",
        "reward_substep_executed_fuel_proxy_step_energy_ml",
    ):
        if torch.any(trajectory[key] < 0.0):
            raise ValueError(f"rollout audit {key} must be non-negative")
    gate = trajectory["reward_substep_safety_gate"]
    if torch.any((gate < 0.0) | (gate > 1.0)):
        raise ValueError("rollout audit reward_substep_safety_gate must remain in [0, 1]")
    if trajectory["reward_substep_energy_distance_valid"].dtype != torch.bool:
        raise TypeError("rollout audit reward_substep_energy_distance_valid must be bool")


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


def _tensordict_device(trajectory: TensorDictBase) -> torch.device:
    for value in trajectory.values(include_nested=True, leaves_only=True):
        if isinstance(value, torch.Tensor):
            return value.device
    raise ValueError("TensorDict must contain at least one tensor")


def _tensor(trajectory: TensorDictBase, key: str | tuple[str, ...]) -> torch.Tensor:
    return cast(torch.Tensor, trajectory[key])


def _tensordict(trajectory: TensorDictBase, key: str | tuple[str, ...]) -> TensorDictBase:
    return cast(TensorDictBase, trajectory[key])
