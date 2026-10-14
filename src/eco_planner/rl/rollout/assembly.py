"""Build device training trajectories and matching CPU rollout audits."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch
from tensordict import TensorDict, TensorDictBase, cat

from eco_planner.contracts import CLOSED_LOOP_EXECUTION_STEPS
from eco_planner.planning.policy import ExplorationPolicyContext, policy_context_tensordict
from eco_planner.reward import RewardResult, aggregate_transition_reward

from .contracts import (
    _AUDIT_SCALAR_GROUPS,
    _DECISION_AUDIT_KEYS,
    _SUBSTEP_AUDIT_KEYS,
    _SUBSTEP_COMPONENT_NAMES,
    ExecutionTransitionAudit,
    RewardProfileName,
    RolloutEpisode,
    RolloutProvenance,
    TailKind,
    _tensor,
    _tensordict,
)


class RolloutEpisodeBuilder:
    """Build matching PPO and audit trajectories for serial or vector collection."""

    def __init__(
        self, *, training_storage: TensorDictBase | None = None, training_start: int = 0
    ) -> None:
        self._training: list[TensorDictBase] = []
        self._audit: list[TensorDictBase] = []
        self._reward_profile: RewardProfileName | None = None
        self._training_storage = training_storage
        self._training_start = training_start

    @property
    def transition_count(self) -> int:
        return len(self._audit)

    @property
    def empty(self) -> bool:
        return not self._training and not self._audit

    def append(
        self,
        training_transition: TensorDictBase,
        decision_audit: TensorDictBase,
        execution: ExecutionTransitionAudit,
        provenance: RolloutProvenance,
    ) -> None:
        self.append_batch((self,), training_transition, decision_audit, (execution,), (provenance,))

    @staticmethod
    def append_batch(
        builders: Sequence[RolloutEpisodeBuilder],
        training: TensorDictBase,
        decision_audit: TensorDictBase,
        executions: Sequence[ExecutionTransitionAudit],
        provenances: Sequence[RolloutProvenance],
    ) -> None:
        """Write one vector round once, retaining episode-local CPU audit ownership."""

        count = len(builders)
        if not count or training.batch_size != (count,) or decision_audit.batch_size != (count,):
            raise ValueError("rollout round must contain one transition per episode builder")
        rows = tuple(zip(builders, executions, provenances, strict=True))
        storage = builders[0]._training_storage
        if any(builder._training_storage is not storage for builder in builders):
            raise ValueError("rollout round builders must share the same training storage")
        for builder, execution, _ in rows:
            profile = execution.reward_result.profile_name
            if builder._reward_profile is None:
                builder._reward_profile = profile
            elif profile != builder._reward_profile:
                raise ValueError("one rollout episode cannot mix reward profiles")
        if storage is None:
            for index, builder in enumerate(builders):
                builder._training.append(cast(TensorDictBase, training[index : index + 1]))
        else:
            if "state_value" not in storage.keys():
                storage.update(training.new_empty(storage.batch_size))
                value_dtype = torch.promote_types(storage["state_value"].dtype, torch.float32)
                storage["next", "state_value"] = torch.empty_like(
                    storage["state_value"], dtype=value_dtype
                )
                storage["next", "done"] = torch.empty_like(storage["next", "terminated"])
            indices = [builder._training_start + builder.transition_count for builder in builders]
            if any(index < 0 or index >= storage.batch_size[0] for index in indices):
                raise ValueError("rollout transition exceeds its update training storage")
            # Indexed writes must not broadcast, cast, or move incompatible decisions.
            source_keys = set(training.keys(include_nested=True, leaves_only=True))
            target_keys = set(storage.keys(include_nested=True, leaves_only=True)) - {
                ("next", "state_value"),
                ("next", "done"),
            }
            if source_keys != target_keys:
                raise ValueError("rollout training transition fields differ from update storage")
            for key, value in training.items(include_nested=True, leaves_only=True):
                target = _tensor(storage, key)
                if (
                    value.shape != (count, *target.shape[1:])
                    or value.dtype != target.dtype
                    or value.device != target.device
                ):
                    raise ValueError(f"rollout training field {key!r} differs from update storage")
            index = torch.tensor(indices, dtype=torch.long, device=storage["state_value"].device)
            storage.update_at_(training, index)
        for index, (builder, execution, provenance) in enumerate(rows):
            builder._audit.append(
                build_rollout_audit(
                    cast(TensorDictBase, decision_audit[index : index + 1]), execution, provenance
                )
            )

    def finish(self, tail_kind: TailKind, tail_bootstrap_value: torch.Tensor) -> RolloutEpisode:
        if self.empty:
            raise ValueError("rollout episode must contain at least one transition")
        training = (
            cat(self._training)
            if self._training_storage is None
            else cast(
                TensorDictBase,
                self._training_storage[
                    self._training_start : self._training_start + self.transition_count
                ],
            )
        )
        audit = cat(self._audit)
        device = training["state_value"].device
        bootstrap = tail_bootstrap_value.detach().to(device)
        next_transition = _tensordict(training, "next")
        if "state_value" not in next_transition.keys():
            next_transition["state_value"] = torch.cat(
                (training["state_value"][1:], bootstrap.reshape(1, 1))
            )
        else:
            next_transition["state_value"][:-1].copy_(training["state_value"][1:])
            next_transition["state_value"][-1].copy_(bootstrap)
        done = next_transition["terminated"] | next_transition["truncated"]
        done[-1] = True
        if "done" in next_transition.keys():
            next_transition["done"].copy_(done)
        else:
            next_transition["done"] = done
        training["next"] = next_transition
        return RolloutEpisode(
            training, audit, tail_kind, bootstrap, cast(RewardProfileName, self._reward_profile)
        )

    def next_episode(self) -> RolloutEpisodeBuilder:
        """Start the next episode after this episode's disjoint storage slice."""

        return RolloutEpisodeBuilder(
            training_storage=self._training_storage,
            training_start=self._training_start + self.transition_count,
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


def build_training_transition(
    decision: TensorDictBase,
    reward_result: RewardResult,
    *,
    terminated: bool,
    truncated: bool,
) -> TensorDictBase:
    """Project a scalar reward and environment flags directly onto the collection device."""

    return build_training_transitions(
        decision, (reward_result,), terminated=(terminated,), truncated=(truncated,)
    )


def build_training_transitions(
    decision: TensorDictBase,
    reward_results: Sequence[RewardResult],
    *,
    terminated: Sequence[bool],
    truncated: Sequence[bool],
) -> TensorDictBase:
    """Project a complete vector round's rewards and flags onto the collection device."""

    count = len(reward_results)
    if decision.batch_size != (count,) or len(terminated) != count or len(truncated) != count:
        raise ValueError("training round rewards and flags must match the decision batch")
    device = decision["state_value"].device
    next_transition = TensorDict(
        {
            "reward": torch.tensor(
                [result.total for result in reward_results], dtype=torch.float32, device=device
            ).reshape(count, 1),
            "terminated": torch.tensor(terminated, dtype=torch.bool, device=device).reshape(
                count, 1
            ),
            "truncated": torch.tensor(truncated, dtype=torch.bool, device=device).reshape(count, 1),
        },
        batch_size=[count],
        device=device,
    )
    return decision.clone(recurse=False).set("next", next_transition)


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
