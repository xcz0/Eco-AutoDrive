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
    _tensordict,
)


class RolloutEpisodeBuilder:
    """Build matching PPO and audit trajectories for serial or vector collection."""

    def __init__(self) -> None:
        self._training: list[TensorDictBase] = []
        self._audit: list[TensorDictBase] = []
        self._reward_profile: RewardProfileName | None = None

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
        self._training.append(training_transition)
        self.append_audit_batch((self,), decision_audit, (execution,), (provenance,))

    @staticmethod
    def append_audit_batch(
        builders: Sequence[RolloutEpisodeBuilder],
        decision_audit: TensorDictBase,
        executions: Sequence[ExecutionTransitionAudit],
        provenances: Sequence[RolloutProvenance],
    ) -> None:
        """Record domain audit while TorchRL assembles the training trajectory."""

        if decision_audit.batch_size != (len(builders),):
            raise ValueError("rollout audit must contain one decision per episode builder")
        for index, (builder, execution, provenance) in enumerate(
            zip(builders, executions, provenances, strict=True)
        ):
            profile = execution.reward_result.profile_name
            if builder._reward_profile is None:
                builder._reward_profile = profile
            elif profile != builder._reward_profile:
                raise ValueError("one rollout episode cannot mix reward profiles")
            builder._audit.append(
                build_rollout_audit(
                    cast(TensorDictBase, decision_audit[index : index + 1]), execution, provenance
                )
            )

    def finish(
        self,
        tail_kind: TailKind,
        tail_bootstrap_value: torch.Tensor,
        *,
        training: TensorDictBase | None = None,
    ) -> RolloutEpisode:
        if self.empty:
            raise ValueError("rollout episode must contain at least one transition")
        if training is None:
            training = cat(self._training)
        if training.batch_size != (self.transition_count,):
            raise ValueError("assembled training must match the episode audit length")
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
