"""Execution and audit results returned by the rollout runtime."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import cast

import numpy as np
import torch
from tensordict import TensorDict, TensorDictBase

from eco_planner.rl.policy import (
    ExplorationPolicyConfig,
    ExplorationPolicyContext,
    validate_exploration_policy_context,
)
from eco_planner.rl.rollout.contracts import DecisionAudit
from eco_planner.runtime.contracts import HostTrajectories
from eco_planner.runtime.host_transfer import DeferredHostTensors, DeferredHostTransferTiming


class RolloutDecision:
    """Execution trajectory plus deferred CPU rollout storage fields."""

    def __init__(
        self,
        execution: HostTrajectories,
        resolve_audit: Callable[[], TensorDictBase],
        diffusion_rng_state: torch.Tensor,
        policy_rng_state: torch.Tensor,
        training_decision: TensorDictBase,
    ) -> None:
        self._execution = execution
        self._resolve_audit = resolve_audit
        self._diffusion_rng_state = diffusion_rng_state
        self._policy_rng_state = policy_rng_state
        self._training_decision = training_decision
        self._audit: DecisionAudit | None = None

    @property
    def ego_trajectory(self) -> np.ndarray:
        return self._execution.ego[0]

    @property
    def training_decision(self) -> TensorDictBase:
        """Return the device-resident PPO inputs without waiting for the audit copy."""

        return self._training_decision

    def audit_result(self) -> DecisionAudit:
        """Wait for the stored PPO/replay payload after simulator execution."""

        if self._audit is None:
            host = self._resolve_audit()
            context = ExplorationPolicyContext(
                scene_tokens=host["scene_tokens"],
                scene_padding_mask=host["scene_padding_mask"],
                navigation_tokens=host["navigation_tokens"],
                navigation_padding_mask=host["navigation_padding_mask"],
                reference_trajectory=host["reference_trajectory"],
            )
            self._audit = DecisionAudit(
                prediction=host["prediction"].numpy(),
                initial_noise=host["initial_noise"],
                policy_context=context,
                base_action=host["base_action"],
                guidance_action=host["guidance_action"],
                old_joint_guidance_log_prob=host["old_joint_guidance_log_prob"],
                old_value=host["old_value"],
                beta_alpha=host["beta_alpha"],
                beta_beta=host["beta_beta"],
                diffusion_rng_state=self._diffusion_rng_state,
                policy_rng_state=self._policy_rng_state,
            )
        return self._audit


class BatchRolloutDecision:
    """Batched policy-guided inference results for fixed vector-rollout slots."""

    def __init__(
        self,
        execution: HostTrajectories,
        deferred: DeferredHostTensors,
        diffusion_rng_states: tuple[torch.Tensor, ...],
        policy_rng_states: tuple[torch.Tensor, ...],
        policy_config: ExplorationPolicyConfig,
        training_decision: TensorDictBase,
    ) -> None:
        self._execution = execution
        self._deferred = deferred
        self._diffusion_rng_states = diffusion_rng_states
        self._policy_rng_states = policy_rng_states
        self._policy_config = policy_config
        self._training_decision = training_decision
        self._audit: TensorDictBase | None = None
        self._slots: list[RolloutDecision | None] = [None] * execution.ego.shape[0]

    @property
    def ego_trajectories(self) -> np.ndarray:
        """Return executable trajectories with shape ``[B, T, 4]``."""

        return self._execution.ego

    @property
    def training_decision(self) -> TensorDictBase:
        """Return batched PPO inputs without waiting for the audit transfer."""

        return self._training_decision

    def audit_result(self) -> TensorDictBase:
        """Resolve and return the complete batched audit payload."""

        return self._resolve_audit()

    @property
    def audit_transfer_timing(self) -> DeferredHostTransferTiming | None:
        """Return profile-only deferred transfer timing after audit resolution."""

        return self._deferred.timing

    def slot(self, index: int) -> RolloutDecision:
        """Adapt one batch slot to the existing serial collector contract."""

        batch = self.ego_trajectories.shape[0]
        if not 0 <= index < batch:
            raise IndexError(f"batch slot {index} is outside [0, {batch})")
        decision = self._slots[index]
        if decision is None:
            decision = RolloutDecision(
                HostTrajectories(self.ego_trajectories[index : index + 1]),
                lambda: _slice_tensordict(self._resolve_audit(), slice(index, index + 1)),
                diffusion_rng_state=self._diffusion_rng_states[index],
                policy_rng_state=self._policy_rng_states[index],
                training_decision=_slice_tensordict(
                    self._training_decision, slice(index, index + 1)
                ),
            )
            self._slots[index] = decision
        return decision

    def _resolve_audit(self) -> TensorDictBase:
        if self._audit is None:
            host = self._deferred.resolve()
            _validate_finite(host)
            validate_exploration_policy_context(
                ExplorationPolicyContext(
                    scene_tokens=host["scene_tokens"],
                    scene_padding_mask=host["scene_padding_mask"],
                    navigation_tokens=host["navigation_tokens"],
                    navigation_padding_mask=host["navigation_padding_mask"],
                    reference_trajectory=host["reference_trajectory"],
                ),
                self._policy_config,
            )
            self._audit = TensorDict(host, batch_size=[self.ego_trajectories.shape[0]])
        return self._audit


def _validate_finite(tensors: Mapping[str, torch.Tensor]) -> None:
    for name, value in tensors.items():
        if value.dtype.is_floating_point and not torch.isfinite(value).all():
            raise RuntimeError(f"rollout host tensor {name!r} contains non-finite values")


def _slice_tensordict(value: TensorDictBase, index: slice) -> TensorDictBase:
    return cast(TensorDictBase, value[index])
