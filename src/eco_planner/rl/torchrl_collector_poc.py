"""Synchronous TorchRL collector proof of concept for one no-traffic rollout slot."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import numpy as np
import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.collectors import Collector

from eco_planner.envs import MetaDriveEnvSlot, PlannerObservationSpec, TorchRLMetaDriveEnv
from eco_planner.evaluation.config import ScenarioConfig
from eco_planner.rl.collector import _seed, _transition_audit
from eco_planner.rl.rollout import (
    RolloutEpisode,
    build_rollout_audit,
    build_training_transition,
    finalize_rollout_episode,
    set_training_transition_next_state_value,
)
from eco_planner.rl.runtime import FabricRolloutRuntime, RolloutAudit, RolloutDecision

_OBSERVATION_KEYS = (
    "ego_current_state",
    "neighbor_agents_past",
    "static_objects",
    "lanes",
    "lanes_speed_limit",
    "lanes_has_speed_limit",
    "route_lanes",
    "route_lanes_speed_limit",
    "route_lanes_has_speed_limit",
)
_CONTEXT_KEYS = (
    "scene_tokens",
    "scene_padding_mask",
    "navigation_tokens",
    "navigation_padding_mask",
    "reference_trajectory",
)


class _DecisionPolicy:
    """Adapt one runtime decision to TorchRL's callable policy boundary."""

    def __init__(
        self,
        runtime: FabricRolloutRuntime,
        diffusion_generator: torch.Generator,
        policy_generator: torch.Generator,
    ) -> None:
        self._runtime = runtime
        self._diffusion_generator = diffusion_generator
        self._policy_generator = policy_generator
        self._decision: RolloutDecision | None = None
        self._audit_records: list[tuple[RolloutAudit, dict[str, float | bool]]] = []

    def __call__(self, tensordict: TensorDictBase) -> TensorDictBase:
        observation = {key: tensordict[key].unsqueeze(0) for key in _OBSERVATION_KEYS}
        decision = self._runtime.decide(
            observation, self._diffusion_generator, self._policy_generator
        )
        self._decision = decision
        training = decision.training_decision.to(device="cpu")
        tensordict.update({key: training[key].squeeze(0) for key in training.keys()})
        tensordict["action"] = torch.from_numpy(decision.ego_trajectory.copy())
        return tensordict

    def resolve_audit(self) -> RolloutAudit:
        if self._decision is None:
            raise RuntimeError("TorchRL collector stepped without a runtime decision")
        return self._decision.audit_result()

    def record_execution(
        self,
        execution: object,
        previous_route_completion: float,
        stopped_speed_threshold_mps: float,
    ) -> None:
        from eco_planner.envs import TrajectoryExecutionRecord

        if not isinstance(execution, TrajectoryExecutionRecord):
            raise TypeError("TorchRL rollout execution must be a TrajectoryExecutionRecord")
        transition = _transition_audit(
            execution, previous_route_completion, stopped_speed_threshold_mps
        )
        transition["dense_reward"] = float(execution.substep_dense_rewards.sum())
        transition["terminal_override"] = float(
            (execution.substep_rewards - execution.substep_dense_rewards).sum()
        )
        self._audit_records.append((self.resolve_audit(), transition))

    @property
    def audit_records(self) -> tuple[tuple[RolloutAudit, dict[str, float | bool]], ...]:
        return tuple(self._audit_records)


class _TorchRLRolloutEnv(TorchRLMetaDriveEnv):
    """Add project audit fields after the simulator step without changing EnvBase specs."""

    def __init__(self, *args: object, stopped_speed_threshold_mps: float, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._stopped_speed_threshold_mps = stopped_speed_threshold_mps
        self._decision_policy: _DecisionPolicy | None = None
        self._previous_route_completion = 0.0

    def bind_decision_policy(self, policy: _DecisionPolicy) -> None:
        self._decision_policy = policy

    def _reset(self, tensordict: TensorDictBase | None, **kwargs: object) -> TensorDictBase:
        observation = super()._reset(tensordict, **kwargs)
        self._previous_route_completion = self._slot.env.route_completion
        return observation

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        result = super()._step(tensordict)
        if self._decision_policy is None:
            raise RuntimeError("TorchRL rollout environment has no decision policy")
        execution = self.last_execution
        if execution.substep_states.shape[0] != 1:
            raise RuntimeError("rollout transition must execute exactly one substep")
        self._decision_policy.record_execution(
            execution,
            self._previous_route_completion,
            self._stopped_speed_threshold_mps,
        )
        self._previous_route_completion = execution.route_completion
        return result


def collect_torchrl_rollout_poc(
    spec: ScenarioConfig,
    runtime: FabricRolloutRuntime,
    env_config: Mapping[str, object],
    *,
    mode: Literal["no_traffic", "traffic"],
    map_query_radius_m: float,
    history_warmup_steps: int,
    max_transitions: int,
    stopped_speed_threshold_mps: float = 0.1,
    diffusion_generator: torch.Generator | None = None,
    policy_generator: torch.Generator | None = None,
    noise_seed: int | None = None,
    policy_action_seed: int | None = None,
) -> RolloutEpisode:
    """Collect one no-traffic episode through TorchRL's synchronous ``Collector``.

    This is intentionally a single-slot PoC.  It proves the standard TorchRL policy→step→stack
    orchestration without changing the production vector collector or its worker scheduling.
    """

    if mode != "no_traffic":
        raise ValueError("TorchRL collector PoC only supports no_traffic mode")
    if type(max_transitions) is not int or max_transitions <= 0:
        raise ValueError("max_transitions must be a positive integer")
    if type(history_warmup_steps) is not int or history_warmup_steps < 0:
        raise ValueError("history_warmup_steps must be a non-negative integer")
    if (
        type(stopped_speed_threshold_mps) is not float
        or not np.isfinite(stopped_speed_threshold_mps)
        or stopped_speed_threshold_mps <= 0.0
    ):
        raise ValueError("stopped_speed_threshold_mps must be a positive finite float")
    if env_config.get("trajectory_execution_steps") != 1:
        raise ValueError("rollout requires env.trajectory_execution_steps=1")

    resolved_noise_seed = runtime.noise_seed if noise_seed is None else _seed(noise_seed, "noise")
    resolved_policy_seed = (
        runtime.policy_action_seed
        if policy_action_seed is None
        else _seed(policy_action_seed, "policy action")
    )
    diffusion_generator = diffusion_generator or runtime.new_noise_generator()
    policy_generator = policy_generator or runtime.new_policy_generator()
    configured = {**env_config, "map": spec.map}
    slot = MetaDriveEnvSlot(
        configured,
        mode=mode,
        observation_spec=PlannerObservationSpec.from_planner_config(runtime.planner_config),
        map_query_radius_m=map_query_radius_m,
        history_warmup_steps=history_warmup_steps,
    )
    env = _TorchRLRolloutEnv(
        slot,
        map_name=spec.map,
        seed=spec.seed,
        observation_spec=PlannerObservationSpec.from_planner_config(runtime.planner_config),
        stopped_speed_threshold_mps=stopped_speed_threshold_mps,
    )
    policy = _DecisionPolicy(runtime, diffusion_generator, policy_generator)
    env.bind_decision_policy(policy)
    diffusion_state = diffusion_generator.get_state().clone()
    policy_state = policy_generator.get_state().clone()
    collector = Collector(
        env,
        policy,
        frames_per_batch=max_transitions,
        total_frames=max_transitions,
        init_random_frames=0,
        reset_at_each_iter=True,
        device="cpu",
        storing_device="cpu",
        policy_device="cpu",
        env_device="cpu",
        trust_policy=True,
    )
    # TorchRL validates a callable policy once while constructing the collector.  That dry run is
    # not an environment transition, so restore the two explicit per-slot streams before collection.
    diffusion_generator.set_state(diffusion_state)
    policy_generator.set_state(policy_state)
    try:
        batch = next(iter(collector))
    finally:
        collector.shutdown()
    return _episode_from_collector_batch(
        batch,
        runtime,
        diffusion_generator,
        policy.audit_records,
        map_seed=spec.seed,
        noise_seed=resolved_noise_seed,
        policy_action_seed=resolved_policy_seed,
    )


def _episode_from_collector_batch(
    batch: TensorDictBase,
    runtime: FabricRolloutRuntime,
    diffusion_generator: torch.Generator,
    audit_records: tuple[tuple[RolloutAudit, dict[str, float | bool]], ...],
    *,
    map_seed: int,
    noise_seed: int,
    policy_action_seed: int,
) -> RolloutEpisode:
    if len(batch.batch_size) != 1 or batch.batch_size[0] <= 0:
        raise RuntimeError("TorchRL collector PoC must return one non-empty time batch")
    transitions = []
    audits = []
    count = batch.batch_size[0]
    if len(audit_records) != count:
        raise RuntimeError("TorchRL collector PoC lost an audit record")
    done = batch["next", "done"].reshape(count)
    if torch.any(done[:-1]):
        raise RuntimeError("TorchRL collector PoC cannot reset within a collected episode")
    for index in range(count):
        training = TensorDict(
            {key: batch[key][index : index + 1] for key in _CONTEXT_KEYS + (
                "guidance_action",
                "old_joint_guidance_log_prob",
                "state_value",
            )},
            batch_size=[1],
        )
        terminated = bool(batch["next", "terminated"][index].item())
        truncated = bool(batch["next", "truncated"][index].item())
        transitions.append(
            build_training_transition(
                training,
                reward=float(batch["next", "reward"][index].item()),
                terminated=terminated,
                truncated=truncated,
            )
        )
        audit_result, audit = audit_records[index]
        audits.append(
            build_rollout_audit(
                policy_context=audit_result.policy_context,
                base_action=audit_result.base_action,
                guidance_action=audit_result.guidance_action,
                old_joint_guidance_log_prob=audit_result.old_joint_guidance_log_prob,
                state_value=audit_result.old_value,
                beta_alpha=audit_result.beta_alpha,
                beta_beta=audit_result.beta_beta,
                initial_noise=audit_result.initial_noise,
                diffusion_rng_state=audit_result.diffusion_rng_state,
                policy_rng_state=audit_result.policy_rng_state,
                reward=float(batch["next", "reward"][index].item()),
                dense_reward=float(audit.pop("dense_reward")),
                terminal_override=float(audit.pop("terminal_override")),
                **audit,
                terminated=terminated,
                truncated=truncated,
                map_seed=map_seed,
                noise_seed=noise_seed,
                policy_action_seed=policy_action_seed,
                planning_cycle_index=index,
            )
        )
    for index in range(count - 1):
        set_training_transition_next_state_value(
            transitions[index], batch["state_value"][index + 1]
        )
    final_terminated = bool(batch["next", "terminated"][-1].item())
    final_truncated = bool(batch["next", "truncated"][-1].item())
    if final_terminated:
        return finalize_rollout_episode(transitions, audits, "terminated", torch.zeros(1))
    next_observation = {key: batch["next", key][-1:].clone() for key in _OBSERVATION_KEYS}
    bootstrap = runtime.bootstrap_value(next_observation, diffusion_generator)
    return finalize_rollout_episode(
        transitions, audits, "truncated" if final_truncated else "rollout_limit", bootstrap
    )
