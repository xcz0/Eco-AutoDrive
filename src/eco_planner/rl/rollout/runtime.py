"""Training-owned learned-guidance rollout adapter over the planning runtime."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import torch
from lightning.fabric import Fabric
from tensordict import TensorDictBase

from eco_planner.planning import PolicyGuidanceRuntime, create_policy_guidance_runtime
from eco_planner.planning.diffusion import (
    CheckpointLoadReport,
    OfficialDiffusionPlannerConfig,
    OrthogonalPolicyGuidanceConfig,
    PretrainedDiffusionPlanner,
    SamplerConfig,
    SamplerReport,
)
from eco_planner.planning.policy import (
    ExplorationPolicy,
    ExplorationPolicyConfig,
    build_policy_inputs,
    policy_context_tensordict,
)
from eco_planner.runtime.config import RuntimeConfig
from eco_planner.runtime.fabric import InferenceRuntimeReport
from eco_planner.runtime.host_transfer import HostTransfer
from eco_planner.runtime.profiling import (
    PhaseProfiler,
    finish_profile,
    profile_call,
    require_phase,
)
from eco_planner.runtime.random import sample_batched_standard_normal

from .contracts import build_training_decision
from .decision import BatchRolloutDecision, RolloutDecision
from .profiling import RolloutPlannerTiming


class FabricRolloutRuntime:
    """Adapt the planning-owned policy guidance runtime to PPO rollout collection."""

    def __init__(self, planning: PolicyGuidanceRuntime) -> None:
        self._planning = planning
        self._host_transfer = HostTransfer(planning.device)

    @property
    def device(self) -> torch.device:
        return self._planning.device

    @property
    def fabric(self) -> Fabric:
        """Expose the single-device Fabric owner for training checkpoint persistence."""

        return self._planning.fabric

    @property
    def planner(self) -> PretrainedDiffusionPlanner:
        """Expose the frozen planner for gradient-absence diagnostics."""

        return self._planning.planner

    @property
    def planner_config(self) -> OfficialDiffusionPlannerConfig:
        """Expose the immutable planner architecture for observation adapters."""

        return self._planning.planner_config

    @property
    def policy(self) -> ExplorationPolicy:
        """Expose the single trainable parameter owner to the PPO updater."""

        return self._planning.policy

    @property
    def report(self) -> InferenceRuntimeReport:
        return self._planning.report

    @property
    def checkpoint_report(self) -> CheckpointLoadReport:
        return self._planning.checkpoint_report

    @property
    def sampler_report(self) -> SamplerReport:
        return self._planning.sampler_report

    @property
    def guidance_config(self) -> OrthogonalPolicyGuidanceConfig:
        return self._planning.guidance_config

    @property
    def planner_compile_mode(self) -> Literal["eager", "dit_reduce_overhead"]:
        return self._planning.planner_compile_mode

    @property
    def noise_seed(self) -> int:
        return self._planning.noise_seed

    @property
    def policy_action_seed(self) -> int:
        return self._planning.policy_action_seed

    def frozen_planner_hash(self) -> str:
        """Hash the frozen planner parameters in stable name order."""

        return self._planning.frozen_planner_hash()

    def new_noise_generator(self, seed: int) -> torch.Generator:
        return self._planning.new_noise_generator(seed)

    def new_policy_generator(self, seed: int) -> torch.Generator:
        return self._planning.new_policy_generator(seed)

    def decide(
        self,
        observation: TensorDictBase,
        diffusion_generator: torch.Generator,
        policy_generator: torch.Generator,
    ) -> RolloutDecision:
        """Sample one action through the shared batched runtime path."""

        return self.decide_batch(
            observation,
            (diffusion_generator,),
            (policy_generator,),
        ).slot(0)

    def decide_batch(
        self,
        observation: TensorDictBase,
        diffusion_generators: Sequence[torch.Generator],
        policy_generators: Sequence[torch.Generator],
        *,
        timings: list[RolloutPlannerTiming] | None = None,
    ) -> BatchRolloutDecision:
        """Sample batched guidance actions with one independent RNG stream per slot."""

        return self._decide_batch(
            observation,
            diffusion_generators,
            policy_generators,
            timings=timings,
        )

    def decide_batch_mean(
        self,
        observation: TensorDictBase,
        diffusion_generators: Sequence[torch.Generator],
        *,
        timings: list[RolloutPlannerTiming] | None = None,
    ) -> BatchRolloutDecision:
        """Evaluate deterministic Beta-mean actions without consuming policy RNG."""

        return self._decide_batch(
            observation,
            diffusion_generators,
            None,
            timings=timings,
        )

    def _decide_batch(
        self,
        observation: TensorDictBase,
        diffusion_generators: Sequence[torch.Generator],
        policy_generators: Sequence[torch.Generator] | None,
        *,
        timings: list[RolloutPlannerTiming] | None,
    ) -> BatchRolloutDecision:
        batch = _observation_batch_size(observation)
        _validate_slot_generators(diffusion_generators, batch, "diffusion_generators")
        if policy_generators is not None:
            _validate_slot_generators(policy_generators, batch, "policy_generators")
        profile = timings is not None
        diffusion_rng_states = tuple(_rng_state(generator) for generator in diffusion_generators)
        policy_rng_states = (
            tuple(_rng_state(generator) for generator in policy_generators)
            if policy_generators is not None
            else tuple(torch.empty(0, dtype=torch.uint8) for _ in range(batch))
        )
        profiler = PhaseProfiler(self.device) if profile else None
        result = self._planning.decide_batch(
            observation,
            diffusion_generators,
            policy_generators,
            profiler=profiler,
        )
        decision = result.policy
        diagnostics = result.guidance_diagnostics
        training_decision = build_training_decision(
            decision.inputs,
            decision.action.guidance_action,
            decision.action.joint_log_prob,
            decision.output.value,
        )
        inputs = decision.inputs
        deferred = self._host_transfer.defer(
            {
                "prediction": (result.prediction, torch.float32),
                "initial_noise": (result.initial_noise, torch.float32),
                "reference_prediction": (result.reference_prediction, torch.float32),
                "lateral_target_offset_m": (
                    diagnostics.lateral_target_offset_m,
                    torch.float32,
                ),
                "longitudinal_target_speed_fraction": (
                    diagnostics.longitudinal_target_speed_fraction,
                    torch.float32,
                ),
                "longitudinal_target_speed_delta_mps": (
                    diagnostics.longitudinal_target_speed_delta_mps,
                    torch.float32,
                ),
                "lateral_objective_delta": (
                    diagnostics.lateral_objective_delta,
                    torch.float32,
                ),
                "longitudinal_objective_delta": (
                    diagnostics.longitudinal_objective_delta,
                    torch.float32,
                ),
                "applied_gradient_l2": (
                    diagnostics.applied_gradient_l2,
                    torch.float32,
                ),
                "applied_gradient_max_abs": (
                    diagnostics.applied_gradient_max_abs,
                    torch.float32,
                ),
                "raw_neighbor_gradient_l2": (
                    diagnostics.raw_neighbor_gradient_l2,
                    torch.float32,
                ),
                "zero_speed_count": (diagnostics.zero_speed_count, torch.int64),
                "scene_tokens": (inputs.scene_tokens, torch.float32),
                "scene_padding_mask": (inputs.scene_padding_mask, torch.bool),
                "navigation_tokens": (inputs.navigation_tokens, torch.float32),
                "navigation_padding_mask": (inputs.navigation_padding_mask, torch.bool),
                "reference_trajectory": (inputs.reference_trajectory, torch.float32),
                "base_action": (decision.action.base_action, torch.float32),
                "guidance_action": (decision.action.guidance_action, torch.float32),
                "old_joint_guidance_log_prob": (
                    decision.action.joint_log_prob.reshape(-1, 1),
                    torch.float32,
                ),
                "state_value": (decision.output.value.reshape(-1, 1), torch.float32),
                "beta_alpha": (decision.output.distribution.parameters.alpha, torch.float32),
                "beta_beta": (decision.output.distribution.parameters.beta, torch.float32),
            },
            profile=profile,
        )
        if profiler is None:
            execution = self._host_transfer.execution_trajectories(result.prediction)
            sync_wait_s = 0.0
        else:
            execution = profiler.measure(
                "execution_to_host",
                lambda: self._host_transfer.execution_trajectories(result.prediction),
            )
            sync_wait_s = profiler.finish()
        if profiler is not None and timings is not None:
            timings.append(
                RolloutPlannerTiming(
                    phase="decision",
                    host_to_device=profiler.phase("host_to_device"),
                    diffusion_noise=profiler.phase("diffusion_noise"),
                    prepare_policy_guidance=profiler.phase("prepare_policy_guidance"),
                    policy_forward=profiler.phase("policy_forward"),
                    action_sampling=profiler.phase("action_sampling"),
                    complete_policy_guidance=profiler.phase("complete_policy_guidance"),
                    execution_to_host=profiler.phase("execution_to_host"),
                    profile_sync_wait_wall_s=sync_wait_s,
                )
            )
        return BatchRolloutDecision(
            execution,
            deferred,
            diffusion_rng_states=diffusion_rng_states,
            policy_rng_states=policy_rng_states,
            policy_config=self._planning.policy.config,
            training_decision=training_decision,
        )

    def bootstrap_value(
        self,
        observation: TensorDictBase,
        diffusion_generator: torch.Generator,
    ) -> torch.Tensor:
        """Evaluate one old critic value through the shared batched runtime path."""

        return self.bootstrap_value_batch(observation, (diffusion_generator,))

    def bootstrap_value_batch(
        self,
        observation: TensorDictBase,
        diffusion_generators: Sequence[torch.Generator],
        *,
        timings: list[RolloutPlannerTiming] | None = None,
    ) -> torch.Tensor:
        """Evaluate old critic values for a batch without sampling or executing actions."""

        batch = _observation_batch_size(observation)
        _validate_slot_generators(diffusion_generators, batch, "diffusion_generators")
        profile = timings is not None
        planner = self._planning.planner
        policy = self._planning.policy
        moved, h2d_timing = profile_call(
            self.device, profile, lambda: self._planning.fabric.to_device(observation)
        )
        if not isinstance(moved, TensorDictBase):
            raise TypeError("Fabric must preserve the rollout TensorDict observation container")
        noise, noise_timing = profile_call(
            self.device,
            profile,
            lambda: sample_batched_standard_normal(
                diffusion_generators,
                (
                    1 + planner.config.predicted_neighbor_num,
                    planner.config.future_len,
                    4,
                ),
                device=self.device,
            ),
        )
        with torch.no_grad():
            prepared, prepare_timing = profile_call(
                self.device,
                profile,
                lambda: planner.prepare_policy_guidance(moved, noise, diffusion_generators),
            )
            context = build_policy_inputs(prepared.representations, prepared.reference_prediction)
            value, policy_timing = profile_call(
                self.device,
                profile,
                lambda: (
                    policy.forward_tensordict(policy_context_tensordict(context))["state_value"]
                    .squeeze(-1)
                    .detach()
                    .clone()
                ),
            )
        sync_wait_s = finish_profile(self.device, profile)
        if timings is not None:
            timings.append(
                RolloutPlannerTiming(
                    phase="bootstrap",
                    host_to_device=require_phase(h2d_timing),
                    diffusion_noise=require_phase(noise_timing),
                    prepare_policy_guidance=require_phase(prepare_timing),
                    policy_forward=require_phase(policy_timing),
                    action_sampling=None,
                    complete_policy_guidance=None,
                    execution_to_host=None,
                    profile_sync_wait_wall_s=sync_wait_s,
                )
            )
        return value


def create_fabric_rollout_runtime(
    runtime_config: RuntimeConfig,
    sampler_config: SamplerConfig,
    guidance_config: OrthogonalPolicyGuidanceConfig,
    policy_config: ExplorationPolicyConfig,
    args_path: Path,
    checkpoint_path: Path,
    policy_action_seed: int,
    *,
    planner_compile_mode: Literal["eager", "dit_reduce_overhead"],
) -> FabricRolloutRuntime:
    """Load the frozen planner and an exploration policy for PPO rollout collection."""

    planning = create_policy_guidance_runtime(
        runtime_config,
        sampler_config,
        guidance_config,
        policy_config,
        args_path,
        checkpoint_path,
        policy_action_seed,
        planner_compile_mode=planner_compile_mode,
    )
    return FabricRolloutRuntime(planning)


def _observation_batch_size(observation: TensorDictBase) -> int:
    value = observation.get("ego_current_state")
    if not isinstance(value, torch.Tensor) or value.ndim < 1 or value.shape[0] <= 0:
        raise ValueError(
            "rollout observation must have a positive ego_current_state batch dimension"
        )
    return value.shape[0]


def _validate_slot_generators(generators: Sequence[torch.Generator], batch: int, name: str) -> None:
    if len(generators) != batch:
        raise ValueError(f"{name} must contain one generator per batch item")


def _rng_state(generator: torch.Generator) -> torch.Tensor:
    return generator.get_state().detach().cpu().clone()
