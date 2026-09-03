"""Fabric-owned learned-guidance rollout inference with one host transfer per decision."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import torch
from lightning.fabric import Fabric
from tensordict import TensorDictBase

from eco_planner.models import (
    CheckpointLoadReport,
    OfficialDiffusionPlannerConfig,
    OrthogonalPolicyGuidanceConfig,
    PretrainedDiffusionPlanner,
    SamplerConfig,
    SamplerReport,
    load_official_diffusion_planner,
    sampler_report,
)
from eco_planner.rl.policy import (
    ExplorationPolicy,
    ExplorationPolicyConfig,
    ExplorationPolicyContext,
    ExplorationPolicyOutput,
    policy_context_tensordict,
)
from eco_planner.rl.policy.distribution import (
    AffineBetaAction,
    AffineBetaParameters,
    ExplicitGeneratorBetaSampler,
)
from eco_planner.rl.rollout.contracts import build_training_decision
from eco_planner.rl.rollout.decision import BatchRolloutDecision, RolloutDecision
from eco_planner.rl.rollout.profiling import (
    RolloutPlannerTiming,
    finish_profile,
    profile_call,
    require_phase,
)
from eco_planner.runtime.config import RuntimeConfig
from eco_planner.runtime.fabric import InferenceRuntimeReport, create_single_device_fabric
from eco_planner.runtime.host_transfer import HostTransfer
from eco_planner.runtime.random import sample_batched_standard_normal


class FabricRolloutRuntime:
    """Own the frozen planner, trainable policy module, and separated rollout RNG streams."""

    def __init__(
        self,
        fabric: Fabric,
        planner: PretrainedDiffusionPlanner,
        policy: ExplorationPolicy,
        report: InferenceRuntimeReport,
        noise_seed: int,
        policy_action_seed: int,
        checkpoint_report: CheckpointLoadReport,
        sampler: SamplerReport,
        guidance_config: OrthogonalPolicyGuidanceConfig,
        planner_compile_mode: Literal["eager", "dit_reduce_overhead"],
    ) -> None:
        self._fabric = fabric
        self._planner = planner
        self._policy = policy
        self._policy.eval()
        self.report = report
        self.noise_seed = noise_seed
        self.policy_action_seed = policy_action_seed
        self.checkpoint_report = checkpoint_report
        self.sampler_report = sampler
        self.guidance_config = guidance_config
        self.planner_compile_mode = planner_compile_mode
        self._host_transfer = HostTransfer(fabric.device)

    @property
    def device(self) -> torch.device:
        return self._fabric.device

    @property
    def fabric(self) -> Fabric:
        """Expose the single-device Fabric owner for training checkpoint persistence."""

        return self._fabric

    @property
    def planner_config(self) -> OfficialDiffusionPlannerConfig:
        """Expose the immutable planner architecture for observation adapters."""

        return self._planner.config

    @property
    def policy(self) -> ExplorationPolicy:
        """Expose the single trainable parameter owner to the PPO updater."""

        return self._policy

    def frozen_planner_hash(self) -> str:
        """Hash the frozen planner parameters in stable name order."""

        digest = hashlib.sha256()
        for name, parameter in sorted(self._planner.named_parameters()):
            if parameter.requires_grad:
                raise RuntimeError(f"planner parameter {name!r} is unexpectedly trainable")
            value = parameter.detach().to(device="cpu").contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(value.numpy().tobytes())
        return digest.hexdigest()

    def new_noise_generator(self, seed: int | None = None) -> torch.Generator:
        selected = self.noise_seed if seed is None else _seed(seed, "noise seed")
        return torch.Generator(device=self.device).manual_seed(selected)

    def new_policy_generator(self, seed: int | None = None) -> torch.Generator:
        selected = self.policy_action_seed if seed is None else _seed(seed, "policy action seed")
        return torch.Generator(device=self.device).manual_seed(selected)

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
        moved, h2d_timing = profile_call(
            self.device, profile, lambda: self._fabric.to_device(observation)
        )
        moved = _fabric_observation(moved)
        diffusion_rng_states = tuple(_rng_state(generator) for generator in diffusion_generators)
        policy_rng_states = (
            tuple(_rng_state(generator) for generator in policy_generators)
            if policy_generators is not None
            else tuple(torch.empty(0, dtype=torch.uint8) for _ in range(batch))
        )
        noise, noise_timing = profile_call(
            self.device,
            profile,
            lambda: sample_batched_standard_normal(
                diffusion_generators,
                (
                    1 + self._planner.config.predicted_neighbor_num,
                    self._planner.config.future_len,
                    4,
                ),
                device=self.device,
            ),
        )
        with torch.no_grad():
            prepared, prepare_timing = profile_call(
                self.device,
                profile,
                lambda: self._planner.prepare_policy_guidance(moved, noise, diffusion_generators),
            )
            policy_context = ExplorationPolicyContext(
                scene_tokens=prepared.policy_context.scene_tokens,
                scene_padding_mask=prepared.policy_context.scene_padding_mask,
                navigation_tokens=prepared.policy_context.navigation_tokens,
                navigation_padding_mask=prepared.policy_context.navigation_padding_mask,
                reference_trajectory=prepared.policy_context.reference_trajectory,
            )
            policy_outputs, policy_timing = profile_call(
                self.device,
                profile,
                lambda: self._policy.forward_tensordict(policy_context_tensordict(policy_context)),
            )
            output = self._policy.output_from_tensordict(policy_outputs)
            action, action_timing = profile_call(
                self.device,
                profile,
                lambda: (
                    _sample_policy_actions(output, policy_generators)
                    if policy_generators is not None
                    else output.distribution.action_mean()
                ),
            )
        with torch.enable_grad():
            result, complete_timing = profile_call(
                self.device,
                profile,
                lambda: self._planner.complete_policy_guidance(prepared, action.guidance_action),
            )
        training_decision = build_training_decision(
            policy_context,
            action.guidance_action,
            action.joint_guidance_log_prob,
            output.value,
        )
        if result.reference_prediction is None or result.guidance_diagnostics is None:
            raise RuntimeError(
                "policy guidance planner result is missing required trace diagnostics"
            )
        diagnostics = result.guidance_diagnostics
        deferred = self._host_transfer.defer(
            {
                "prediction": (result.prediction, torch.float32),
                "initial_noise": (noise, torch.float32),
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
                "scene_tokens": (policy_context.scene_tokens, torch.float32),
                "scene_padding_mask": (policy_context.scene_padding_mask, torch.bool),
                "navigation_tokens": (policy_context.navigation_tokens, torch.float32),
                "navigation_padding_mask": (policy_context.navigation_padding_mask, torch.bool),
                "reference_trajectory": (policy_context.reference_trajectory, torch.float32),
                "base_action": (action.base_action, torch.float32),
                "guidance_action": (action.guidance_action, torch.float32),
                "old_joint_guidance_log_prob": (action.joint_guidance_log_prob, torch.float32),
                "old_value": (output.value, torch.float32),
                "beta_alpha": (output.distribution.parameters.alpha, torch.float32),
                "beta_beta": (output.distribution.parameters.beta, torch.float32),
            },
            profile=profile,
        )
        execution, execution_timing = profile_call(
            self.device,
            profile,
            lambda: self._host_transfer.execution_trajectories(result.prediction),
        )
        sync_wait_s = finish_profile(self.device, profile)
        if timings is not None:
            timings.append(
                RolloutPlannerTiming(
                    phase="decision",
                    host_to_device=require_phase(h2d_timing),
                    diffusion_noise=require_phase(noise_timing),
                    prepare_policy_guidance=require_phase(prepare_timing),
                    policy_forward=require_phase(policy_timing),
                    action_sampling=require_phase(action_timing),
                    complete_policy_guidance=require_phase(complete_timing),
                    execution_to_host=require_phase(execution_timing),
                    profile_sync_wait_wall_s=sync_wait_s,
                )
            )
        return BatchRolloutDecision(
            execution,
            deferred,
            diffusion_rng_states=diffusion_rng_states,
            policy_rng_states=policy_rng_states,
            policy_config=self._policy.config,
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
        moved, h2d_timing = profile_call(
            self.device, profile, lambda: self._fabric.to_device(observation)
        )
        moved = _fabric_observation(moved)
        noise, noise_timing = profile_call(
            self.device,
            profile,
            lambda: sample_batched_standard_normal(
                diffusion_generators,
                (
                    1 + self._planner.config.predicted_neighbor_num,
                    self._planner.config.future_len,
                    4,
                ),
                device=self.device,
            ),
        )
        with torch.no_grad():
            prepared, prepare_timing = profile_call(
                self.device,
                profile,
                lambda: self._planner.prepare_policy_guidance(moved, noise, diffusion_generators),
            )
            context = ExplorationPolicyContext(
                scene_tokens=prepared.policy_context.scene_tokens,
                scene_padding_mask=prepared.policy_context.scene_padding_mask,
                navigation_tokens=prepared.policy_context.navigation_tokens,
                navigation_padding_mask=prepared.policy_context.navigation_padding_mask,
                reference_trajectory=prepared.policy_context.reference_trajectory,
            )
            value, policy_timing = profile_call(
                self.device,
                profile,
                lambda: (
                    self._policy.forward_tensordict(policy_context_tensordict(context))[
                        "state_value"
                    ]
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
    """Load the frozen planner and an exploration policy through one single-device Fabric."""

    if type(policy_action_seed) is not int or policy_action_seed < 0:
        raise ValueError("policy_action_seed must be a non-negative integer")
    fabric, report = create_single_device_fabric(runtime_config)
    planner, checkpoint_report = load_official_diffusion_planner(
        args_path, checkpoint_path, sampler_config, guidance_config
    )
    if planner_compile_mode == "dit_reduce_overhead":
        if fabric.device.type != "cuda":
            raise ValueError("dit_reduce_overhead requires a CUDA rollout runtime")
        planner.model.decoder.dit.forward = torch.compile(
            planner.model.decoder.dit.forward,
            mode="reduce-overhead",
            fullgraph=True,
            dynamic=False,
        )
    policy = ExplorationPolicy(policy_config)
    wrapped_planner = fabric.setup_module(planner)
    planner = _unwrap_diffusion_planner(wrapped_planner)
    wrapped_policy = fabric.setup_module(policy)
    policy = _unwrap_exploration_policy(wrapped_policy)
    if report.world_size != 1:
        raise RuntimeError("rollout runtime requires Fabric world_size=1")
    return FabricRolloutRuntime(
        fabric,
        planner,
        policy,
        report,
        noise_seed=report.seed,
        policy_action_seed=policy_action_seed,
        checkpoint_report=checkpoint_report,
        sampler=sampler_report(sampler_config),
        guidance_config=guidance_config,
        planner_compile_mode=planner_compile_mode,
    )


def _unwrap_exploration_policy(module: torch.nn.Module) -> ExplorationPolicy:
    if isinstance(module, ExplorationPolicy):
        return module
    unwrapped = getattr(module, "module", None)
    if not isinstance(unwrapped, ExplorationPolicy):
        raise TypeError("Fabric did not preserve the ExplorationPolicy module")
    return unwrapped


def _unwrap_diffusion_planner(module: torch.nn.Module) -> PretrainedDiffusionPlanner:
    if isinstance(module, PretrainedDiffusionPlanner):
        return module
    unwrapped = getattr(module, "module", None)
    if not isinstance(unwrapped, PretrainedDiffusionPlanner):
        raise TypeError("Fabric did not preserve the PretrainedDiffusionPlanner module")
    return unwrapped


def _fabric_observation(value: object) -> TensorDictBase:
    """Validate one Fabric observation transfer at the third-party boundary."""

    if not isinstance(value, TensorDictBase):
        raise TypeError("Fabric must preserve the rollout TensorDict observation container")
    return value


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


def _sample_policy_actions(
    output: ExplorationPolicyOutput, generators: Sequence[torch.Generator]
) -> AffineBetaAction:
    parameters = output.distribution.parameters
    base_action = torch.cat(
        [
            ExplicitGeneratorBetaSampler.draw(
                AffineBetaParameters(
                    alpha=parameters.alpha[index : index + 1],
                    beta=parameters.beta[index : index + 1],
                ),
                generator,
                validate_args=False,
            )
            for index, generator in enumerate(generators)
        ]
    )
    return output.distribution.evaluate_base_action(base_action)


def _rng_state(generator: torch.Generator) -> torch.Tensor:
    return generator.get_state().detach().cpu().clone()


def _seed(value: int, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value
