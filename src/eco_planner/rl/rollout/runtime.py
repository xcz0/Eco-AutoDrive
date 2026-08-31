"""Fabric-owned learned-guidance rollout inference with one host transfer per decision."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Literal, TypeVar, cast

import numpy as np
import torch
from lightning.fabric import Fabric
from tensordict import TensorDict, TensorDictBase

from eco_planner.envs.array_types import BatchObservation
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
    validate_exploration_policy_context,
)
from eco_planner.rl.policy.distribution import (
    AffineBetaAction,
    AffineBetaParameters,
    ExplicitGeneratorBetaSampler,
)
from eco_planner.rl.rollout.contracts import DecisionAudit, build_training_decision
from eco_planner.runtime.config import RuntimeConfig
from eco_planner.runtime.contracts import HostExecutionResult
from eco_planner.runtime.fabric import InferenceRuntimeReport, create_single_device_fabric
from eco_planner.runtime.host_transfer import (
    DeferredHostTensors,
    DeferredHostTransferTiming,
    copy_execution_trajectory,
    defer_host_tensors,
)


@dataclass(frozen=True)
class RolloutPlannerPhaseTiming:
    """Host call wall and accelerator work for one profiled planner phase."""

    host_call_wall_s: float
    accelerator_s: float


@dataclass(frozen=True)
class RolloutPlannerTiming:
    """Profile-only timing for one decision or bootstrap planner batch."""

    phase: Literal["decision", "bootstrap"]
    host_to_device: RolloutPlannerPhaseTiming
    diffusion_noise: RolloutPlannerPhaseTiming
    prepare_policy_guidance: RolloutPlannerPhaseTiming
    policy_forward: RolloutPlannerPhaseTiming
    action_sampling: RolloutPlannerPhaseTiming | None
    complete_policy_guidance: RolloutPlannerPhaseTiming | None
    guidance_action_check: RolloutPlannerPhaseTiming | None
    execution_to_host: RolloutPlannerPhaseTiming | None
    profile_sync_wait_wall_s: float


@dataclass(frozen=True)
class _PendingPhaseTiming:
    host_call_wall_s: float
    start_event: torch.cuda.Event | None
    end_event: torch.cuda.Event | None

    def resolve(self) -> RolloutPlannerPhaseTiming:
        accelerator_s = self.host_call_wall_s
        if self.start_event is not None and self.end_event is not None:
            accelerator_s = self.start_event.elapsed_time(self.end_event) / 1000.0
        return RolloutPlannerPhaseTiming(self.host_call_wall_s, accelerator_s)


_T = TypeVar("_T")


class RolloutDecision:
    """Execution trajectory plus deferred CPU rollout storage fields."""

    def __init__(
        self,
        execution: HostExecutionResult,
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
        return self._execution.ego_trajectory[0]

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
        execution: HostExecutionResult,
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
        self._slots: list[RolloutDecision | None] = [None] * execution.ego_trajectory.shape[0]

    @property
    def ego_trajectories(self) -> np.ndarray:
        """Return executable trajectories with shape ``[B, T, 4]``."""

        return self._execution.ego_trajectory

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
                HostExecutionResult(self.ego_trajectories[index : index + 1]),
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
            _validate_rollout_context(
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
        observation: BatchObservation | Mapping[str, torch.Tensor],
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
        observation: BatchObservation | Mapping[str, torch.Tensor],
        diffusion_generators: Sequence[torch.Generator],
        policy_generators: Sequence[torch.Generator],
        *,
        timings: list[RolloutPlannerTiming] | None = None,
    ) -> BatchRolloutDecision:
        """Sample batched guidance actions with one independent RNG stream per slot."""

        raw_observation = cast(dict[str, torch.Tensor], dict(observation))
        batch = _observation_batch_size(raw_observation)
        _validate_slot_generators(diffusion_generators, batch, "diffusion_generators")
        _validate_slot_generators(policy_generators, batch, "policy_generators")
        planner_config = self._planner.config
        profile = timings is not None
        moved, h2d_timing = _profile_call(
            self.device, profile, lambda: self._fabric.to_device(raw_observation)
        )
        moved = _fabric_observation(moved)
        diffusion_rng_states = tuple(_rng_state(generator) for generator in diffusion_generators)
        policy_rng_states = tuple(_rng_state(generator) for generator in policy_generators)
        noise, noise_timing = _profile_call(
            self.device,
            profile,
            lambda: _slot_noise(planner_config, self.device, diffusion_generators),
        )
        with torch.no_grad():
            prepared, prepare_timing = _profile_call(
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
            policy_outputs, policy_timing = _profile_call(
                self.device,
                profile,
                lambda: self._policy.forward_tensordict(policy_context_tensordict(policy_context)),
            )
            output = self._policy.output_from_tensordict(policy_outputs)
            action, action_timing = _profile_call(
                self.device,
                profile,
                lambda: _sample_policy_actions(output, policy_generators),
            )
        with torch.enable_grad():
            result, complete_timing = _profile_call(
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
        deferred = defer_host_tensors(
            {
                "prediction": (result.prediction, torch.float32),
                "initial_noise": (noise, torch.float32),
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
            self.device,
            profile=profile,
        )
        execution, execution_timing = _profile_call(
            self.device,
            profile,
            lambda: copy_execution_trajectory(result.prediction, self.device),
        )
        sync_wait_s = _finish_profile(self.device, profile)
        if timings is not None:
            timings.append(
                RolloutPlannerTiming(
                    phase="decision",
                    host_to_device=_require_phase(h2d_timing),
                    diffusion_noise=_require_phase(noise_timing),
                    prepare_policy_guidance=_require_phase(prepare_timing),
                    policy_forward=_require_phase(policy_timing),
                    action_sampling=_require_phase(action_timing),
                    complete_policy_guidance=_require_phase(complete_timing),
                    guidance_action_check=None,
                    execution_to_host=_require_phase(execution_timing),
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
        observation: BatchObservation | Mapping[str, torch.Tensor],
        diffusion_generator: torch.Generator,
    ) -> torch.Tensor:
        """Evaluate one old critic value through the shared batched runtime path."""

        return self.bootstrap_value_batch(observation, (diffusion_generator,))

    def bootstrap_value_batch(
        self,
        observation: BatchObservation | Mapping[str, torch.Tensor],
        diffusion_generators: Sequence[torch.Generator],
        *,
        timings: list[RolloutPlannerTiming] | None = None,
    ) -> torch.Tensor:
        """Evaluate old critic values for a batch without sampling or executing actions."""

        raw_observation = cast(dict[str, torch.Tensor], dict(observation))
        batch = _observation_batch_size(raw_observation)
        _validate_slot_generators(diffusion_generators, batch, "diffusion_generators")
        profile = timings is not None
        moved, h2d_timing = _profile_call(
            self.device, profile, lambda: self._fabric.to_device(raw_observation)
        )
        moved = _fabric_observation(moved)
        noise, noise_timing = _profile_call(
            self.device,
            profile,
            lambda: _slot_noise(self._planner.config, self.device, diffusion_generators),
        )
        with torch.no_grad():
            prepared, prepare_timing = _profile_call(
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
            value, policy_timing = _profile_call(
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
        sync_wait_s = _finish_profile(self.device, profile)
        if timings is not None:
            timings.append(
                RolloutPlannerTiming(
                    phase="bootstrap",
                    host_to_device=_require_phase(h2d_timing),
                    diffusion_noise=_require_phase(noise_timing),
                    prepare_policy_guidance=_require_phase(prepare_timing),
                    policy_forward=_require_phase(policy_timing),
                    action_sampling=None,
                    complete_policy_guidance=None,
                    guidance_action_check=None,
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


def _validate_finite(tensors: Mapping[str, torch.Tensor]) -> None:
    for name, value in tensors.items():
        if value.dtype.is_floating_point and not torch.isfinite(value).all():
            raise RuntimeError(f"rollout host tensor {name!r} contains non-finite values")


def _fabric_observation(value: object) -> dict[str, torch.Tensor]:
    """Validate one Fabric observation transfer at the third-party boundary."""

    if not isinstance(value, dict) or not all(
        isinstance(name, str) and isinstance(tensor, torch.Tensor)
        for name, tensor in value.items()
    ):
        raise TypeError("Fabric must move rollout observations as a string-to-tensor mapping")
    return cast(dict[str, torch.Tensor], value)


def _slice_tensordict(value: TensorDictBase, index: slice) -> TensorDictBase:
    return cast(TensorDictBase, value[index])


def _validate_rollout_context(
    context: ExplorationPolicyContext, config: ExplorationPolicyConfig
) -> None:
    validate_exploration_policy_context(context, config)


def _observation_batch_size(observation: Mapping[str, torch.Tensor]) -> int:
    value = observation.get("ego_current_state")
    if not isinstance(value, torch.Tensor) or value.ndim < 1 or value.shape[0] <= 0:
        raise ValueError(
            "rollout observation must have a positive ego_current_state batch dimension"
        )
    return value.shape[0]


def _validate_slot_generators(generators: Sequence[torch.Generator], batch: int, name: str) -> None:
    if len(generators) != batch:
        raise ValueError(f"{name} must contain one generator per batch item")


def _slot_noise(
    config: OfficialDiffusionPlannerConfig,
    device: torch.device,
    generators: Sequence[torch.Generator],
) -> torch.Tensor:
    shape = (1, 1 + config.predicted_neighbor_num, config.future_len, 4)
    return torch.cat(
        [
            torch.randn(shape, dtype=torch.float32, device=device, generator=generator)
            for generator in generators
        ]
    )


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


def _profile_call(
    device: torch.device,
    enabled: bool,
    operation: Callable[[], _T],
) -> tuple[_T, _PendingPhaseTiming | None]:
    if not enabled:
        return operation(), None
    start_event: torch.cuda.Event | None = None
    end_event: torch.cuda.Event | None = None
    if device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record(torch.cuda.current_stream(device))
    started = perf_counter()
    result = operation()
    host_call_wall_s = perf_counter() - started
    if end_event is not None:
        end_event.record(torch.cuda.current_stream(device))
    return result, _PendingPhaseTiming(host_call_wall_s, start_event, end_event)


def _finish_profile(device: torch.device, enabled: bool) -> float:
    if not enabled or device.type != "cuda":
        return 0.0
    started = perf_counter()
    torch.cuda.current_stream(device).synchronize()
    return perf_counter() - started


def _require_phase(timing: _PendingPhaseTiming | None) -> RolloutPlannerPhaseTiming:
    if timing is None:
        raise RuntimeError("profiled rollout phase did not return timing")
    return timing.resolve()
