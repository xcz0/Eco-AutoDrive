"""Planning-owned single learned-guidance decision composition."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal, TypeVar

import torch
from lightning.fabric import Fabric
from tensordict import TensorDictBase

from eco_planner.planning.diffusion import (
    CheckpointLoadReport,
    OfficialDiffusionPlannerConfig,
    OrthogonalPolicyGuidanceConfig,
    PretrainedDiffusionPlanner,
    SamplerConfig,
    SamplerReport,
    load_official_diffusion_planner,
    sampler_report,
)
from eco_planner.planning.policy import (
    ExplorationPolicy,
    ExplorationPolicyConfig,
    ExplorationPolicyOutput,
    build_policy_inputs,
    policy_context_tensordict,
)
from eco_planner.planning.policy.distribution import (
    AffineBetaAction,
    AffineBetaParameters,
    ExplicitGeneratorBetaSampler,
)
from eco_planner.planning.result import (
    GuidanceAction,
    PolicyDecision,
    PolicyGuidanceDecisionResult,
)
from eco_planner.runtime.config import RuntimeConfig
from eco_planner.runtime.fabric import InferenceRuntimeReport, create_single_device_fabric
from eco_planner.runtime.profiling import PhaseProfiler
from eco_planner.runtime.random import sample_batched_standard_normal

_T = TypeVar("_T")


class PlanningInference:
    """Compose diffusion reference preparation and the trainable guidance policy."""

    def __init__(
        self,
        planner: PretrainedDiffusionPlanner,
        policy: ExplorationPolicy,
        device: torch.device,
        to_device: Callable[[TensorDictBase], TensorDictBase],
    ) -> None:
        self._planner = planner
        self._policy = policy
        self._device = device
        self._to_device = to_device

    def decide_batch(
        self,
        observation: TensorDictBase,
        diffusion_generators: Sequence[torch.Generator],
        policy_generators: Sequence[torch.Generator] | None,
        *,
        profiler: PhaseProfiler | None = None,
    ) -> PolicyGuidanceDecisionResult:
        """Sample batched guidance actions with one independent RNG stream per slot.

        ``policy_generators=None`` evaluates deterministic Beta-mean actions and
        consumes no policy RNG, but still draws diffusion noise and runs the
        reference pass.
        """

        batch = _observation_batch_size(observation)
        _validate_slot_generators(diffusion_generators, batch, "diffusion_generators")
        if policy_generators is not None:
            _validate_slot_generators(policy_generators, batch, "policy_generators")
        moved = _fabric_observation(
            _measure(profiler, "host_to_device", lambda: self._to_device(observation))
        )
        noise = _measure(
            profiler,
            "diffusion_noise",
            lambda: sample_batched_standard_normal(
                diffusion_generators,
                (
                    1 + self._planner.config.predicted_neighbor_num,
                    self._planner.config.future_len,
                    4,
                ),
                device=self._device,
            ),
        )
        with torch.no_grad():
            prepared = _measure(
                profiler,
                "prepare_policy_guidance",
                lambda: self._planner.prepare_policy_guidance(moved, noise, diffusion_generators),
            )
            policy_context = build_policy_inputs(
                prepared.representations, prepared.reference_prediction
            )
            policy_outputs = _measure(
                profiler,
                "policy_forward",
                lambda: self._policy.forward_tensordict(policy_context_tensordict(policy_context)),
            )
            output = self._policy.output_from_tensordict(policy_outputs)
            action = _measure(
                profiler,
                "action_sampling",
                lambda: (
                    _sample_policy_actions(output, policy_generators)
                    if policy_generators is not None
                    else output.distribution.action_mean()
                ),
            )
        with torch.enable_grad():
            result = _measure(
                profiler,
                "complete_policy_guidance",
                lambda: self._planner.complete_policy_guidance(prepared, action.guidance_action),
            )
        if result.reference_prediction is None or result.guidance_diagnostics is None:
            raise RuntimeError(
                "policy guidance planner result is missing required trace diagnostics"
            )
        return PolicyGuidanceDecisionResult(
            prediction=result.prediction,
            initial_noise=noise,
            reference_prediction=result.reference_prediction,
            policy=PolicyDecision(
                inputs=policy_context,
                output=output,
                action=GuidanceAction(
                    base_action=action.base_action,
                    guidance_action=action.guidance_action,
                    joint_log_prob=action.joint_guidance_log_prob,
                ),
            ),
            guidance_diagnostics=result.guidance_diagnostics,
        )


class PolicyGuidanceRuntime:
    """Own the frozen planner, trainable policy, and one Fabric device owner.

    Both training rollout and evaluation reuse this runtime to run the same
    learned-guidance decision; consumers only adapt its typed decision result.
    """

    planner_compile_mode: Literal["eager", "dit_reduce_overhead"]

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
        self._inference = PlanningInference(planner, policy, fabric.device, fabric.to_device)

    @property
    def device(self) -> torch.device:
        return self._fabric.device

    @property
    def fabric(self) -> Fabric:
        """Expose the single-device Fabric owner for checkpoint persistence."""

        return self._fabric

    @property
    def planner(self) -> PretrainedDiffusionPlanner:
        """Expose the frozen planner to inference-time consumers such as bootstrap."""

        return self._planner

    @property
    def policy(self) -> ExplorationPolicy:
        """Expose the single trainable parameter owner to the PPO updater."""

        return self._policy

    @property
    def planner_config(self) -> OfficialDiffusionPlannerConfig:
        """Expose the immutable planner architecture for observation adapters."""

        return self._planner.config

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

    def new_noise_generator(self, seed: int) -> torch.Generator:
        return torch.Generator(device=self.device).manual_seed(_seed(seed, "noise seed"))

    def new_policy_generator(self, seed: int) -> torch.Generator:
        return torch.Generator(device=self.device).manual_seed(_seed(seed, "policy action seed"))

    def decide(
        self,
        observation: TensorDictBase,
        diffusion_generator: torch.Generator,
        policy_generator: torch.Generator | None,
    ) -> PolicyGuidanceDecisionResult:
        """Run one single-slot learned-guidance decision."""

        policy_generators = None if policy_generator is None else (policy_generator,)
        return self.decide_batch(observation, (diffusion_generator,), policy_generators)

    def decide_batch(
        self,
        observation: TensorDictBase,
        diffusion_generators: Sequence[torch.Generator],
        policy_generators: Sequence[torch.Generator] | None,
        *,
        profiler: PhaseProfiler | None = None,
    ) -> PolicyGuidanceDecisionResult:
        """Sample batched guidance actions with one independent RNG stream per slot."""

        return self._inference.decide_batch(
            observation, diffusion_generators, policy_generators, profiler=profiler
        )

    def decide_batch_mean(
        self,
        observation: TensorDictBase,
        diffusion_generators: Sequence[torch.Generator],
        *,
        profiler: PhaseProfiler | None = None,
    ) -> PolicyGuidanceDecisionResult:
        """Evaluate deterministic Beta-mean actions without consuming policy RNG."""

        return self._inference.decide_batch(
            observation, diffusion_generators, None, profiler=profiler
        )


def create_policy_guidance_runtime(
    runtime_config: RuntimeConfig,
    sampler_config: SamplerConfig,
    guidance_config: OrthogonalPolicyGuidanceConfig,
    policy_config: ExplorationPolicyConfig,
    args_path: Path,
    checkpoint_path: Path,
    policy_action_seed: int,
    *,
    planner_compile_mode: Literal["eager", "dit_reduce_overhead"],
) -> PolicyGuidanceRuntime:
    """Load the frozen planner and an exploration policy through one single-device Fabric."""

    if type(policy_action_seed) is not int or policy_action_seed < 0:
        raise ValueError("policy_action_seed must be a non-negative integer")
    fabric, report = create_single_device_fabric(runtime_config)
    planner, checkpoint_report = load_official_diffusion_planner(
        args_path, checkpoint_path, sampler_config, guidance_config
    )
    if planner_compile_mode == "dit_reduce_overhead":
        if fabric.device.type != "cuda":
            raise ValueError("dit_reduce_overhead requires a CUDA policy guidance runtime")
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
        raise RuntimeError("policy guidance runtime requires Fabric world_size=1")
    return PolicyGuidanceRuntime(
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


def _measure(profiler: PhaseProfiler | None, name: str, operation: Callable[[], _T]) -> _T:
    if profiler is None:
        return operation()
    return profiler.measure(name, operation)


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


def _seed(value: int, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value
