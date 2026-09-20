"""Planning-owned single learned-guidance decision composition."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeVar

import torch
from tensordict import TensorDictBase

from eco_planner.planning.diffusion import PretrainedDiffusionPlanner
from eco_planner.planning.policy import (
    ExplorationPolicy,
    ExplorationPolicyOutput,
    build_policy_inputs,
    policy_context_tensordict,
)
from eco_planner.planning.policy.distribution import (
    AffineBetaAction,
    AffineBetaParameters,
    ExplicitGeneratorBetaSampler,
)
from eco_planner.planning.result import DecisionResult, GuidanceAction, PolicyDecision
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
    ) -> DecisionResult:
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
        return DecisionResult(
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
