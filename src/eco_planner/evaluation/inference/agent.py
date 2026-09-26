"""Thin planner adapters consumed by the shared closed-loop evaluation engine."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np
import torch
from tensordict import TensorDictBase

from eco_planner.planning import DiffusionRuntime, PolicyGuidanceRuntime
from eco_planner.planning.diffusion import (
    CheckpointLoadReport,
    GuidanceConfig,
    OfficialDiffusionPlannerConfig,
    SamplerReport,
)
from eco_planner.runtime.fabric import InferenceRuntimeReport
from eco_planner.runtime.host_transfer import HostTransfer

from ..artifacts.models import PolicyCheckpointProvenance
from .decision import (
    InferenceDecision,
    prepare_diffusion_inference_decision,
    prepare_learned_inference_decision,
    validate_artifact_observation_fields,
)


class EvaluationDecision(Protocol):
    """One batched planner decision containing execution and trace audit data."""

    @property
    def ego_trajectories(self) -> np.ndarray: ...

    def audit_result(self) -> TensorDictBase: ...


class EvaluationAgent(Protocol):
    """Adapter boundary between a planner variant and generic environment execution."""

    @property
    def planner_config(self) -> OfficialDiffusionPlannerConfig: ...

    @property
    def report(self) -> InferenceRuntimeReport: ...

    @property
    def checkpoint_report(self) -> CheckpointLoadReport: ...

    @property
    def sampler_report(self) -> SamplerReport: ...

    @property
    def guidance_config(self) -> GuidanceConfig: ...

    @property
    def policy_checkpoint(self) -> PolicyCheckpointProvenance | None: ...

    @property
    def guided(self) -> bool: ...

    def new_noise_generator(self, scenario_index: int) -> torch.Generator: ...

    def noise_seed(self, scenario_index: int) -> int: ...

    def new_policy_generator(self, scenario_index: int) -> torch.Generator | None: ...

    def decide_batch(
        self,
        observation: TensorDictBase,
        generators: Sequence[torch.Generator],
        *,
        policy_generators: Sequence[torch.Generator] | None = None,
    ) -> EvaluationDecision: ...


@dataclass(frozen=True)
class DiffusionEvaluationAgent:
    """Expose base and fixed-guidance diffusion planners to the common engine."""

    runtime: DiffusionRuntime
    _host_transfer: HostTransfer = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_host_transfer", HostTransfer(self.runtime.device))

    @property
    def planner_config(self) -> OfficialDiffusionPlannerConfig:
        return self.runtime.planner_config

    @property
    def report(self) -> InferenceRuntimeReport:
        return self.runtime.report

    @property
    def checkpoint_report(self) -> CheckpointLoadReport:
        return self.runtime.checkpoint_report

    @property
    def sampler_report(self) -> SamplerReport:
        return self.runtime.sampler_report

    @property
    def guidance_config(self) -> GuidanceConfig:
        return self.runtime.guidance_config

    @property
    def policy_checkpoint(self) -> PolicyCheckpointProvenance | None:
        return None

    @property
    def guided(self) -> bool:
        return self.runtime.guidance_config.name != "none"

    def new_noise_generator(self, scenario_index: int) -> torch.Generator:
        return self.runtime.new_noise_generator()

    def noise_seed(self, scenario_index: int) -> int:
        return self.runtime.report.seed

    def new_policy_generator(self, scenario_index: int) -> torch.Generator | None:
        return None

    def decide_batch(
        self,
        observation: TensorDictBase,
        generators: Sequence[torch.Generator],
        *,
        policy_generators: Sequence[torch.Generator] | None = None,
    ) -> InferenceDecision:
        if policy_generators is not None:
            raise ValueError("diffusion evaluation does not accept policy generators")
        noise = self.runtime.sample_noise(generators)
        return self.infer_batch(observation, noise, generators)

    def infer_batch(
        self,
        observation: TensorDictBase,
        standard_normal_noise: torch.Tensor,
        transition_generators: Sequence[torch.Generator | None],
        *,
        profile: bool = False,
        guidance_action: torch.Tensor | None = None,
    ) -> InferenceDecision:
        """Adapt explicit-noise diagnostic and benchmark calls to evaluation host fields."""

        validate_artifact_observation_fields(observation, self.planner_config)
        result = self.runtime.decide_batch(
            observation,
            standard_normal_noise,
            transition_generators,
            profile=profile,
            guidance_action=guidance_action,
        )
        return prepare_diffusion_inference_decision(result, self._host_transfer)


@dataclass(frozen=True)
class PolicyCheckpointEvaluationAgent:
    """Adapt one exploration-policy checkpoint to the generic evaluation engine."""

    runtime: PolicyGuidanceRuntime
    noise_seeds: tuple[int, ...]
    policy_checkpoint: PolicyCheckpointProvenance
    action_mode: Literal["mean", "sample"]
    policy_action_seeds: tuple[int, ...]
    _host_transfer: HostTransfer = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_host_transfer", HostTransfer(self.runtime.device))
        if self.action_mode == "sample":
            if not self.policy_action_seeds:
                raise ValueError(
                    "sample-mode evaluation requires one policy action seed per scenario"
                )
            if len(self.policy_action_seeds) != len(self.noise_seeds):
                raise ValueError("policy action seeds must align with the noise seed per scenario")
        elif self.policy_action_seeds:
            raise ValueError("mean-mode evaluation must not configure policy action seeds")

    @property
    def planner_config(self) -> OfficialDiffusionPlannerConfig:
        return self.runtime.planner_config

    @property
    def report(self) -> InferenceRuntimeReport:
        return self.runtime.report

    @property
    def checkpoint_report(self) -> CheckpointLoadReport:
        return self.runtime.checkpoint_report

    @property
    def sampler_report(self) -> SamplerReport:
        return self.runtime.sampler_report

    @property
    def guidance_config(self) -> GuidanceConfig:
        return self.runtime.guidance_config

    @property
    def guided(self) -> bool:
        return True

    def new_noise_generator(self, scenario_index: int) -> torch.Generator:
        return self.runtime.new_noise_generator(self.noise_seed(scenario_index))

    def noise_seed(self, scenario_index: int) -> int:
        return self.noise_seeds[scenario_index]

    def new_policy_generator(self, scenario_index: int) -> torch.Generator | None:
        if self.action_mode == "mean":
            return None
        return self.runtime.new_policy_generator(self.policy_action_seeds[scenario_index])

    def decide_batch(
        self,
        observation: TensorDictBase,
        generators: Sequence[torch.Generator],
        *,
        policy_generators: Sequence[torch.Generator] | None = None,
    ) -> EvaluationDecision:
        """Evaluate deterministic Beta-mean or seeded stochastic Beta actions."""

        if self.action_mode == "mean":
            if policy_generators is not None:
                raise ValueError("mean-mode evaluation does not accept policy generators")
            result = self.runtime.decide_batch_mean(observation, tuple(generators))
        else:
            if policy_generators is None:
                raise ValueError("sample-mode evaluation requires policy generators")
            result = self.runtime.decide_batch(
                observation, tuple(generators), tuple(policy_generators)
            )
        return prepare_learned_inference_decision(result, self._host_transfer)
