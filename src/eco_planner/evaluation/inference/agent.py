"""Thin planner adapters consumed by the shared closed-loop evaluation engine."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
from tensordict import TensorDictBase

from eco_planner.envs.array_types import BatchObservation
from eco_planner.models import (
    CheckpointLoadReport,
    GuidanceConfig,
    OfficialDiffusionPlannerConfig,
    SamplerReport,
)
from eco_planner.runtime.fabric import InferenceRuntimeReport

from .decision import InferenceDecision
from .runtime import FabricInferenceRuntime


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
    def guided(self) -> bool: ...

    def new_noise_generator(self, scenario_index: int) -> torch.Generator: ...

    def noise_seed(self, scenario_index: int) -> int: ...

    def decide_batch(
        self, observation: BatchObservation, generators: Sequence[torch.Generator]
    ) -> EvaluationDecision: ...


@dataclass(frozen=True)
class DiffusionEvaluationAgent:
    """Expose base and fixed-guidance diffusion planners to the common engine."""

    runtime: FabricInferenceRuntime

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
        return self.runtime.guidance_config.name != "none"

    def new_noise_generator(self, scenario_index: int) -> torch.Generator:
        return self.runtime.new_noise_generator()

    def noise_seed(self, scenario_index: int) -> int:
        return self.runtime.report.seed

    def decide_batch(
        self, observation: BatchObservation, generators: Sequence[torch.Generator]
    ) -> InferenceDecision:
        if len(generators) == 1:
            return self.runtime.infer(observation, generators[0])
        noise = self.runtime.sample_noise(generators)
        return self.runtime.infer_batch(observation, noise, generators)
