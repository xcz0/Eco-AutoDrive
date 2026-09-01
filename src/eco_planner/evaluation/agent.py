"""Thin planner adapters consumed by the shared closed-loop evaluation engine."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
from tensordict import TensorDictBase

from eco_planner.envs.array_types import BatchObservation
from eco_planner.evaluation.runtime import FabricInferenceRuntime, InferenceDecision
from eco_planner.models import (
    CheckpointLoadReport,
    GuidanceConfig,
    OfficialDiffusionPlannerConfig,
    SamplerReport,
)
from eco_planner.rl.rollout import FabricRolloutRuntime
from eco_planner.runtime.fabric import InferenceRuntimeReport


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

    def new_episode(self, scenario_index: int) -> object: ...

    def noise_seed(self, scenario_index: int) -> int: ...

    def decide_batch(
        self, observation: BatchObservation, episodes: Sequence[object]
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

    def new_episode(self, scenario_index: int) -> object:
        return self.runtime.new_noise_generator()

    def noise_seed(self, scenario_index: int) -> int:
        return self.runtime.report.seed

    def decide_batch(
        self, observation: BatchObservation, episodes: Sequence[object]
    ) -> InferenceDecision:
        generators = tuple(_noise_generator(item) for item in episodes)
        if len(generators) == 1:
            return self.runtime.infer(observation, generators[0])
        noise = self.runtime.sample_noise(generators)
        return self.runtime.infer_batch(observation, noise, generators)


@dataclass(frozen=True)
class PPOCheckpointEvaluationAgent:
    """Evaluate a PPO guidance checkpoint with deterministic policy-mean actions."""

    runtime: FabricRolloutRuntime
    noise_seeds: tuple[int, ...]

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

    def new_episode(self, scenario_index: int) -> object:
        return self.runtime.new_noise_generator(self.noise_seed(scenario_index))

    def noise_seed(self, scenario_index: int) -> int:
        return self.noise_seeds[scenario_index]

    def decide_batch(
        self, observation: BatchObservation, episodes: Sequence[object]
    ) -> EvaluationDecision:
        return self.runtime.decide_batch_mean(
            observation,
            tuple(_noise_generator(item) for item in episodes),
        )


def _noise_generator(value: object) -> torch.Generator:
    if not isinstance(value, torch.Generator):
        raise TypeError("evaluation agent episode state must be a torch.Generator")
    return value
