"""Typed manifest for the A0/A1/A2 matched scalar-reward protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from eco_planner._repository import CONFIG_ROOT
from eco_planner.configuration import ScenarioConfig, load_resolved_yaml_mapping

DEFAULT_PROTOCOL = CONFIG_ROOT / "experiments" / "scalar_reward" / "protocol.yaml"


class _StrictModel(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)


class TrainingProtocolConfig(_StrictModel):
    base_job: str = Field(min_length=1)
    maps: list[str] = Field(min_length=1)
    map_seeds: list[StrictInt] = Field(min_length=1)
    seeds: list[StrictInt] = Field(min_length=1)


class EvaluationProtocolConfig(_StrictModel):
    job: str = Field(min_length=1)
    policy_job: str = Field(min_length=1)
    maps: list[str] = Field(min_length=1)
    map_seeds: list[StrictInt] = Field(min_length=1)
    seed: StrictInt = Field(ge=0)
    horizon_steps: StrictInt = Field(gt=0)
    sampler: Literal["ddim5"]


class FrozenPlannerArmConfig(_StrictModel):
    label: Literal["frozen_diffusion_planner"]


class TrainedPolicyArmConfig(_StrictModel):
    label: str = Field(min_length=1)
    reward_profile: Literal["plannerrft_no_energy_v1", "plannerrft_energy_v1"]


class ArmsConfig(_StrictModel):
    a0: FrozenPlannerArmConfig
    a1: TrainedPolicyArmConfig
    a2: TrainedPolicyArmConfig


class ScalarRewardProtocolConfig(_StrictModel):
    version: Literal[1]
    study_name: str = Field(min_length=1)
    training: TrainingProtocolConfig
    arms: ArmsConfig
    evaluation: EvaluationProtocolConfig
    update0_evaluation: Literal["diagnostic"]

    @model_validator(mode="after")
    def validate_disjoint_pools(self) -> ScalarRewardProtocolConfig:
        overlap = self.training_pairs() & self.held_out_pairs()
        if overlap:
            raise ValueError(f"training and held-out scenario pools overlap: {sorted(overlap)}")
        return self

    def training_pairs(self) -> frozenset[tuple[str, int]]:
        return _pairs(self.training.maps, self.training.map_seeds)

    def held_out_pairs(self) -> frozenset[tuple[str, int]]:
        return _pairs(self.evaluation.maps, self.evaluation.map_seeds)

    def training_scenarios(self) -> tuple[ScenarioConfig, ...]:
        return _scenarios(self.training.maps, self.training.map_seeds)

    def held_out_scenarios(self) -> tuple[ScenarioConfig, ...]:
        return _scenarios(self.evaluation.maps, self.evaluation.map_seeds)


def _pairs(maps: list[str], seeds: list[int]) -> frozenset[tuple[str, int]]:
    return frozenset((map_name, seed) for map_name in maps for seed in seeds)


def _scenarios(maps: list[str], seeds: list[int]) -> tuple[ScenarioConfig, ...]:
    return tuple(
        ScenarioConfig(name=f"{map_name.lower()}_s{seed}", map=map_name, seed=seed)
        for map_name in maps
        for seed in seeds
    )


def load_scalar_reward_protocol(path: Path) -> ScalarRewardProtocolConfig:
    """Load one explicit matched-protocol manifest."""

    return ScalarRewardProtocolConfig.model_validate(load_resolved_yaml_mapping(path))
