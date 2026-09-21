"""Shared matched training and held-out protocol with explicit reward arms."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from eco_planner._repository import CONFIG_ROOT
from eco_planner.analysis.statistics import ScenarioBootstrapConfig
from eco_planner.configuration import ScenarioConfig, load_resolved_yaml_mapping
from eco_planner.reward.result import RewardProfileName

DEFAULT_PROTOCOL = CONFIG_ROOT / "experiments" / "comparison" / "default.yaml"


class _StrictModel(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)


class TrainingProtocolConfig(_StrictModel):
    base_job: str = Field(min_length=1)
    maps: list[str] = Field(min_length=1)
    map_seeds: list[StrictInt] = Field(min_length=1)
    seeds: list[StrictInt] = Field(min_length=1)
    overrides: list[str]


class EvaluationProtocolConfig(_StrictModel):
    job: str = Field(min_length=1)
    policy_job: str = Field(min_length=1)
    maps: list[str] = Field(min_length=1)
    map_seeds: list[StrictInt] = Field(min_length=1)
    seed: StrictInt = Field(ge=0)
    horizon_steps: StrictInt = Field(gt=0)
    sampler: Literal["ddim5"]


class TrainedPolicyArmConfig(_StrictModel):
    label: str = Field(min_length=1)
    reward_profile: RewardProfileName | None


class ComparisonProtocol(_StrictModel):
    version: Literal[1]
    study_name: str = Field(min_length=1)
    training: TrainingProtocolConfig
    arms: dict[str, TrainedPolicyArmConfig]
    contrasts: list[list[str]]
    evaluation: EvaluationProtocolConfig
    bootstrap: ScenarioBootstrapConfig

    @model_validator(mode="after")
    def validate_disjoint_pools(self) -> ComparisonProtocol:
        if not self.arms or sum(a.reward_profile is None for a in self.arms.values()) > 1:
            raise ValueError("protocol requires arms and at most one frozen planner")
        if not self.contrasts or any(
            len(pair) != 2 or pair[0] == pair[1] or any(a not in self.arms for a in pair)
            for pair in self.contrasts
        ):
            raise ValueError("contrasts require two distinct declared arms")
        overlap = self.training_pairs() & self.held_out_pairs()
        if overlap:
            raise ValueError(f"training and held-out scenario pools overlap: {sorted(overlap)}")
        return self

    @property
    def frozen_arm(self) -> str | None:
        return next((name for name, arm in self.arms.items() if arm.reward_profile is None), None)

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


def load_protocol(path: Path) -> ComparisonProtocol:
    """Load one explicit matched-protocol manifest."""

    return ComparisonProtocol.model_validate(load_resolved_yaml_mapping(path))
