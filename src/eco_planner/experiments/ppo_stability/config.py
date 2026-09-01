"""Configuration models and scenario construction for PPO stability."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, model_validator

from eco_planner._repository import CONFIG_ROOT
from eco_planner.configuration import load_resolved_yaml_mapping
from eco_planner.evaluation.config import ScenarioConfig

DEFAULT_STUDY = CONFIG_ROOT / "studies" / "ppo" / "stability.yaml"


class _StrictModel(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)


class SearchSpace(_StrictModel):
    learning_rate_min: StrictFloat = Field(gt=0.0)
    learning_rate_max: StrictFloat = Field(gt=0.0)
    epochs: list[StrictInt] = Field(min_length=1)
    batch_sizes: list[StrictInt] = Field(min_length=1)
    minibatch_sizes: list[StrictInt] = Field(min_length=1)
    target_kl_min: StrictFloat = Field(gt=0.0)
    target_kl_max: StrictFloat = Field(gt=0.0)


class PruningConfig(_StrictModel):
    report_interval_updates: StrictInt = Field(gt=0)
    consecutive_update_count: StrictInt = Field(gt=0)
    minimum_episode_length_retention: StrictFloat = Field(gt=0.0, le=1.0)
    out_of_road_fraction: StrictFloat = Field(gt=0.0, le=1.0)
    clip_fraction: StrictFloat | None = Field(default=None, gt=0.0, le=1.0)
    median_startup_trials: StrictInt = Field(ge=0)
    median_warmup_updates: StrictInt = Field(ge=0)


class StageConfig(_StrictModel):
    update_count: StrictInt = Field(gt=0)
    training_seeds: list[StrictInt] = Field(min_length=1)
    top_config_count: StrictInt = Field(gt=0)


class EvaluationConfig(_StrictModel):
    seed: StrictInt = Field(ge=0)
    maps: list[str] = Field(min_length=1)
    map_seeds: list[StrictInt] = Field(min_length=1)
    transitions_per_scenario: StrictInt = Field(gt=0)
    minimum_retention: StrictFloat = Field(gt=0.0, le=1.0)


class DiagnosticConfig(_StrictModel):
    update_count: StrictInt = Field(gt=0)
    training_seed: StrictInt = Field(ge=0)
    lateral_max_offset_m: StrictFloat = Field(gt=0.0)
    longitudinal_max_speed_fraction: StrictFloat = Field(gt=0.0, lt=1.0)


class PPOStabilityStudyConfig(_StrictModel):
    version: Literal[1]
    study_name: str = Field(min_length=1)
    base_training_config: str = Field(min_length=1)
    sampler_seed: StrictInt = Field(ge=0)
    trial_count: StrictInt = Field(gt=0)
    stage_a_update_count: StrictInt = Field(gt=0)
    stage_a_training_seed: StrictInt = Field(ge=0)
    transitions_per_scenario: StrictInt = Field(gt=0)
    training_maps: list[str] = Field(min_length=1)
    training_map_seeds: list[StrictInt] = Field(min_length=1)
    search: SearchSpace
    pruning: PruningConfig
    stage_b: StageConfig
    stage_c: StageConfig
    evaluation: EvaluationConfig
    diagnostics: DiagnosticConfig

    @model_validator(mode="after")
    def validate_study(self) -> PPOStabilityStudyConfig:
        if self.search.learning_rate_min >= self.search.learning_rate_max:
            raise ValueError("learning-rate search bounds must be increasing")
        if self.search.target_kl_min >= self.search.target_kl_max:
            raise ValueError("target-KL search bounds must be increasing")
        if len(set(self.stage_b.training_seeds)) != len(self.stage_b.training_seeds):
            raise ValueError("Stage B training seeds must be unique")
        if len(set(self.stage_c.training_seeds)) != len(self.stage_c.training_seeds):
            raise ValueError("Stage C training seeds must be unique")
        return self


class TrialParameters(_StrictModel):
    learning_rate: StrictFloat = Field(gt=0.0)
    epochs: StrictInt = Field(gt=0)
    batch_size: StrictInt = Field(gt=0)
    minibatch_size: StrictInt = Field(gt=0)
    target_kl: StrictFloat = Field(gt=0.0)

    @property
    def valid_minibatch(self) -> bool:
        return self.minibatch_size <= self.batch_size and not (
            self.batch_size % self.minibatch_size
        )


def load_stability_config(path: Path) -> PPOStabilityStudyConfig:
    """Load one explicit staged PPO stability manifest."""

    return PPOStabilityStudyConfig.model_validate(load_resolved_yaml_mapping(path))


def scenarios(
    maps: list[str], seeds: list[int], *, limit: int | None
) -> tuple[ScenarioConfig, ...]:
    """Build a deterministic map-major scenario matrix."""

    values = tuple(
        ScenarioConfig(name=f"{map_name.lower()}_s{seed}", map=map_name, seed=seed)
        for seed in seeds
        for map_name in maps
    )
    return values if limit is None else values[:limit]
