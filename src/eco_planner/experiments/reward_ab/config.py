"""Validated manifest for the matched PPO reward A/B experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, model_validator

from eco_planner._repository import CONFIG_ROOT
from eco_planner.configuration import load_resolved_yaml_mapping

DEFAULT_STUDY = CONFIG_ROOT / "experiments" / "reward" / "ppo_ab.yaml"


class _StrictModel(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)


class RewardProfileSpec(_StrictModel):
    id: Literal["builtin", "energy"]
    reward_config: Literal["metadrive_builtin_v1", "plannerrft_energy_v1"]


class MatchedTrainingSpec(_StrictModel):
    update_count: StrictInt = Field(ge=2)
    transitions_per_environment: StrictInt = Field(gt=0)
    scheduler_total_optimizer_steps: StrictInt = Field(gt=0)
    training_seeds: list[StrictInt] = Field(min_length=1)
    replay_ids: list[StrictInt] = Field(min_length=1)


class ReviewThresholds(_StrictModel):
    longitudinal_action_mean_deadband: StrictFloat = Field(ge=0.0)
    energy_intensity_deadband_fraction: StrictFloat = Field(ge=0.0)
    maximum_progress_drop_fraction: StrictFloat = Field(ge=0.0, le=1.0)
    maximum_mean_speed_drop_fraction: StrictFloat = Field(ge=0.0, le=1.0)
    maximum_collision_count_increase: StrictInt = Field(ge=0)
    maximum_out_of_road_count_increase: StrictInt = Field(ge=0)


class PPORewardABConfig(_StrictModel):
    version: Literal[2]
    base_training_config: str
    profiles: list[RewardProfileSpec]
    matched_training: MatchedTrainingSpec
    review_thresholds: ReviewThresholds

    @model_validator(mode="after")
    def validate_pair(self) -> PPORewardABConfig:
        if [(item.id, item.reward_config) for item in self.profiles] != [
            ("builtin", "metadrive_builtin_v1"),
            ("energy", "plannerrft_energy_v1"),
        ]:
            raise ValueError("PPO reward A/B profiles must be builtin then energy")
        if len(set(self.matched_training.training_seeds)) != len(
            self.matched_training.training_seeds
        ):
            raise ValueError("PPO reward A/B training seeds must be unique")
        if len(set(self.matched_training.replay_ids)) != len(self.matched_training.replay_ids):
            raise ValueError("PPO reward A/B replay ids must be unique")
        return self


def load_ab_config(path: Path) -> PPORewardABConfig:
    """Load one explicit reward A/B experiment manifest."""

    return PPORewardABConfig.model_validate(load_resolved_yaml_mapping(path))
