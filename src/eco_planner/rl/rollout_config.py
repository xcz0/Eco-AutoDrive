"""Strict Hydra boundary for the serial Stage-4 rollout collector."""

from __future__ import annotations

from typing import Any, Literal

from omegaconf import DictConfig, OmegaConf
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    model_validator,
)

from eco_planner.evaluation.config import ModelPathsConfig, RuntimeConfig, ScenarioConfig
from eco_planner.models.guidance import (
    OrthogonalPolicyGuidanceConfig,
    parse_guidance_config,
    validate_guidance_sampler,
)
from eco_planner.models.sampling import SamplerConfig, parse_sampler_config
from eco_planner.rl.config import ExplorationPolicyConfig, parse_exploration_policy_config


class _StrictModel(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)


class RolloutCollectionConfig(_StrictModel):
    mode: Literal["no_traffic", "traffic"]
    max_transitions: StrictInt = Field(gt=0)
    history_warmup_steps: StrictInt = Field(ge=0)
    transition_dt_s: StrictFloat
    policy_action_seed: StrictInt = Field(ge=0)
    reward_source: Literal["metadrive_builtin_v1"]
    bootstrap_time_limit: StrictBool
    candidate_count: StrictInt
    stopped_speed_threshold_mps: StrictFloat = Field(gt=0.0)

    @model_validator(mode="after")
    def validate_contract(self) -> RolloutCollectionConfig:
        if self.transition_dt_s != 0.1:
            raise ValueError("Stage-4 rollout transition_dt_s must be exactly 0.1")
        if not self.bootstrap_time_limit:
            raise ValueError("Stage-4 rollout requires bootstrap_time_limit=true")
        if self.candidate_count != 1:
            raise ValueError("Stage-4 rollout requires candidate_count=1")
        if self.mode == "no_traffic" and self.history_warmup_steps != 0:
            raise ValueError("no-traffic rollout requires zero history warmup steps")
        if self.mode == "traffic" and self.history_warmup_steps != 20:
            raise ValueError("traffic rollout requires exactly 20 history warmup steps")
        return self


class RolloutJobConfig(_StrictModel):
    name: str = Field(min_length=1)
    map_query_radius_m: StrictFloat = Field(gt=0.0)
    rollout: RolloutCollectionConfig
    env: dict[str, Any]
    model: ModelPathsConfig
    runtime: RuntimeConfig
    sampler: SamplerConfig
    guidance: OrthogonalPolicyGuidanceConfig
    policy: ExplorationPolicyConfig
    scenario: ScenarioConfig

    @model_validator(mode="after")
    def validate_job(self) -> RolloutJobConfig:
        if self.env.get("trajectory_execution_steps") != 1:
            raise ValueError("rollout env.trajectory_execution_steps must equal 1")
        if self.env.get("trajectory_horizon") != 80:
            raise ValueError("rollout env.trajectory_horizon must equal 80")
        if self.env.get("decision_repeat") != 5:
            raise ValueError("rollout env.decision_repeat must equal 5")
        horizon = self.env.get("horizon")
        required_horizon = self.rollout.history_warmup_steps + self.rollout.max_transitions
        if type(horizon) is not int or horizon < required_horizon:
            raise ValueError("rollout env.horizon must cover warmup plus max_transitions")
        return self


def parse_rollout_config(config: DictConfig) -> RolloutJobConfig:
    """Resolve one rollout profile without passing DictConfig into runtime code."""

    if not isinstance(config, DictConfig):
        raise TypeError("rollout configuration must be a DictConfig")
    raw = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    if not isinstance(raw, dict):
        raise TypeError("rollout configuration must resolve to a dictionary")
    sampler_node = config.get("sampler")
    guidance_node = config.get("guidance")
    policy_node = config.get("policy")
    if not all(isinstance(node, DictConfig) for node in (sampler_node, guidance_node, policy_node)):
        raise ValueError("rollout must select sampler, guidance, and policy profiles")
    guidance = parse_guidance_config(guidance_node)
    if not isinstance(guidance, OrthogonalPolicyGuidanceConfig):
        raise ValueError("rollout requires guidance=orthogonal_policy")
    payload = dict(raw)
    sampler = parse_sampler_config(sampler_node)
    validate_guidance_sampler(guidance, sampler)
    payload["sampler"] = sampler
    payload["guidance"] = guidance
    payload["policy"] = parse_exploration_policy_config(policy_node)
    return RolloutJobConfig.model_validate(payload)
