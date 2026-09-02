"""Strict, typed Hydra boundaries for one PPO-guided RL job."""

from __future__ import annotations

from typing import Any, Literal

from omegaconf import DictConfig
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    model_validator,
)

from eco_planner.configuration import resolve_config_mapping
from eco_planner.envs.metadrive.reward import (
    MetaDriveBuiltinRewardConfig,
    RewardProfileConfig,
)
from eco_planner.evaluation import ModelPathsConfig, ScenarioConfig
from eco_planner.models import (
    OrthogonalPolicyGuidanceConfig,
    SamplerConfig,
    parse_guidance_config,
    parse_sampler_config,
)
from eco_planner.rl.optimization.config import PPOConfig, parse_ppo_config
from eco_planner.rl.policy.config import (
    ExplorationPolicyConfig,
    parse_exploration_policy_config,
)
from eco_planner.runtime.config import RuntimeConfig
from eco_planner.runtime.resources import ResourceProfileConfig


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        allow_inf_nan=False,
        arbitrary_types_allowed=True,
    )


class RolloutConfig(_StrictModel):
    """One serial policy-guided MetaDrive collection profile."""

    mode: Literal["no_traffic", "traffic"]
    max_transitions: StrictInt = Field(gt=0)
    history_warmup_steps: StrictInt = Field(ge=0)
    policy_action_seed: StrictInt = Field(ge=0)
    stopped_speed_threshold_mps: StrictFloat = Field(gt=0.0)

    @model_validator(mode="after")
    def validate_rollout_contract(self) -> RolloutConfig:
        if self.mode == "no_traffic" and self.history_warmup_steps != 0:
            raise ValueError("no-traffic rollout requires zero history warmup steps")
        if self.mode == "traffic" and self.history_warmup_steps != 20:
            raise ValueError("traffic rollout requires exactly 20 history warmup steps")
        return self


class RolloutJobConfig(_StrictModel):
    name: str = Field(min_length=1)
    map_query_radius_m: StrictFloat = Field(gt=0.0)
    rollout: RolloutConfig
    env: dict[str, Any]
    model: ModelPathsConfig
    runtime: RuntimeConfig
    sampler: SamplerConfig
    guidance: OrthogonalPolicyGuidanceConfig
    policy: ExplorationPolicyConfig
    scenario: ScenarioConfig

    @model_validator(mode="after")
    def validate_job(self) -> RolloutJobConfig:
        _validate_rollout_environment(
            self.env, self.rollout.history_warmup_steps, self.rollout.max_transitions
        )
        return self


class TrainingLoopConfig(_StrictModel):
    """General closed-loop PPO run controls; smoke values live in YAML."""

    update_count: StrictInt = Field(gt=0)
    transitions_per_environment: StrictInt = Field(gt=0)
    mode: Literal["no_traffic", "traffic"]
    history_warmup_steps: StrictInt = Field(ge=0)
    replay_id: StrictInt = Field(ge=0)
    deterministic: StrictBool
    stopped_speed_threshold_mps: StrictFloat = Field(gt=0.0)
    boundary_distance: StrictFloat = Field(gt=0.0, lt=0.5)
    boundary_sample_count: StrictInt = Field(gt=0)
    diagnostic_seed: StrictInt = Field(ge=0)
    planner_compile_mode: Literal["eager", "dit_reduce_overhead"]
    resume_checkpoint_path: str | None = None

    @model_validator(mode="after")
    def validate_training_mode(self) -> TrainingLoopConfig:
        if self.mode == "no_traffic" and self.history_warmup_steps != 0:
            raise ValueError("no-traffic training requires zero history warmup steps")
        if self.mode == "traffic" and self.history_warmup_steps != 20:
            raise ValueError("traffic training requires exactly 20 history warmup steps")
        return self


class TrainingJobConfig(_StrictModel):
    name: str = Field(min_length=1)
    map_query_radius_m: StrictFloat = Field(gt=0.0)
    training: TrainingLoopConfig
    env: dict[str, Any]
    model: ModelPathsConfig
    runtime: RuntimeConfig
    sampler: SamplerConfig
    guidance: OrthogonalPolicyGuidanceConfig
    policy: ExplorationPolicyConfig
    reward: RewardProfileConfig
    ppo: PPOConfig
    resources: ResourceProfileConfig | None = None
    scenarios: tuple[ScenarioConfig, ...]

    @model_validator(mode="after")
    def validate_training_job(self) -> TrainingJobConfig:
        if not self.scenarios:
            raise ValueError("training requires at least one scenario")
        _validate_rollout_environment(self.env, 0, self.training.transitions_per_environment)
        if isinstance(self.reward, MetaDriveBuiltinRewardConfig):
            conflicting = {
                name
                for name in (
                    "driving_reward",
                    "speed_reward",
                    "success_reward",
                    "out_of_road_penalty",
                    "crash_vehicle_penalty",
                    "crash_object_penalty",
                    "crash_sidewalk_penalty",
                    "use_lateral_reward",
                )
                if name in self.env and self.env[name] != getattr(self.reward, name)
            }
            if conflicting:
                raise ValueError(
                    "env reward fields conflict with the selected reward profile: "
                    f"{sorted(conflicting)}"
                )
        sample_count = len(self.scenarios) * self.training.transitions_per_environment
        if self.ppo.batch_size != sample_count:
            raise ValueError("ppo.batch_size must equal all closed-loop transitions per update")
        required_steps = self.training.update_count * self.ppo.optimizer_steps_per_update
        if self.ppo.scheduler_total_optimizer_steps < required_steps:
            raise ValueError("scheduler horizon must cover every configured PPO update")
        return self


def parse_rollout_config(config: DictConfig) -> RolloutJobConfig:
    raw = resolve_config_mapping(config)
    guidance = parse_guidance_config(config["guidance"])
    if not isinstance(guidance, OrthogonalPolicyGuidanceConfig):
        raise ValueError("rollout requires guidance=orthogonal_policy")
    raw["sampler"] = parse_sampler_config(config["sampler"])
    raw["guidance"] = guidance
    raw["policy"] = parse_exploration_policy_config(config["policy"])
    return RolloutJobConfig.model_validate(raw)


def parse_training_config(config: DictConfig) -> TrainingJobConfig:
    raw = resolve_config_mapping(config)
    guidance = parse_guidance_config(config["guidance"])
    if not isinstance(guidance, OrthogonalPolicyGuidanceConfig):
        raise ValueError("training requires guidance=orthogonal_policy")
    raw["sampler"] = parse_sampler_config(config["sampler"])
    raw["guidance"] = guidance
    raw["policy"] = parse_exploration_policy_config(config["policy"])
    raw["ppo"] = parse_ppo_config(config["ppo"])
    if isinstance(raw.get("scenarios"), list):
        raw["scenarios"] = tuple(raw["scenarios"])
    return TrainingJobConfig.model_validate(raw)


def _validate_rollout_environment(
    env: dict[str, Any], history_warmup_steps: int, transition_count: int
) -> None:
    mode = env.get("execution_mode")
    if mode is not None and mode != "rollout":
        raise ValueError("rollout requires env.execution_mode=rollout")
    if mode is None and env.get("trajectory_execution_steps") != 1:
        raise ValueError("rollout requires env.execution_mode=rollout")
    horizon = env.get("horizon")
    required_horizon = history_warmup_steps + transition_count
    if type(horizon) is not int or horizon < required_horizon:
        raise ValueError("rollout env.horizon must cover warmup plus requested transitions")
