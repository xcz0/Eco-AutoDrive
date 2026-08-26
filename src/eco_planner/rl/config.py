"""Strict, typed Hydra boundaries for one PPO-guided RL job."""

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

from eco_planner.evaluation.config import ModelPathsConfig, ScenarioConfig
from eco_planner.models import (
    OrthogonalPolicyGuidanceConfig,
    SamplerConfig,
    parse_guidance_config,
    parse_sampler_config,
)
from eco_planner.runtime.config import RuntimeConfig
from eco_planner.runtime_resources import ResourceProfileConfig


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        allow_inf_nan=False,
        arbitrary_types_allowed=True,
    )


class ExplorationPolicyConfig(_StrictModel):
    """Architecture and affine-Beta initialization parameters."""

    hidden_dim: StrictInt = Field(gt=0)
    reference_mixer_depth: StrictInt = Field(gt=0)
    reference_token_mlp_hidden_dim: StrictInt = Field(gt=0)
    reference_channel_mlp_hidden_dim: StrictInt = Field(gt=0)
    cross_attention_heads: StrictInt = Field(gt=0)
    cross_attention_dropout: StrictFloat = Field(ge=0.0, lt=1.0)
    fusion_mlp_depth: StrictInt = Field(gt=0)
    fusion_hidden_dim: StrictInt = Field(gt=0)
    initial_concentration: StrictFloat = Field(gt=0.0)
    minimum_concentration: StrictFloat = Field(gt=0.0)

    @model_validator(mode="after")
    def validate_policy_contract(self) -> ExplorationPolicyConfig:
        if self.hidden_dim % self.cross_attention_heads:
            raise ValueError("policy.hidden_dim must be divisible by cross_attention_heads")
        if self.initial_concentration <= self.minimum_concentration:
            raise ValueError("policy.initial_concentration must exceed minimum_concentration")
        return self


class PPOConfig(_StrictModel):
    """GAE, PPO, optimizer, and scheduler parameters for one update profile."""

    name: str = Field(min_length=1)
    gamma: StrictFloat = Field(gt=0.0, lt=1.0)
    gae_lambda: StrictFloat = Field(gt=0.0, lt=1.0)
    clip_epsilon: StrictFloat = Field(gt=0.0, lt=1.0)
    value_coefficient: StrictFloat = Field(ge=0.0)
    entropy_coefficient: StrictFloat = Field(ge=0.0)
    learning_rate: StrictFloat = Field(gt=0.0)
    adam_epsilon: StrictFloat = Field(gt=0.0)
    weight_decay: StrictFloat = Field(ge=0.0)
    max_gradient_norm: StrictFloat = Field(gt=0.0)
    epochs: StrictInt = Field(gt=0)
    batch_size: StrictInt = Field(gt=0)
    minibatch_size: StrictInt = Field(gt=0)
    minibatch_seed: StrictInt = Field(ge=0)
    scheduler_total_optimizer_steps: StrictInt = Field(gt=0)
    scheduler_minimum_learning_rate: StrictFloat = Field(ge=0.0)

    @property
    def optimizer_steps_per_update(self) -> int:
        return self.epochs * (self.batch_size // self.minibatch_size)

    @model_validator(mode="after")
    def validate_optimization_contract(self) -> PPOConfig:
        if self.batch_size % self.minibatch_size:
            raise ValueError("rl.batch_size must be divisible by rl.minibatch_size")
        if self.scheduler_minimum_learning_rate >= self.learning_rate:
            raise ValueError("scheduler minimum learning rate must be below the initial rate")
        if self.scheduler_total_optimizer_steps < self.optimizer_steps_per_update:
            raise ValueError("scheduler horizon must cover at least one complete PPO update")
        return self


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


class MetaDriveBuiltinRewardConfig(_StrictModel):
    driving_reward: StrictFloat = Field(ge=0.0)
    speed_reward: StrictFloat = Field(ge=0.0)
    success_reward: StrictFloat = Field(ge=0.0)
    out_of_road_penalty: StrictFloat = Field(ge=0.0)
    crash_vehicle_penalty: StrictFloat = Field(ge=0.0)
    crash_object_penalty: StrictFloat = Field(ge=0.0)
    crash_sidewalk_penalty: StrictFloat = Field(ge=0.0)
    use_lateral_reward: StrictBool


class TrainingConfig(_StrictModel):
    """General closed-loop PPO run controls; smoke values live in YAML."""

    update_count: StrictInt = Field(gt=0)
    transitions_per_environment: StrictInt = Field(gt=0)
    replay_id: StrictInt = Field(ge=0)
    deterministic: StrictBool
    stopped_speed_threshold_mps: StrictFloat = Field(gt=0.0)
    boundary_distance: StrictFloat = Field(gt=0.0, lt=0.5)
    boundary_sample_count: StrictInt = Field(gt=0)
    diagnostic_seed: StrictInt = Field(ge=0)
    resume_checkpoint_path: str | None = None


class RLTrainingJobConfig(_StrictModel):
    name: str = Field(min_length=1)
    map_query_radius_m: StrictFloat = Field(gt=0.0)
    training: TrainingConfig
    env: dict[str, Any]
    model: ModelPathsConfig
    runtime: RuntimeConfig
    sampler: SamplerConfig
    guidance: OrthogonalPolicyGuidanceConfig
    policy: ExplorationPolicyConfig
    reward: MetaDriveBuiltinRewardConfig
    rl: PPOConfig
    resources: ResourceProfileConfig
    scenarios: tuple[ScenarioConfig, ...]

    @model_validator(mode="after")
    def validate_training_job(self) -> RLTrainingJobConfig:
        if not self.scenarios:
            raise ValueError("training requires at least one scenario")
        _validate_rollout_environment(self.env, 0, self.training.transitions_per_environment)
        for name in (
            "driving_reward",
            "speed_reward",
            "success_reward",
            "out_of_road_penalty",
            "crash_vehicle_penalty",
            "crash_object_penalty",
            "crash_sidewalk_penalty",
            "use_lateral_reward",
        ):
            if self.env.get(name) != getattr(self.reward, name):
                raise ValueError(f"env.{name} must equal the selected reward profile")
        sample_count = len(self.scenarios) * self.training.transitions_per_environment
        if self.rl.batch_size != sample_count:
            raise ValueError("rl.batch_size must equal all closed-loop transitions per update")
        required_steps = self.training.update_count * self.rl.optimizer_steps_per_update
        if self.rl.scheduler_total_optimizer_steps < required_steps:
            raise ValueError("scheduler horizon must cover every configured PPO update")
        return self


def parse_exploration_policy_config(config: DictConfig) -> ExplorationPolicyConfig:
    return ExplorationPolicyConfig.model_validate(
        dict(OmegaConf.to_container(config, resolve=True, throw_on_missing=True))
    )


def parse_ppo_config(config: DictConfig) -> PPOConfig:
    return PPOConfig.model_validate(
        dict(OmegaConf.to_container(config, resolve=True, throw_on_missing=True))
    )


def parse_rollout_config(config: DictConfig) -> RolloutJobConfig:
    raw = dict(OmegaConf.to_container(config, resolve=True, throw_on_missing=True))
    guidance = parse_guidance_config(config["guidance"])
    if not isinstance(guidance, OrthogonalPolicyGuidanceConfig):
        raise ValueError("rollout requires guidance=orthogonal_policy")
    raw["sampler"] = parse_sampler_config(config["sampler"])
    raw["guidance"] = guidance
    raw["policy"] = parse_exploration_policy_config(config["policy"])
    return RolloutJobConfig.model_validate(raw)


def parse_training_config(config: DictConfig) -> RLTrainingJobConfig:
    raw = dict(OmegaConf.to_container(config, resolve=True, throw_on_missing=True))
    guidance = parse_guidance_config(config["guidance"])
    if not isinstance(guidance, OrthogonalPolicyGuidanceConfig):
        raise ValueError("training requires guidance=orthogonal_policy")
    raw["sampler"] = parse_sampler_config(config["sampler"])
    raw["guidance"] = guidance
    raw["policy"] = parse_exploration_policy_config(config["policy"])
    raw["rl"] = parse_ppo_config(config["rl"])
    if isinstance(raw.get("scenarios"), list):
        raw["scenarios"] = tuple(raw["scenarios"])
    return RLTrainingJobConfig.model_validate(raw)


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
