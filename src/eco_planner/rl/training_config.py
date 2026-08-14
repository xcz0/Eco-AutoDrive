"""Strict Hydra boundary for Stage-6 closed-loop smoke training."""

from __future__ import annotations

from typing import Any, Literal

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, model_validator

from eco_planner.evaluation.config import ModelPathsConfig, RuntimeConfig, ScenarioConfig
from eco_planner.models.guidance import (
    OrthogonalPolicyGuidanceConfig,
    parse_guidance_config,
    validate_guidance_sampler,
)
from eco_planner.models.sampling import SamplerConfig, parse_sampler_config
from eco_planner.rl.config import ExplorationPolicyConfig, parse_exploration_policy_config
from eco_planner.rl.ppo_config import PPOOptimizationConfig, parse_ppo_optimization_config


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        allow_inf_nan=False,
        arbitrary_types_allowed=True,
    )


class MetaDriveBuiltinRewardConfig(_StrictModel):
    name: Literal["metadrive_builtin_v1"]
    unit: Literal["dimensionless_score"]
    driving_reward: Literal[1.0]
    speed_reward: Literal[0.1]
    success_reward: Literal[10.0]
    out_of_road_penalty: Literal[5.0]
    crash_vehicle_penalty: Literal[5.0]
    crash_object_penalty: Literal[5.0]
    crash_sidewalk_penalty: Literal[0.0]
    use_lateral_reward: Literal[False]


class Stage6TrainingConfig(_StrictModel):
    profile: Literal["smoke"]
    update_count: Literal[4]
    transitions_per_environment: Literal[16]
    replay_id: StrictInt = Field(ge=0, le=1)
    deterministic: Literal[True]
    stopped_speed_threshold_mps: StrictFloat = Field(gt=0.0)
    boundary_distance: StrictFloat = Field(gt=0.0, lt=0.5)
    boundary_sample_count: Literal[4096]
    diagnostic_seed: StrictInt = Field(ge=0)


class Stage6TrainingJobConfig(_StrictModel):
    name: Literal["plannerrft_stage6_closed_loop_smoke"]
    map_query_radius_m: StrictFloat = Field(gt=0.0)
    training: Stage6TrainingConfig
    env: dict[str, Any]
    model: ModelPathsConfig
    runtime: RuntimeConfig
    sampler: SamplerConfig
    guidance: OrthogonalPolicyGuidanceConfig
    policy: ExplorationPolicyConfig
    reward: MetaDriveBuiltinRewardConfig
    train: PPOOptimizationConfig
    scenarios: tuple[ScenarioConfig, ...]

    @model_validator(mode="after")
    def validate_stage6_contract(self) -> Stage6TrainingJobConfig:
        if self.runtime.accelerator != "cuda" or self.runtime.precision != "bf16-mixed":
            raise ValueError("Stage-6 smoke requires explicit CUDA BF16 runtime")
        expected_scenarios = (("straight", "S", 0), ("gentle_curve", "SC", 0))
        actual_scenarios = tuple((item.name, item.map, item.seed) for item in self.scenarios)
        if actual_scenarios != expected_scenarios:
            raise ValueError("Stage-6 smoke scenarios must be fixed S/SC with map seed zero")
        required_env = {
            "trajectory_execution_steps": 1,
            "trajectory_horizon": 80,
            "decision_repeat": 5,
            "horizon": 16,
            "traffic_density": 0.0,
            "random_traffic": False,
            "accident_prob": 0.0,
        }
        for name, expected in required_env.items():
            if self.env.get(name) != expected:
                raise ValueError(f"Stage-6 smoke requires env.{name}={expected!r}")
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
        if self.train.name != "ppo_stage6_smoke" or self.train.batch_size != sample_count:
            raise ValueError("Stage-6 PPO batch must contain exactly 32 closed-loop transitions")
        expected_steps = self.training.update_count * self.train.optimizer_steps_per_update
        if self.train.scheduler_total_optimizer_steps != expected_steps:
            raise ValueError("Stage-6 scheduler horizon must cover all four PPO updates")
        return self


def parse_stage6_training_config(config: DictConfig) -> Stage6TrainingJobConfig:
    """Resolve the complete Stage-6 smoke profile at the CLI boundary."""

    if not isinstance(config, DictConfig):
        raise TypeError("Stage-6 training configuration must be a DictConfig")
    raw = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    if not isinstance(raw, dict):
        raise TypeError("Stage-6 training configuration must resolve to a dictionary")
    required_nodes = ("sampler", "guidance", "policy", "train")
    nodes = {name: config.get(name) for name in required_nodes}
    if not all(isinstance(node, DictConfig) for node in nodes.values()):
        raise ValueError("Stage-6 training must select sampler, guidance, policy, and train")
    sampler = parse_sampler_config(nodes["sampler"])
    guidance = parse_guidance_config(nodes["guidance"])
    if not isinstance(guidance, OrthogonalPolicyGuidanceConfig):
        raise ValueError("Stage-6 training requires orthogonal policy guidance")
    validate_guidance_sampler(guidance, sampler)
    payload = dict(raw)
    payload["sampler"] = sampler
    payload["guidance"] = guidance
    payload["policy"] = parse_exploration_policy_config(nodes["policy"])
    payload["train"] = parse_ppo_optimization_config(nodes["train"])
    if isinstance(payload.get("scenarios"), list):
        payload["scenarios"] = tuple(payload["scenarios"])
    return Stage6TrainingJobConfig.model_validate(payload)
