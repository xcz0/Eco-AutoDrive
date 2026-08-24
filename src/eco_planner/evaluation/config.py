"""Typed evaluation configuration parsed once at the Hydra boundary."""

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

from eco_planner.models import (
    GuidanceConfig,
    SamplerConfig,
    parse_guidance_config,
    parse_sampler_config,
)
from eco_planner.runtime_resources import ResourceProfileConfig


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True,
        allow_inf_nan=False,
    )


class ScenarioConfig(_StrictModel):
    name: str = Field(min_length=1)
    map: str = Field(min_length=1)
    seed: StrictInt = Field(ge=0)


class RuntimeConfig(_StrictModel):
    accelerator: Literal["auto", "cpu", "cuda"]
    precision: Literal["auto", "32-true", "16-mixed", "bf16-mixed"]
    seed: StrictInt = Field(ge=0)


class ExecutionConfig(_StrictModel):
    mode: Literal["serial", "parallel"]
    vector_env_slots: StrictInt | None
    torch_threads_per_worker: StrictInt | None = Field(default=None, gt=0)
    deterministic: StrictBool


class MatrixConfig(_StrictModel):
    seeds: tuple[StrictInt, ...]
    traffic_densities: tuple[StrictFloat, ...]

    @model_validator(mode="after")
    def validate_grid(self) -> MatrixConfig:
        if not self.seeds or any(seed < 0 for seed in self.seeds):
            raise ValueError("matrix seeds must be non-empty non-negative integers")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("matrix seeds must be unique")
        if not self.traffic_densities or any(
            density <= 0.0 or density > 1.0 for density in self.traffic_densities
        ):
            raise ValueError("matrix traffic densities must be in (0, 1]")
        if len(set(self.traffic_densities)) != len(self.traffic_densities):
            raise ValueError("matrix traffic densities must be unique")
        return self


class EvaluationConfig(_StrictModel):
    mode: Literal["no_traffic", "traffic"]
    profile: str = Field(min_length=1)
    history_warmup_steps: StrictInt = Field(ge=0)
    evaluated_horizon_steps: StrictInt = Field(gt=0)
    execution: ExecutionConfig
    matrix: MatrixConfig | None = None


class ModelPathsConfig(_StrictModel):
    args_path: str = Field(min_length=1)
    checkpoint_path: str = Field(min_length=1)


class VideoConfig(_StrictModel):
    enabled: StrictBool
    fps: StrictInt = Field(gt=0)
    screen_width: StrictInt = Field(gt=0)
    screen_height: StrictInt = Field(gt=0)
    film_width: StrictInt = Field(gt=0)
    film_height: StrictInt = Field(gt=0)
    scaling: StrictFloat = Field(gt=0.0)


class EvaluationJobConfig(_StrictModel):
    name: str = Field(min_length=1)
    map_query_radius_m: StrictFloat = Field(gt=0.0)
    evaluation: EvaluationConfig
    env: dict[str, Any]
    model: ModelPathsConfig
    runtime: RuntimeConfig
    sampler: SamplerConfig
    guidance: GuidanceConfig
    resources: ResourceProfileConfig | None = None
    scenarios: tuple[ScenarioConfig, ...]
    video: VideoConfig

    @model_validator(mode="after")
    def validate_job(self) -> EvaluationJobConfig:
        evaluation = self.evaluation
        if self.env.get("trajectory_execution_steps") != 5:
            raise ValueError("evaluation env.trajectory_execution_steps must equal 5")
        horizon = self.env.get("horizon")
        if type(horizon) is not int:
            raise TypeError("env.horizon must be an integer")
        if horizon != evaluation.history_warmup_steps + evaluation.evaluated_horizon_steps:
            raise ValueError("env.horizon must equal warmup plus evaluated horizon steps")
        if not self.scenarios:
            raise ValueError("at least one evaluation scenario is required")
        names = [scenario.name for scenario in self.scenarios]
        if len(set(names)) != len(names):
            raise ValueError("evaluation scenario names must be unique")
        if evaluation.mode == "no_traffic":
            if evaluation.history_warmup_steps != 0:
                raise ValueError("no-traffic evaluation requires zero history warmup steps")
        else:
            self._validate_traffic_environment()
        execution = evaluation.execution
        if execution.mode == "parallel" and self.resources is None:
            raise ValueError("parallel evaluation requires a resource profile")
        if execution.mode == "parallel" and self.video.enabled:
            raise ValueError("parallel execution requires video.enabled=false")
        if execution.vector_env_slots is not None:
            if execution.vector_env_slots <= 0:
                raise ValueError("vector_env_slots must be positive when configured")
            if execution.mode != "serial":
                raise ValueError("vector evaluation requires execution.mode=serial")
            if self.video.enabled:
                raise ValueError("vector evaluation requires video.enabled=false")
        return self

    def _validate_traffic_environment(self) -> None:
        if self.evaluation.history_warmup_steps != 20:
            raise ValueError("traffic evaluation requires exactly 20 history warmup steps")
        required = {
            "traffic_mode": "trigger",
            "random_traffic": False,
            "accident_prob": 0.0,
        }
        for name, expected in required.items():
            if self.env.get(name) != expected:
                raise ValueError(f"traffic evaluation requires env.{name}={expected!r}")
        density = self.env.get("traffic_density")
        if type(density) not in {int, float} or not 0.0 < float(density) <= 1.0:
            raise ValueError("traffic evaluation requires traffic_density in (0, 1]")


def parse_evaluation_config(config: DictConfig) -> EvaluationJobConfig:
    """Resolve Hydra values and return the sole typed configuration used by evaluation."""

    raw = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    payload = dict(raw)
    payload["sampler"] = parse_sampler_config(config["sampler"])
    payload["guidance"] = parse_guidance_config(config["guidance"])
    scenarios = payload.get("scenarios")
    if isinstance(scenarios, list):
        payload["scenarios"] = tuple(scenarios)
    evaluation = payload.get("evaluation")
    if isinstance(evaluation, dict) and isinstance(evaluation.get("matrix"), dict):
        matrix = dict(evaluation["matrix"])
        for name in ("seeds", "traffic_densities"):
            if isinstance(matrix.get(name), list):
                matrix[name] = tuple(matrix[name])
        evaluation = dict(evaluation)
        evaluation["matrix"] = matrix
        payload["evaluation"] = evaluation
    result = EvaluationJobConfig.model_validate(payload)
    return result
