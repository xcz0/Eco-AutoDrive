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
    devices: StrictInt
    precision: Literal["auto", "32-true", "16-mixed", "bf16-mixed"]
    seed: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def require_one_device(self) -> RuntimeConfig:
        if self.devices != 1:
            raise ValueError("runtime.devices must be 1")
        return self


class ExecutionConfig(_StrictModel):
    mode: Literal["serial", "parallel"]
    launcher: Literal["basic", "joblib"]
    worker_count: StrictInt = Field(gt=0)
    torch_threads_per_worker: StrictInt | None = Field(default=None, gt=0)
    deterministic: StrictBool

    @model_validator(mode="after")
    def validate_mode(self) -> ExecutionConfig:
        if self.mode == "serial" and (self.launcher != "basic" or self.worker_count != 1):
            raise ValueError("serial execution requires the basic launcher and one worker")
        if self.mode == "parallel" and (self.launcher != "joblib" or self.worker_count != 2):
            raise ValueError("parallel execution requires Joblib with exactly two workers")
        return self


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
        if evaluation.execution.mode == "parallel" and self.video.enabled:
            raise ValueError("parallel execution requires video.enabled=false")
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

    if not isinstance(config, DictConfig):
        raise TypeError("evaluation configuration must be a DictConfig")
    raw = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    if not isinstance(raw, dict):
        raise TypeError("evaluation configuration must resolve to a dictionary")
    sampler_node = config.get("sampler")
    guidance_node = config.get("guidance")
    if not isinstance(sampler_node, DictConfig) or not isinstance(guidance_node, DictConfig):
        raise ValueError("evaluation configuration must select sampler and guidance profiles")
    payload = dict(raw)
    payload["sampler"] = parse_sampler_config(sampler_node)
    payload["guidance"] = parse_guidance_config(guidance_node)
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
