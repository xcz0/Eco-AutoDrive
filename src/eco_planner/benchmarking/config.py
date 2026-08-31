"""Shared benchmark configuration, measurements, and artifact persistence."""

from __future__ import annotations

import os
import platform
import statistics
from collections.abc import Sequence
from math import isfinite
from pathlib import Path
from typing import Any, Literal, TypedDict, TypeVar

import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    field_validator,
    model_validator,
)

from eco_planner.artifacts import (
    collect_repository_metadata,
    write_json,
    write_tracked_diff,
)
from eco_planner.evaluation.config import ModelPathsConfig


class StrictBenchmarkModel(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)


class ScalingBenchmarkConfig(StrictBenchmarkModel):
    kind: Literal["throughput"]
    batch_sizes: tuple[StrictInt, ...]
    worker_counts: tuple[StrictInt, ...]
    warmup_cycles: StrictInt = Field(gt=0)
    measured_cycles: StrictInt = Field(gt=0)
    repeats: StrictInt = Field(gt=0)

    @field_validator("batch_sizes", "worker_counts", mode="before")
    @classmethod
    def tuple_scales(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_scales(self) -> ScalingBenchmarkConfig:
        _validate_scales(self.batch_sizes, "batch_sizes")
        _validate_scales(self.worker_counts, "worker_counts")
        return self


class RolloutBenchmarkConfig(StrictBenchmarkModel):
    kind: Literal["rollout"]
    batch_sizes: tuple[StrictInt, ...]
    collector_modes: tuple[Literal["serial", "vector"], ...]
    mode: Literal["no_traffic", "traffic"]
    history_warmup_steps: StrictInt = Field(ge=0)
    ppo_epochs: StrictInt = Field(gt=0)
    ppo_minibatch_size: StrictInt = Field(gt=0)
    scenario_seed_base: StrictInt = Field(ge=0)
    noise_seed_base: StrictInt = Field(ge=0)
    policy_action_seed_base: StrictInt = Field(ge=0)
    warmup_updates: StrictInt = Field(gt=0)
    measured_updates: StrictInt = Field(gt=0)
    transitions_per_slot: StrictInt = Field(gt=0)
    repeats: StrictInt = Field(gt=0)

    @field_validator("batch_sizes", "collector_modes", mode="before")
    @classmethod
    def tuple_scales(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_scales(self) -> RolloutBenchmarkConfig:
        _validate_scales(self.batch_sizes, "batch_sizes")
        if not self.collector_modes or len(set(self.collector_modes)) != len(self.collector_modes):
            raise ValueError("collector_modes must be non-empty and unique")
        if self.mode == "no_traffic" and self.history_warmup_steps != 0:
            raise ValueError("no-traffic rollout benchmark requires zero history warmup steps")
        if any(batch_size * self.transitions_per_slot < 2 for batch_size in self.batch_sizes):
            raise ValueError("each rollout benchmark batch must contain at least two transitions")
        if any(
            (batch_size * self.transitions_per_slot) % self.ppo_minibatch_size
            for batch_size in self.batch_sizes
        ):
            raise ValueError(
                "each rollout benchmark sample count must be divisible by ppo_minibatch_size"
            )
        return self


class EnvironmentBenchmarkConfig(StrictBenchmarkModel):
    kind: Literal["environment"]
    map: str = Field(min_length=1)
    seed: StrictInt = Field(ge=0)
    map_query_radius_m: StrictFloat = Field(gt=0.0)
    traffic_density: StrictFloat = Field(gt=0.0, le=1.0)
    history_warmup_steps: StrictInt = Field(gt=0)
    timing_warmup_cycles: StrictInt = Field(gt=0)
    measured_cycles: StrictInt = Field(gt=0)
    repeats: StrictInt = Field(gt=0)
    traffic_baseline_ms: StrictFloat | None = Field(default=None, gt=0.0)
    no_traffic_baseline_ms: StrictFloat | None = Field(default=None, gt=0.0)
    traffic_required_improvement_fraction: StrictFloat = Field(ge=0.0, lt=1.0)
    no_traffic_allowed_regression_fraction: StrictFloat = Field(ge=0.0)


class EnvironmentBenchmarkJobConfig(StrictBenchmarkModel):
    name: str = Field(min_length=1)
    env: dict[str, Any]
    model: ModelPathsConfig
    benchmark: EnvironmentBenchmarkConfig


class Measurement(TypedDict):
    samples: list[float]
    median: float
    minimum: float
    maximum: float


BenchmarkConfigT = TypeVar("BenchmarkConfigT", bound=StrictBenchmarkModel)


def measurement(samples: Sequence[float]) -> Measurement:
    values = [float(value) for value in samples]
    if not values:
        raise ValueError("benchmark measurement requires at least one sample")
    if not all(isfinite(value) for value in values):
        raise ValueError("benchmark measurement samples must be finite")
    if any(value < 0.0 for value in values):
        raise ValueError("benchmark measurement samples must be non-negative")
    return {
        "samples": values,
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def split_benchmark_config(
    config: DictConfig, model: type[BenchmarkConfigT]
) -> tuple[DictConfig, BenchmarkConfigT]:
    payload = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    if not isinstance(payload, dict):
        raise TypeError("benchmark configuration must resolve to a mapping")
    benchmark_payload = payload.pop("benchmark", None)
    if not isinstance(benchmark_payload, dict):
        raise TypeError("benchmark subtree must resolve to a mapping")
    return OmegaConf.create(payload), model.model_validate(benchmark_payload)


def parse_environment_job(config: DictConfig) -> EnvironmentBenchmarkJobConfig:
    payload = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    if not isinstance(payload, dict):
        raise TypeError("environment benchmark configuration must resolve to a mapping")
    return EnvironmentBenchmarkJobConfig.model_validate(payload)


def benchmark_provenance(benchmark: StrictBenchmarkModel) -> dict[str, object]:
    repository_root = Path(to_absolute_path("."))
    return {
        **collect_repository_metadata(repository_root),
        "benchmark": benchmark.model_dump(mode="json"),
    }


def host_resource_provenance() -> dict[str, object]:
    gpu = None
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        gpu = {
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "torch_cuda": torch.version.cuda,
        }
    return {
        "cpu": {
            "processor": platform.processor(),
            "logical_cores": os.cpu_count(),
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
        },
        "gpu": gpu,
    }


def write_benchmark_artifacts(
    output_dir: Path,
    config: DictConfig,
    filename: str,
    report: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / filename, report)
    OmegaConf.save(config, output_dir / "resolved_config.yaml", resolve=True)
    write_tracked_diff(output_dir / "tracked_diff.patch", Path(to_absolute_path(".")))


def _validate_scales(scales: tuple[int, ...], name: str) -> None:
    if not scales or any(value <= 0 for value in scales):
        raise ValueError(f"{name} must contain positive integers")
    if len(set(scales)) != len(scales):
        raise ValueError(f"{name} must not contain duplicates")
