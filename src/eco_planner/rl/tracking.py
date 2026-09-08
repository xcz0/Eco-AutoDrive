"""Training-only Fabric/MLflow boundary; research artifacts remain authoritative."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import TracebackType
from typing import Any
from uuid import uuid4

from lightning.fabric import Fabric
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, Field

from eco_planner._repository import REPOSITORY_ROOT
from eco_planner.rl.artifacts import PolicyProbeSummary, TrainingUpdateSummary
from eco_planner.rl.config import TrainingJobConfig, parse_training_config


class TrackingIdentity(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")
    run_id: str = Field(min_length=1)
    tracking_uri: str = Field(min_length=1)


def update_metrics(summary: TrainingUpdateSummary) -> dict[str, float]:
    """Keep units, reduction denominators and optional diagnostics explicit."""
    metrics: dict[str, float] = {}
    for name in (
        "mean_policy_loss",
        "mean_value_loss",
        "mean_entropy_loss",
        "mean_total_loss",
        "mean_approximate_kl",
        "mean_clip_fraction",
        "mean_entropy",
        "mean_explained_variance",
        "evaluated_minibatch_count",
        "optimizer_step_count",
        "final_learning_rate",
        "raw_advantage_mean",
        "raw_advantage_std",
        "normalized_advantage_mean",
        "normalized_advantage_std",
        "mean_value_target",
        "std_value_target",
        "kl_early_stopped",
        "kl_early_stop_trigger",
        "cumulative_kl_early_stop_count",
        "policy_ratio_mean",
        "policy_ratio_std",
        "policy_ratio_p95",
        "policy_ratio_max",
    ):
        value = getattr(summary, name)
        if value is not None:
            metrics[f"ppo/{name}"] = float(value)
    metrics["gradient/maximum_pre_clip_norm"] = summary.maximum_pre_clip_gradient_norm
    if summary.gradient_diagnostics is not None:
        for name, value in summary.gradient_diagnostics.model_dump().items():
            metrics[f"gradient/{name}"] = value
    metrics.update(
        {
            "reward/total_sum": summary.total_reward,
            "reward/base_sum": summary.base_reward,
            "reward/total_mean": summary.total_reward / summary.sample_count,
            "reward/base_mean": summary.base_reward / summary.sample_count,
            "reward/safety_gate_mean": summary.mean_safety_gate,
            "behavior/collision_transition_count": float(summary.collision_count),
            "behavior/collision_transition_fraction": summary.collision_count
            / summary.sample_count,
            "behavior/out_of_road_transition_count": float(summary.out_of_road_count),
            "behavior/out_of_road_transition_fraction": summary.out_of_road_count
            / summary.sample_count,
        }
    )
    for name, value in summary.reward_component_means.model_dump().items():
        metrics[f"reward_component/{name}_mean"] = value
    for name, value in summary.reward_diagnostic_means.model_dump().items():
        metrics[f"reward_diagnostic/{name}_mean"] = value
    for name in (
        "sample_count",
        "episode_count",
        "mean_episode_length",
        "route_completion_delta",
        "distance_m",
        "mean_speed_mps",
        "stopped_fraction",
        "maximum_position_error_m",
        "maximum_heading_error_rad",
    ):
        metrics[f"behavior/{name}"] = float(getattr(summary, name))
    for name in (
        "beta_alpha_mean",
        "beta_alpha_min",
        "beta_alpha_max",
        "beta_beta_mean",
        "beta_beta_min",
        "beta_beta_max",
        "action_mean",
        "action_std",
        "action_min",
        "action_max",
    ):
        for dimension, value in enumerate(getattr(summary, name)):
            metrics[f"policy/{name}/dim_{dimension}"] = value
    metrics["policy/mean_state_value"] = summary.mean_state_value
    metrics["policy/std_state_value"] = summary.std_state_value
    for name in (
        "native_step_energy_total_ml",
        "executed_fuel_proxy_total_ml",
        "executed_fuel_proxy_distance_m",
        "executed_fuel_proxy_ml_per_km",
    ):
        value = getattr(summary, name)
        if value is not None:
            metrics[f"energy/{name}"] = value
    return metrics


def _flatten(value: Any, prefix: str = "") -> dict[str, str]:
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            result.update(_flatten(child, f"{prefix}.{key}" if prefix else key))
        return result
    if isinstance(value, (list, tuple)):
        result = {}
        for index, child in enumerate(value):
            result.update(_flatten(child, f"{prefix}.{index}"))
        return result
    return {prefix: str(value) if isinstance(value, str) else json.dumps(value)}


def _scientific_params(config: TrainingJobConfig) -> dict[str, str]:
    payload = config.model_dump(mode="json", exclude={"tracking", "resources", "name"})
    payload["training"].pop("update_count")
    payload["training"].pop("resume_checkpoint_path")
    return _flatten(payload)


def _tracking_uri(uri: str) -> str:
    if uri.startswith("sqlite:///"):
        path = Path(uri.removeprefix("sqlite:///"))
        path = (REPOSITORY_ROOT / path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{path.as_posix()}"
    return uri


class TrainingTracking:
    def __init__(self, config: TrainingJobConfig, output_dir: Path) -> None:
        self.config = config
        self.output_dir = output_dir
        self.identity: TrackingIdentity | None = None
        self.logger: MLFlowLogger | None = None
        self.fabric: Fabric | None = None
        self.invocation = f"invocations/{uuid4().hex}"
        self._owns_run = False

    def __enter__(self) -> TrainingTracking:
        return self

    def start_new(self) -> None:
        if self.config.tracking.enabled and self.config.training.resume_checkpoint_path is None:
            self._open(None)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self.logger is None or not self._owns_run:
            return
        status = "FINISHED" if exc_type is None else "FAILED"
        if exc_type is not None and issubclass(exc_type, (KeyboardInterrupt, SystemExit)):
            status = "KILLED"
        try:
            self.logger.finalize(status)
        except Exception:
            if exc is None:
                raise
            # A failed backend must not replace the original training exception.
            import logging

            logging.getLogger(__name__).exception("Could not finalize the failed MLflow run")

    def _open(self, identity: TrackingIdentity | None) -> None:
        config = self.config.tracking
        uri = _tracking_uri(config.tracking_uri)
        if identity is not None and identity.tracking_uri != uri:
            raise ValueError("resume MLflow tracking store differs from checkpoint")
        location = config.artifact_location
        if location is not None and "://" not in location:
            location = (REPOSITORY_ROOT / location).resolve().as_uri()
        self.logger = MLFlowLogger(
            experiment_name=config.experiment_name,
            tracking_uri=uri,
            artifact_location=location,
            run_name=config.run_name or f"{self.config.name}-seed-{self.config.runtime.seed}",
            tags={**config.tags, "reward_profile": self.config.reward.name},
            run_id=identity.run_id if identity is not None else None,
            synchronous=True,
        )
        run_id = self.logger.run_id
        if run_id is None:
            raise RuntimeError("MLflow did not create a run")
        self.identity = TrackingIdentity(run_id=run_id, tracking_uri=uri)
        if identity is None:
            self._owns_run = True
            self.artifact("resolved_config.yaml")

    def attach(
        self,
        fabric: Fabric,
        history: Sequence[TrainingUpdateSummary],
        identity: TrackingIdentity | None,
    ) -> None:
        self.fabric = fabric
        if not self.config.tracking.enabled:
            self.identity = identity
            return
        existing = identity is not None
        if self.logger is None:
            self._open(identity)
        assert self.logger is not None and self.identity is not None
        client = self.logger.experiment
        run_id = self.identity.run_id
        params = _scientific_params(self.config)
        if existing:
            stored = client.get_run(run_id).data.params
            prefix = (
                "config."
                if any(k.startswith("config.") for k in stored)
                else "continuation_config."
            )
            previous = {
                k.removeprefix(prefix): v for k, v in stored.items() if k.startswith(prefix)
            }
            if previous != params:
                raise ValueError("resume changes immutable MLflow experiment parameters")
        else:
            prefix = "config."
            path = self.config.training.resume_checkpoint_path
            if path is not None:
                original = (REPOSITORY_ROOT / path).resolve().parent / "resolved_config.yaml"
                if original.is_file():
                    raw = OmegaConf.load(original)
                    if not isinstance(raw, DictConfig):
                        raise TypeError("original resolved config must be a mapping")
                    raw["tracking"] = self.config.tracking.model_dump()
                    historical = _scientific_params(parse_training_config(raw))
                    if historical != params:
                        raise ValueError("resume changes original experiment parameters")
                    client.log_artifact(run_id, str(original), "history")
                    provenance = "original_resolved_config"
                else:
                    prefix = "continuation_config."
                    provenance = "historical_parameters_unrecorded"
                client.set_tag(run_id, "history.parameter_provenance", provenance)
            self._params({f"{prefix}{key}": value for key, value in params.items()})
        # Check the entire history before changing an existing run or writing any repairs.
        missing = (
            self._missing_history(history) if existing else [update_metrics(s) for s in history]
        )
        client.update_run(run_id, status="RUNNING")
        self._owns_run = True
        fabric.loggers.append(self.logger)
        for summary, metrics in zip(history, missing, strict=True):
            if metrics:
                fabric.log_dict(metrics, step=summary.update_index)
        invocation_params = {
            "training": {
                "update_count": self.config.training.update_count,
                "resume_checkpoint_path": self.config.training.resume_checkpoint_path,
            },
            "resources": self.config.resources.model_dump() if self.config.resources else None,
            "tracking": self.config.tracking.model_dump(exclude={"tracking_uri"}),
        }
        self._params(_flatten(invocation_params, self.invocation.replace("/", ".")))
        self.artifact("resolved_config.yaml")

    def _params(self, params: Mapping[str, str]) -> None:
        from mlflow.entities import Param

        assert self.logger is not None and self.identity is not None
        values = [Param(key, value) for key, value in params.items()]
        for start in range(0, len(values), 100):
            self.logger.experiment.log_batch(
                self.identity.run_id, params=values[start : start + 100], synchronous=True
            )

    def _missing_history(
        self,
        history: Sequence[TrainingUpdateSummary],
    ) -> list[dict[str, float]]:
        assert self.logger is not None and self.identity is not None
        client = self.logger.experiment
        run_id = self.identity.run_id
        expected = {s.update_index: update_metrics(s) for s in history}
        missing = {step: dict(metrics) for step, metrics in expected.items()}
        for key in client.get_run(run_id).data.metrics:
            if key.startswith("policy/probe_"):
                continue
            for metric in client.get_metric_history(run_id, key):
                if metric.step not in expected:
                    raise ValueError("MLflow run is ahead of the resume checkpoint")
                if key not in expected[metric.step] or expected[metric.step][key] != metric.value:
                    raise ValueError("MLflow metric conflicts with checkpoint history")
                missing[metric.step].pop(key, None)
        return [missing[s.update_index] for s in history]

    def artifact(self, name: str) -> None:
        if self.logger is not None and self.identity is not None:
            self.logger.experiment.log_artifact(
                self.identity.run_id,
                str(self.output_dir / name),
                self.invocation,
            )

    def runtime_metadata(self) -> None:
        if self.identity is None:
            return
        metadata = json.loads(
            (self.output_dir / "runtime_metadata.json").read_text(encoding="utf-8")
        )
        from eco_planner.artifacts import write_json

        metadata["tracking"] = self.identity.model_dump()
        write_json(self.output_dir / "runtime_metadata.json", metadata)
        if self.logger is None:
            return
        self.artifact("runtime_metadata.json")
        self.artifact("tracked_diff.patch")
        for key, value in {
            "git.commit": metadata["git_head"],
            "git.branch": metadata["git_branch"],
        }.items():
            self.logger.experiment.set_tag(self.identity.run_id, key, value)

    def update(self, summary: TrainingUpdateSummary) -> None:
        if self.logger is None:
            return
        assert self.fabric is not None
        self.fabric.log_dict(update_metrics(summary), step=summary.update_index)
        if (summary.update_index + 1) % self.config.tracking.checkpoint_interval == 0:
            self.artifact(f"policy-update-{summary.update_index:03d}.pt")
            self.artifact("training-state.ckpt")

    def probe(self, label: str, probe: PolicyProbeSummary, step: int) -> None:
        if self.logger is None:
            return
        assert self.fabric is not None
        metrics = {
            f"policy/probe_{label}/{name}/scenario_{scenario}/dim_{dimension}": value
            for name, pairs in probe.model_dump().items()
            for scenario, pair in enumerate(pairs)
            for dimension, value in enumerate(pair)
        }
        self.fabric.log_dict(metrics, step=step)
