"""Training-only Fabric/MLflow boundary; research artifacts remain authoritative."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from types import TracebackType
from typing import Any

from lightning.fabric import Fabric
from lightning.pytorch.loggers import MLFlowLogger

from eco_planner._repository import REPOSITORY_ROOT

from .artifacts import PolicyProbeSummary, TrainingUpdateSummary
from .artifacts.metadata import TrackingIdentity as TrackingIdentity
from .artifacts.metadata import TrainingRuntimeMetadata
from .config import TrainingJobConfig


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

    def __enter__(self) -> TrainingTracking:
        return self

    def start_new(self) -> None:
        if self.config.tracking.enabled and self.logger is None:
            self._open()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self.logger is None:
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

    def _open(self) -> None:
        config = self.config.tracking
        uri = _tracking_uri(config.tracking_uri)
        location = config.artifact_location
        if location is not None and "://" not in location:
            location = (REPOSITORY_ROOT / location).resolve().as_uri()
        self.logger = MLFlowLogger(
            experiment_name=config.experiment_name,
            tracking_uri=uri,
            artifact_location=location,
            run_name=config.run_name or f"{self.config.name}-seed-{self.config.runtime.seed}",
            tags={**config.tags, "reward_profile": self.config.reward.name},
            synchronous=True,
        )
        run_id = self.logger.run_id
        if run_id is None:
            raise RuntimeError("MLflow did not create a run")
        self.identity = TrackingIdentity(run_id=run_id, tracking_uri=uri)
        self._params(_flatten(self.config.model_dump(mode="json"), "config"))
        self.artifact("resolved_config.yaml")

    def attach(self, fabric: Fabric, source_identity: TrackingIdentity | None) -> None:
        """Attach this invocation's run; a checkpoint identity is only a source reference."""

        self.fabric = fabric
        if not self.config.tracking.enabled:
            self.identity = source_identity
            return
        self.start_new()
        assert self.logger is not None and self.identity is not None
        client = self.logger.experiment
        run_id = self.identity.run_id
        checkpoint = self.config.training.resume_checkpoint_path
        if checkpoint is not None:
            client.set_tag(run_id, "resume.checkpoint_path", checkpoint)
        if source_identity is not None:
            client.set_tag(run_id, "resume.source_run_id", source_identity.run_id)
            client.set_tag(run_id, "resume.source_tracking_uri", source_identity.tracking_uri)
        fabric.loggers.append(self.logger)

    def _params(self, params: Mapping[str, str]) -> None:
        from mlflow.entities import Param

        assert self.logger is not None and self.identity is not None
        values = [Param(key, value) for key, value in params.items()]
        for start in range(0, len(values), 100):
            self.logger.experiment.log_batch(
                self.identity.run_id, params=values[start : start + 100], synchronous=True
            )

    def artifact(self, name: str) -> None:
        if self.logger is not None and self.identity is not None:
            self.logger.experiment.log_artifact(
                self.identity.run_id,
                str(self.output_dir / name),
            )

    def runtime_metadata(self) -> None:
        if self.identity is None:
            return
        metadata = json.loads(
            (self.output_dir / "runtime_metadata.json").read_text(encoding="utf-8")
        )
        from eco_planner.artifacts import write_json

        metadata["tracking"] = self.identity.model_dump()
        validated = TrainingRuntimeMetadata.model_validate_json(json.dumps(metadata))
        write_json(self.output_dir / "runtime_metadata.json", validated)
        if self.logger is None:
            return
        self.artifact("runtime_metadata.json")
        if (self.output_dir / "overrides.yaml").is_file():
            self.artifact("overrides.yaml")
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
