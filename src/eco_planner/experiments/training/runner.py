"""Explicit training-run diagnostics and matched policy evaluations."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, model_validator

from eco_planner.analysis import publish
from eco_planner.analysis.evaluation import arm_outcomes, paired
from eco_planner.analysis.training import beta_probe_statistics
from eco_planner.artifacts import write_json
from eco_planner.configuration import load_resolved_yaml_mapping
from eco_planner.evaluation.artifacts import load_job_summary
from eco_planner.experiments.protocol.composition import (
    compose_policy_evaluation_config,
    validate_evaluation,
)
from eco_planner.experiments.protocol.config import load_protocol
from eco_planner.jobs import run_evaluation_job
from eco_planner.rl.artifacts import TrainingRunSummary
from eco_planner.rl.optimization.update_diagnostics import (
    extract_arm_metrics,
    post_update_kl_series,
)


class DiagnosticConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    training_summaries: list[Path] = Field(min_length=1)
    training_seeds: list[int] = Field(min_length=1)
    mc_draws: int = Field(gt=0)
    mc_seed: int = Field(ge=0)


class EvaluationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    arm: str
    training_summary: Path
    checkpoint_label: Literal["initial", "final"]
    checkpoint_path: Path
    deterministic_evaluation_dir: Path | None


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    protocol: Path
    records: list[EvaluationRecord] = Field(min_length=1)
    policy_action_seeds: list[int] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_seeds(self) -> EvaluationConfig:
        if min(self.policy_action_seeds) < 0 or len(set(self.policy_action_seeds)) != len(
            self.policy_action_seeds
        ):
            raise ValueError("policy action seeds must be nonnegative and unique")
        return self


def diagnose(config_path: Path, output: Path, *, figures: bool = True) -> dict:
    study = DiagnosticConfig.model_validate(load_resolved_yaml_mapping(config_path))
    records, seen = [], set()
    for path in study.training_summaries:
        source = (config_path.parent / path).resolve()
        if source in seen:
            raise ValueError("duplicate training summary")
        seen.add(source)
        summary = TrainingRunSummary.model_validate_json(source.read_text(encoding="utf-8"))
        if summary.training_seed not in study.training_seeds:
            raise ValueError("training seed absent from diagnostic design")
        resolved = load_resolved_yaml_mapping(source.parent / "resolved_config.yaml")
        metrics = None
        if summary.status == "completed":
            metrics = extract_arm_metrics(
                source.parent,
                max_gradient_norm=resolved["ppo"]["max_gradient_norm"],
                update_count=len(summary.updates),
            )
            metrics.update(
                post_update_kl_series(
                    source.parent,
                    update_count=len(summary.updates),
                    mc_draws=study.mc_draws,
                    mc_seed=study.mc_seed,
                )
            )
        records.append(
            {
                "source": str(source),
                "training_seed": summary.training_seed,
                "status": summary.status,
                "metrics": metrics,
                "training": summary.model_dump(mode="json"),
            }
        )
    output.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(
        OmegaConf.create(study.model_dump(mode="json")), output / "diagnostic_config.yaml"
    )
    write_json(
        output / "summary.json",
        {
            "kind": "training-diagnostics",
            "status": "completed",
            "runs": records,
            "training_seeds": study.training_seeds,
            "missing_seeds": sorted(
                set(study.training_seeds) - {r["training_seed"] for r in records}
            ),
        },
    )
    return publish("training", output, output, figures=figures)


def evaluate(config_path: Path, output: Path, *, figures: bool = True) -> dict:
    study = EvaluationConfig.model_validate(load_resolved_yaml_mapping(config_path))
    protocol = load_protocol(config_path.parent / study.protocol)
    output.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(
        OmegaConf.create(study.model_dump(mode="json")), output / "evaluation_config.yaml"
    )
    records, seen = [], set()
    for record in study.records:
        root = config_path.parent
        training = TrainingRunSummary.model_validate_json(
            (root / record.training_summary).read_text(encoding="utf-8")
        )
        if (
            record.arm not in protocol.arms
            or training.reward_profile != protocol.arms[record.arm].reward_profile
        ):
            raise ValueError("training reward differs from declared arm")
        if training.training_seed not in protocol.training.seeds:
            raise ValueError("training seed absent from protocol")
        key = (record.arm, training.training_seed, record.checkpoint_label)
        if key in seen:
            raise ValueError("duplicate arm/seed/checkpoint")
        seen.add(key)
        checkpoint = (root / record.checkpoint_path).resolve()
        expected_hash = (
            training.initial_policy_hash
            if record.checkpoint_label == "initial"
            else training.final_policy_hash
        )
        label = f"{record.arm}-seed-{training.training_seed}-{record.checkpoint_label}"
        destination = output / label
        deterministic = (
            (root / record.deterministic_evaluation_dir).resolve()
            if record.deterministic_evaluation_dir is not None
            else destination / "deterministic"
        )

        def evaluation_at(
            directory: Path,
            action_seed: int | None,
            reuse: bool,
            record=record,
            checkpoint=checkpoint,
            expected_hash=expected_hash,
        ):
            if not reuse:
                overrides = (
                    ["evaluation.policy_checkpoint.action_mode=mean"]
                    if action_seed is None
                    else [
                        "evaluation.policy_checkpoint.action_mode=sample",
                        f"evaluation.policy_checkpoint.policy_action_seed={action_seed}",
                    ]
                )
                resolved, _ = compose_policy_evaluation_config(
                    protocol, record.arm, record.checkpoint_label, checkpoint, overrides
                )
                run_evaluation_job(resolved, directory)
            summary = load_job_summary(directory / "summary.json")
            validate_evaluation(protocol, summary)
            identity = summary.policy_checkpoint
            if (
                identity is None
                or identity.policy_hash != expected_hash
                or identity.label != record.checkpoint_label
            ):
                raise ValueError("evaluation checkpoint differs from declared training state")
            expected_mode = "mean" if action_seed is None else "sample"
            provenance = load_resolved_yaml_mapping(directory / "resolved_config.yaml")[
                "evaluation"
            ]["policy_checkpoint"]
            if provenance["action_mode"] != expected_mode or (
                action_seed is not None and provenance["policy_action_seed"] != action_seed
            ):
                raise ValueError("evaluation action mode differs from requested condition")
            return summary

        baseline = evaluation_at(
            deterministic, None, record.deterministic_evaluation_dir is not None
        )
        stochastic = {}
        for seed in study.policy_action_seeds:
            summary = evaluation_at(destination / f"action-{seed}", seed, False)
            stochastic[str(seed)] = {
                "outcomes": arm_outcomes(summary),
                "comparison": paired(baseline, summary),
            }
        probe = (
            training.probe_before if record.checkpoint_label == "initial" else training.probe_after
        )
        records.append(
            {
                "arm": record.arm,
                "training_seed": training.training_seed,
                "checkpoint_label": record.checkpoint_label,
                "checkpoint_hash": expected_hash,
                "deterministic": arm_outcomes(baseline),
                "stochastic": stochastic,
                "probe": beta_probe_statistics(probe.model_dump(mode="json")),
            }
        )
    write_json(
        output / "summary.json",
        {
            "kind": "training-evaluation",
            "status": "completed",
            "runs": records,
            "policy_action_seeds": study.policy_action_seeds,
            "training_seeds": protocol.training.seeds,
            "missing_seeds": sorted(
                set(protocol.training.seeds) - {r["training_seed"] for r in records}
            ),
            "interpretation": (
                "Stochastic-minus-deterministic outcomes with matched scenario and diffusion noise."
            ),
        },
    )
    return publish("training", output, output, figures=figures)
