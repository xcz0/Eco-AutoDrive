"""Configuration, diagnostics and execution for calibration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, model_validator

from eco_planner.analysis import publish, render_report, statistics
from eco_planner.artifacts import write_json, write_npz
from eco_planner.configuration import load_resolved_yaml_mapping
from eco_planner.experiments.reward.fixed_batch import (
    calibrate,
    load_fixed_batch,
    raw_arrays,
    rescore,
    restore_runtime,
    verify_original_components,
    verify_reference,
    write_runtime_metadata,
)
from eco_planner.experiments.reward.fixed_batch.calibration import MOTION_LIMITS, scored_arrays
from eco_planner.experiments.reward.lambda_identifiability import analyze
from eco_planner.rl.reward import PlannerRFTNoEnergyRewardConfig


class CalibrationConfig(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)
    progress_target_score: StrictFloat = Field(gt=0.0, lt=1.0)
    comfort_target_score: StrictFloat = Field(gt=0.0, lt=1.0)
    lambdas: list[StrictFloat] = Field(min_length=2)
    quantiles: list[StrictFloat] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_axes(self) -> CalibrationConfig:
        if self.lambdas[0] != 0 or sorted(set(self.lambdas)) != self.lambdas:
            raise ValueError("lambdas must start at zero and strictly increase")
        if (
            self.quantiles[0] != 0
            or self.quantiles[-1] != 1
            or sorted(set(self.quantiles)) != self.quantiles
        ):
            raise ValueError("quantiles must increase from zero to one")
        return self


def dynamic_range_audit(
    raw: dict[str, np.ndarray],
    base: PlannerRFTNoEnergyRewardConfig,
    calibrated: PlannerRFTNoEnergyRewardConfig,
    study: CalibrationConfig,
    scenario_ids: np.ndarray,
    cycle_ids: np.ndarray,
) -> tuple[dict, dict[str, np.ndarray]]:
    before, after = scored_arrays(raw, base), scored_arrays(raw, calibrated)
    arrays = {**raw, "scenario_index": scenario_ids, "planning_cycle_index": cycle_ids}
    arrays.update(
        {
            f"{label}_{k}": v
            for label, s in (("original", before), ("calibrated", after))
            for k, v in s.items()
        }
    )

    def describe(mask: np.ndarray) -> dict:
        result: dict = {"sample_count": int(mask.sum()), "raw": {}, "scores": {}}
        for key, value in raw.items():
            measured = np.abs(value) if key in MOTION_LIMITS else value
            result["raw"][key] = statistics(measured[mask], study.quantiles)
            if key in MOTION_LIMITS:
                limit = getattr(base.comfort, MOTION_LIMITS[key])
                result["raw"][key]["original_limit_exceeded_fraction"] = float(
                    np.mean(measured[mask] > limit)
                )
        for label, scores in (("original", before), ("calibrated", after)):
            result["scores"][label] = {
                key: {
                    **statistics(value[mask], study.quantiles),
                    "zero_fraction": float(np.mean(value[mask] == 0)),
                    "one_fraction": float(np.mean(value[mask] == 1)),
                    **(
                        {
                            "minimum_fraction_including_ties": float(
                                np.mean(value[mask] == scores["comfort"][mask])
                            )
                        }
                        if key in MOTION_LIMITS
                        else {}
                    ),
                }
                for key, value in scores.items()
            }
        return result

    return {
        "all": describe(np.ones(len(scenario_ids), dtype=bool)),
        "per_scenario": {str(i): describe(scenario_ids == i) for i in np.unique(scenario_ids)},
        "per_planning_cycle": {str(i): describe(cycle_ids == i) for i in np.unique(cycle_ids)},
        "interpretation": (
            "Distribution-relative smoothness; original limits remain audit references, "
            "not redefined physical comfort standards. All transitions retained."
        ),
    }, arrays


def run(
    source: Path, reference: Path, config_path: Path, output: Path, *, figures: bool = True
) -> dict:
    study = CalibrationConfig.model_validate(load_resolved_yaml_mapping(config_path))
    batch = load_fixed_batch(source)
    config = batch.resolved_config
    reference_summary = verify_reference(reference, source, batch)
    if study.lambdas != [a["lambda"] for a in reference_summary["arms"]]:
        raise ValueError("lambda axes must match the reference diagnostic")
    base = PlannerRFTNoEnergyRewardConfig.model_validate(config["reward"])
    episodes, samples = batch.episodes, batch.samples
    verify_original_components(episodes, base)
    raw = raw_arrays(episodes)
    calibrated = calibrate(raw, base, study)
    scenario_ids = batch.scenario_ids
    cycles = np.asarray([s["planning_cycle_index"] for s in samples], dtype=np.int64)
    audit, audit_arrays = dynamic_range_audit(raw, base, calibrated, study, scenario_ids, cycles)
    runtime = restore_runtime(source, batch)
    updater, initial_hash = runtime.updater, runtime.initial_policy_hash

    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "sample_index.json", {"samples": samples})
    write_json(output / "audit.json", audit)
    write_npz(output / "audit.npz", audit_arrays)
    OmegaConf.save(OmegaConf.create(study.model_dump()), output / "calibration_config.yaml")
    write_runtime_metadata(output, source, batch, runtime)
    results, original_errors = {}, {}
    for label, profile in (("original", base), ("calibrated", calibrated)):
        print(f"Running {label}: {len(samples)} fixed transitions, backward only.", flush=True)
        matched = episodes if label == "original" else [rescore(e, profile) for e in episodes]
        summary, arrays = analyze(
            updater, matched, profile, study.lambdas, study.quantiles, scenario_ids
        )
        if label == "original":
            with np.load(reference / "diagnostics.npz", allow_pickle=False) as historical:
                for key, value in arrays.items():
                    np.testing.assert_allclose(
                        value, historical[key], rtol=1e-5, atol=1e-6, err_msg=key
                    )
                    original_errors[key] = float(
                        np.max(np.abs(value.astype(float) - historical[key]))
                    )
        destination = output / label
        destination.mkdir()
        resolved = {**config, "reward": profile.model_dump()}
        OmegaConf.save(OmegaConf.create(resolved), destination / "resolved_config.yaml")
        write_json(destination / "summary.json", summary)
        write_npz(destination / "diagnostics.npz", arrays)
        (destination / "report.md").write_text(
            render_report(summary, batch_origin="Reused fixed source batch").replace(
                "(sample_index.json)", "(../sample_index.json)"
            ),
            encoding="utf-8",
        )
        results[label] = summary
    runtime.verify_unchanged()
    summary = {
        "status": "completed",
        "sample_count": len(samples),
        "episode_count": len(episodes),
        "optimizer_steps": updater.completed_optimizer_steps,
        "policy_unchanged": True,
        "source_batch": str(source.resolve()),
        "initial_policy_hash": initial_hash,
        "reference": str(reference.resolve()),
        "calibrated_reward": calibrated.model_dump(),
        "original_replay_max_abs_errors": original_errors,
        "decision": "continuous evidence only; user decides whether to enter Task C",
        "comparisons": {label: result["pairs"] for label, result in results.items()},
    }
    write_json(output / "summary.json", summary)
    publish("reward-calibration", output, output, figures=figures)
    return {"status": "completed", "output_dir": str(output), "sample_count": len(samples)}
