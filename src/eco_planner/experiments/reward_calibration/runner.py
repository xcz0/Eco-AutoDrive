"""Run original and calibrated diagnostics against an explicit lambda reference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from eco_planner._repository import REPOSITORY_ROOT
from eco_planner.analysis import publish, render_report
from eco_planner.artifacts import (
    write_json,
    write_npz,
)
from eco_planner.configuration import load_resolved_yaml_mapping
from eco_planner.experiments.lambda_identifiability.diagnostics import analyze
from eco_planner.rl.reward import PlannerRFTNoEnergyRewardConfig

from ..fixed_batch import (
    SHARED_SOURCES,
    calibrate,
    copy_sources,
    load_fixed_batch,
    raw_arrays,
    rescore,
    restore_runtime,
    verify_original_components,
    verify_reference,
    write_runtime_metadata,
)
from .config import CalibrationConfig
from .diagnostics import dynamic_range_audit


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
    copy_sources(
        output,
        (
            *SHARED_SOURCES,
            Path(__file__),
            Path(__file__).with_name("config.py"),
            Path(__file__).with_name("diagnostics.py"),
            REPOSITORY_ROOT / "scripts/experiments/__main__.py",
            REPOSITORY_ROOT / "src/eco_planner/experiments/lambda_identifiability/diagnostics.py",
        ),
    )
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
