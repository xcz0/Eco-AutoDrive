"""Run objective decomposition on an explicitly collected fixed batch."""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from eco_planner._repository import REPOSITORY_ROOT
from eco_planner.analysis import publish
from eco_planner.artifacts import (
    write_json,
    write_npz,
)
from eco_planner.configuration import load_resolved_yaml_mapping
from eco_planner.rl import PlannerRFTNoEnergyRewardConfig

from ..fixed_batch import (
    ADVANTAGE_FORMS,
    SHARED_SOURCES,
    calibrate,
    copy_sources,
    load_fixed_batch,
    raw_arrays,
    rescore,
    restore_runtime,
    verify_expected_calibration,
    write_runtime_metadata,
)
from .config import DecompositionConfig
from .diagnostics import analyze_decomposition


def run(source: Path, config_path: Path, output: Path, *, figures: bool = True) -> dict:
    study = DecompositionConfig.model_validate(load_resolved_yaml_mapping(config_path))
    batch = load_fixed_batch(source)
    config = batch.resolved_config
    base = PlannerRFTNoEnergyRewardConfig.model_validate(config["reward"])
    episodes, samples = batch.episodes, batch.samples
    verify_original_components(episodes, base)
    calibrated = calibrate(raw_arrays(episodes), base, study)
    calibration_checks = verify_expected_calibration(calibrated, study)
    scenario_ids = batch.scenario_ids
    runtime = restore_runtime(source, batch)
    updater, initial_hash = runtime.updater, runtime.initial_policy_hash

    calibrated_episodes = [rescore(episode, calibrated) for episode in episodes]
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "sample_index.json", {"samples": samples})
    write_json(output / "calibration_verification.json", calibration_checks)
    OmegaConf.save(OmegaConf.create(study.model_dump()), output / "diagnostic_config.yaml")
    resolved = {**config, "reward": calibrated.model_dump()}
    OmegaConf.save(OmegaConf.create(resolved), output / "resolved_config.yaml")
    write_runtime_metadata(output, source, batch, runtime)
    copy_sources(
        output,
        (
            *SHARED_SOURCES,
            Path(__file__),
            Path(__file__).with_name("config.py"),
            Path(__file__).with_name("diagnostics.py"),
            REPOSITORY_ROOT / "scripts/experiments/__main__.py",
        ),
    )
    print(
        f"Running Task C decomposition: {len(samples)} fixed transitions, "
        f"{len(study.lambdas) + 2} arms x {len(ADVANTAGE_FORMS)} advantage forms, "
        "backward only.",
        flush=True,
    )
    summary, arrays = analyze_decomposition(
        updater,
        calibrated_episodes,
        calibrated,
        study.lambdas,
        study.quantiles,
        scenario_ids,
        study.gate,
    )
    runtime.verify_unchanged()
    write_npz(output / "diagnostics.npz", arrays)
    summary.update(
        {
            "status": "completed",
            "episode_count": len(episodes),
            "policy_unchanged": True,
            "source_batch": str(source.resolve()),
            "calibrated_reward": calibrated.model_dump(),
            "calibration_verification": calibration_checks,
            "initial_policy_hash": initial_hash,
            "decision": "Gate C evaluated with Issue #94 engineering thresholds; "
            "Task D entry is decided from the recorded gate verdict",
        }
    )
    write_json(output / "summary.json", summary)
    publish("objective-decomposition", output, output, figures=figures)
    return {
        "status": "completed",
        "output_dir": str(output),
        "sample_count": len(samples),
        "optimizer_steps": 0,
        "gate_c_passed": summary["gate"]["gate_c_passed"],
        "attribution": summary["gate"]["attribution"],
    }
