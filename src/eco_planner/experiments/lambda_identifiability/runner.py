"""Run matched lambda diagnostics on an explicitly collected fixed batch."""

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from eco_planner._repository import REPOSITORY_ROOT
from eco_planner.analysis.runner import publish
from eco_planner.artifacts import write_json, write_npz
from eco_planner.configuration import load_resolved_yaml_mapping
from eco_planner.experiments.fixed_batch.artifacts import (
    SHARED_SOURCES,
    copy_sources,
    load_fixed_batch,
)
from eco_planner.experiments.fixed_batch.calibration import verify_original_components
from eco_planner.experiments.fixed_batch.runtime import restore_runtime, write_runtime_metadata
from eco_planner.experiments.lambda_identifiability.config import IdentifiabilityConfig
from eco_planner.experiments.lambda_identifiability.diagnostics import analyze
from eco_planner.rl.reward.config import PlannerRFTNoEnergyRewardConfig


def run(source: Path, config_path: Path, output: Path, *, figures: bool = True) -> dict[str, Any]:
    study = IdentifiabilityConfig.model_validate(load_resolved_yaml_mapping(config_path))
    batch = load_fixed_batch(source)
    base = PlannerRFTNoEnergyRewardConfig.model_validate(batch.resolved_config["reward"])
    verify_original_components(batch.episodes, base)
    runtime = restore_runtime(source, batch)
    output.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(OmegaConf.create(study.model_dump()), output / "diagnostic_config.yaml")
    OmegaConf.save(OmegaConf.create(batch.resolved_config), output / "resolved_config.yaml")
    write_json(output / "sample_index.json", {"samples": batch.samples})
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
    summary, arrays = analyze(
        runtime.updater, batch.episodes, base, study.lambdas, study.quantiles, batch.scenario_ids
    )
    runtime.verify_unchanged()
    summary.update(
        {
            "status": "completed",
            "source_batch": str(source.resolve()),
            "initial_policy_hash": runtime.initial_policy_hash,
            "policy_unchanged": True,
            "episode_count": len(batch.episodes),
            "batch_source": "reused fixed source batch",
        }
    )
    write_npz(output / "diagnostics.npz", arrays)
    write_json(output / "summary.json", summary)
    publish("lambda-identifiability", output, output, figures=figures)
    return {
        "status": "completed",
        "output_dir": str(output),
        "sample_count": len(batch.samples),
        "optimizer_steps": 0,
    }
