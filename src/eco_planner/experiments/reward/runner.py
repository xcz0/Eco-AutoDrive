"""Choose reward comparisons; rollout and reward operations belong to RL."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from tensordict import cat

from eco_planner.analysis import publish
from eco_planner.analysis.reward import dynamic_range_audit
from eco_planner.artifacts import write_json, write_npz
from eco_planner.configuration import load_resolved_yaml_mapping
from eco_planner.jobs import compose_job_config
from eco_planner.rl.config import parse_training_config
from eco_planner.rl.reward.calibration import (
    MOTION_LIMITS,
    apply_energy_band,
    calibrate,
    raw_arrays,
    rescore,
    scored_arrays,
    verify_original_components,
)
from eco_planner.rl.reward.config import PlannerRFTNoEnergyRewardConfig
from eco_planner.rl.reward.reweighting import COMPONENTS, reward_profile, reweight
from eco_planner.rl.rollout.collection import collect as collect_batch
from eco_planner.rl.rollout.fixed_batch import load_fixed_batch

from .config import RewardStudyConfig


def collect(config_path: Path, output: Path) -> dict:
    from pydantic import BaseModel, ConfigDict

    class CollectionConfig(BaseModel):
        model_config = ConfigDict(extra="forbid")
        job: str
        overrides: list[str]

    study = CollectionConfig.model_validate(load_resolved_yaml_mapping(config_path))
    resolved = compose_job_config(study.job, study.overrides)
    return collect_batch(resolved, parse_training_config(resolved), output)


def run(source: Path, config_path: Path, output: Path, *, figures: bool = True) -> dict:
    study = RewardStudyConfig.model_validate(load_resolved_yaml_mapping(config_path))
    batch = load_fixed_batch(source)
    base = PlannerRFTNoEnergyRewardConfig.model_validate(batch.resolved_config["reward"])
    verify_original_components(batch.episodes, base)
    raw = raw_arrays(batch.episodes)
    calibrated = calibrate(raw, base, study.calibration)
    profiles = {"original": base, "calibrated": calibrated}
    if study.energy_band is not None:
        profiles["band"] = apply_energy_band(calibrated, batch.episodes, study.energy_band)
    cycles = np.asarray([s["planning_cycle_index"] for s in batch.samples], dtype=np.int64)
    audit, audit_arrays = dynamic_range_audit(
        raw,
        base,
        calibrated,
        study.quantiles,
        scored_arrays(raw, base),
        scored_arrays(raw, calibrated),
        MOTION_LIMITS,
        batch.scenario_ids,
        cycles,
    )
    arrays = {"scenario_index": batch.scenario_ids}
    arms = []
    for representation in study.representations:
        profile = profiles[representation]
        episodes = [rescore(e, profile) for e in batch.episodes]
        for weight in study.lambdas:
            selected = reward_profile(profile, weight)
            matched = [reweight(e, selected) for e in episodes]
            label = f"{representation}_lambda_{weight:g}"
            values = cat([e.audit for e in matched])
            for key in (
                *[f"reward_component_{c}" for c in COMPONENTS],
                "reward_safety_gate",
                "reward_total",
            ):
                arrays[f"{label}__{key}"] = values[key].numpy().reshape(-1)
            arms.append({"label": label, "reward_profile": selected.model_dump()})
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "sample_index.json", {"samples": batch.samples})
    write_json(output / "audit.json", audit)
    write_npz(output / "audit.npz", audit_arrays)
    write_npz(output / "diagnostics.npz", arrays)
    OmegaConf.save(OmegaConf.create(study.model_dump()), output / "diagnostic_config.yaml")
    summary = {
        "status": "completed",
        "kind": "reward",
        "arms": arms,
        "source_batch": str(source.resolve()),
        "sample_count": len(batch.samples),
        "quantiles": study.quantiles,
        "optimizer_steps": 0,
        "calibrated_reward": calibrated.model_dump(),
        "interpretation": "Batch-relative calibration; no actor backward or learned behavior.",
    }
    write_json(output / "summary.json", summary)
    publish("reward", output, output, figures=figures)
    return {"status": "completed", "output_dir": str(output), "sample_count": len(batch.samples)}
