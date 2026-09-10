"""Run critic/GAE ablation against an explicit same-batch decomposition reference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from eco_planner._repository import REPOSITORY_ROOT
from eco_planner.analysis.runner import publish
from eco_planner.artifacts import (
    write_json,
    write_npz,
)
from eco_planner.configuration import load_resolved_yaml_mapping
from eco_planner.experiments.critic_gae_ablation.config import AblationConfig
from eco_planner.experiments.critic_gae_ablation.diagnostics import (
    CREDIT_FORMS,
    analyze_critic_gae_ablation,
)
from eco_planner.experiments.fixed_batch.artifacts import (
    SHARED_SOURCES,
    copy_sources,
    load_fixed_batch,
    verify_reference,
)
from eco_planner.experiments.fixed_batch.calibration import (
    calibrate,
    raw_arrays,
    rescore,
    verify_expected_calibration,
    verify_original_components,
)
from eco_planner.experiments.fixed_batch.gradients import ADVANTAGE_FORMS, GRADIENT_GROUPS
from eco_planner.experiments.fixed_batch.runtime import restore_runtime, write_runtime_metadata
from eco_planner.rl.reward.config import PlannerRFTNoEnergyRewardConfig

_VALUE_KEYS = (
    "reward",
    "raw_advantage",
    "center_advantage",
    "normalized_advantage",
    "value_target",
)


def verify_against_decomposition(
    arrays: dict[str, np.ndarray], reference: Path, study: AblationConfig
) -> dict[str, Any]:
    """Cross-check standard-GAE endpoints by their recorded decomposition arm indices."""
    summary = json.loads((reference / "summary.json").read_text(encoding="utf-8"))
    endpoints = {}
    for arm in summary["arms"]:
        if arm["label"] in ("r0", "energy_only"):
            if arm["label"] in endpoints:
                raise ValueError("reference contains duplicate endpoint arms")
            endpoints[arm["label"]] = arm["index"]
    if set(endpoints) != {"r0", "energy_only"}:
        raise ValueError("reference requires r0 and energy_only endpoints")
    with np.load(reference / "diagnostics.npz", allow_pickle=False) as historical:
        errors: dict[str, float] = {}
        for label, reference_index in endpoints.items():
            for key in _VALUE_KEYS:
                if key == "reward":
                    ours = arrays[f"arm_{label}_reward"]
                else:
                    ours = arrays[f"arm_{label}__standard_gae__{key}"]
                theirs = historical[f"arm_{reference_index}_{key}"]
                np.testing.assert_allclose(
                    ours,
                    theirs,
                    rtol=study.reference_match_tolerance.rtol,
                    atol=study.reference_match_tolerance.atol,
                    err_msg=f"{label}/{key}",
                )
                errors[f"{label}/{key}"] = float(np.max(np.abs(ours.astype(np.float64) - theirs)))
            for advantage_form in ADVANTAGE_FORMS:
                for group in GRADIENT_GROUPS:
                    ours = arrays[f"arm_{label}__standard_gae__gradient_{advantage_form}_{group}"]
                    theirs = historical[f"arm_{reference_index}_gradient_{advantage_form}_{group}"]
                    np.testing.assert_allclose(
                        ours,
                        theirs,
                        rtol=study.reference_match_tolerance.rtol,
                        atol=study.reference_match_tolerance.atol,
                        err_msg=f"{label}/gradient/{advantage_form}/{group}",
                    )
                    errors[f"{label}/gradient/{advantage_form}/{group}"] = float(
                        np.max(np.abs(ours.astype(np.float64) - theirs))
                    )
    return {
        "reference": str(reference),
        "max_abs_errors": errors,
        "tolerance": study.reference_match_tolerance.model_dump(),
    }


def run(
    source: Path, reference: Path, config_path: Path, output: Path, *, figures: bool = True
) -> dict:
    study = AblationConfig.model_validate(load_resolved_yaml_mapping(config_path))
    batch = load_fixed_batch(source)
    config = batch.resolved_config
    verify_reference(reference, source, batch)
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
        f"Running Task C4 critic/GAE ablation: {len(samples)} fixed transitions, "
        f"{len(calibrated_episodes)} episodes, 2 arms x {len(CREDIT_FORMS)} credit forms x "
        f"{len(ADVANTAGE_FORMS)} advantage forms, backward only.",
        flush=True,
    )
    summary, arrays = analyze_critic_gae_ablation(
        updater,
        calibrated_episodes,
        calibrated,
        study.quantiles,
        scenario_ids,
        study.gate,
    )
    runtime.verify_unchanged()
    reference_check = verify_against_decomposition(arrays, reference, study)
    write_npz(output / "diagnostics.npz", arrays)
    summary.update(
        {
            "status": "completed",
            "episode_count": len(episodes),
            "policy_unchanged": True,
            "source_batch": str(source.resolve()),
            "reference_decomposition": str(reference),
            "calibrated_reward": calibrated.model_dump(),
            "calibration_verification": calibration_checks,
            "initial_policy_hash": initial_hash,
            "standard_gae_reference_check": reference_check,
        }
    )
    write_json(output / "summary.json", summary)
    publish("critic-gae-ablation", output, output, figures=figures)
    return {
        "status": "completed",
        "output_dir": str(output),
        "sample_count": len(samples),
        "optimizer_steps": 0,
        "gate_c_endpoint_identifiable_under_standard_gae": summary["attribution"][
            "gate_c_endpoint_identifiable_under_standard_gae"
        ],
        "attribution": summary["attribution"]["attribution"],
    }
