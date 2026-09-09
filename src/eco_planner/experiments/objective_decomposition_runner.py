"""Restore the fixed E-033/E-034 batch and run Issue 94 Task C objective decomposition."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from omegaconf import OmegaConf

from eco_planner._repository import REPOSITORY_ROOT
from eco_planner.artifacts import (
    collect_repository_metadata,
    write_json,
    write_npz,
    write_tracked_diff,
)
from eco_planner.configuration import load_resolved_yaml_mapping
from eco_planner.experiments.objective_decomposition import (
    ADVANTAGE_FORMS,
    CalibrationMatchTolerance,
    DecompositionConfig,
    ExpectedCalibration,
    analyze_decomposition,
    render_decomposition_report,
)
from eco_planner.experiments.reward_calibration import calibrate, raw_arrays, rescore
from eco_planner.experiments.reward_calibration_runner import (
    load_batch,
    verify_original_components,
)
from eco_planner.rl.artifacts import policy_state_hash
from eco_planner.rl.optimization import PPOUpdater, load_exploration_policy_checkpoint
from eco_planner.rl.optimization.config import PPOConfig
from eco_planner.rl.policy import ExplorationPolicy, ExplorationPolicyConfig
from eco_planner.rl.reward.config import PlannerRFTNoEnergyRewardConfig

_EXPECTED_CALIBRATION_FIELDS = (
    ("progress.full_score_delta_m", "full_score_delta_m"),
    (
        "comfort.longitudinal_acceleration_limit_mps2",
        "longitudinal_acceleration_limit_mps2",
    ),
    ("comfort.lateral_acceleration_limit_mps2", "lateral_acceleration_limit_mps2"),
    ("comfort.jerk_limit_mps3", "jerk_limit_mps3"),
    ("comfort.yaw_rate_limit_radps", "yaw_rate_limit_radps"),
)


class CalibrationGuardSource(Protocol):
    """Study configs carrying the E-034 frozen calibration provenance fields."""

    expected_calibration: ExpectedCalibration
    calibration_match_tolerance: CalibrationMatchTolerance


def verify_expected_calibration(
    calibrated: PlannerRFTNoEnergyRewardConfig, study: CalibrationGuardSource
) -> dict[str, Any]:
    """Guard the source-batch provenance against the E-034 frozen calibration values.

    The re-collected rollout matches the E-033 protocol and hashes but not bitwise, so
    raw kinematic medians drift by up to ~0.5%; the tolerance catches a wrong source
    batch or protocol drift, not float noise.
    """
    checks: dict[str, dict[str, float]] = {}
    for field, attribute in _EXPECTED_CALIBRATION_FIELDS:
        section, _, name = field.partition(".")
        actual = float(getattr(getattr(calibrated, section), name))
        expected = float(getattr(study.expected_calibration, attribute))
        if not math.isclose(
            actual,
            expected,
            rel_tol=study.calibration_match_tolerance.rtol,
            abs_tol=study.calibration_match_tolerance.atol,
        ):
            raise ValueError(
                f"source batch calibration is inconsistent with the E-034 frozen value for "
                f"{field}: {actual!r} vs {expected!r}"
            )
        checks[field] = {"actual": actual, "expected": expected}
    return dict(checks)


def run(source: Path, config_path: Path, output: Path) -> dict:
    study = DecompositionConfig.model_validate(load_resolved_yaml_mapping(config_path))
    config = load_resolved_yaml_mapping(source / "resolved_config.yaml")
    source_summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    if not source_summary.get("arms") or source_summary["arms"][0]["lambda"] != 0:
        raise ValueError("source must be a fixed update-0 identifiability batch starting at R0")
    base = PlannerRFTNoEnergyRewardConfig.model_validate(config["reward"])
    episodes, samples = load_batch(source)
    verify_original_components(episodes, base)
    calibrated = calibrate(raw_arrays(episodes), base, study)
    calibration_checks = verify_expected_calibration(calibrated, study)
    scenario_ids = np.asarray([s["scenario_index"] for s in samples], dtype=np.int64)
    torch.use_deterministic_algorithms(config["training"]["deterministic"])
    torch.set_float32_matmul_precision("high")
    policy = ExplorationPolicy(ExplorationPolicyConfig.model_validate(config["policy"]))
    load_exploration_policy_checkpoint(source / "policy-initial.pt", policy)
    policy.to(torch.device(config["runtime"]["accelerator"]))
    initial_hash = policy_state_hash(policy)
    if initial_hash != source_summary["initial_policy_hash"]:
        raise ValueError("loaded policy differs from source initial policy")
    updater = PPOUpdater(policy, PPOConfig.model_validate(config["ppo"]))
    calibrated_episodes = [rescore(episode, calibrated) for episode in episodes]
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "sample_index.json", {"samples": samples})
    write_json(output / "calibration_verification.json", calibration_checks)
    OmegaConf.save(OmegaConf.create(study.model_dump()), output / "diagnostic_config.yaml")
    resolved = {**config, "reward": calibrated.model_dump()}
    OmegaConf.save(OmegaConf.create(resolved), output / "resolved_config.yaml")
    metadata = {
        **collect_repository_metadata(REPOSITORY_ROOT),
        "source_batch": str(source),
        "source_runtime": json.loads(
            (source / "runtime_metadata.json").read_text(encoding="utf-8")
        ),
        "device": str(updater.device),
        "actor_backward_precision": "float32, matching Tasks A/B (no rollout autocast)",
        "torch_version": str(torch.__version__),
        "initial_policy_hash": initial_hash,
        "planner": "not instantiated; original fixed contexts and values reused",
    }
    write_json(output / "runtime_metadata.json", metadata)
    write_tracked_diff(output / "tracked_diff.patch", REPOSITORY_ROOT)
    for file in (
        Path(__file__),
        Path(__file__).with_name("objective_decomposition.py"),
        REPOSITORY_ROOT / "src/eco_planner/experiments/lambda_identifiability/diagnostics.py",
        REPOSITORY_ROOT / "src/eco_planner/experiments/reward_calibration.py",
        REPOSITORY_ROOT / "src/eco_planner/experiments/reward_calibration_runner.py",
        REPOSITORY_ROOT / "scripts/experiments/objective_decomposition.py",
    ):
        target = output / "source" / file.relative_to(REPOSITORY_ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(file, target)
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
    if policy_state_hash(policy) != initial_hash or updater.completed_optimizer_steps != 0:
        raise RuntimeError("backward-only decomposition changed the policy or performed an update")
    write_npz(output / "diagnostics.npz", arrays)
    summary.update(
        {
            "status": "completed",
            "episode_count": len(episodes),
            "policy_unchanged": True,
            "source_batch": str(source),
            "calibrated_reward": calibrated.model_dump(),
            "calibration_verification": calibration_checks,
            "initial_policy_hash": initial_hash,
            "decision": "Gate C evaluated with Issue #94 engineering thresholds; "
            "Task D entry is decided from the recorded gate verdict",
        }
    )
    write_json(output / "summary.json", summary)
    (output / "report.md").write_text(render_decomposition_report(summary), encoding="utf-8")
    return {
        "status": "completed",
        "output_dir": str(output),
        "sample_count": len(samples),
        "optimizer_steps": 0,
        "gate_c_passed": summary["gate"]["gate_c_passed"],
        "attribution": summary["gate"]["attribution"],
    }
