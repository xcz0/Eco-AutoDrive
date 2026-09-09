"""Restore the fixed E-035 batch and run Issue 94 Task C4 critic/GAE common-term ablation."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

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
from eco_planner.experiments.critic_gae_ablation import (
    ADVANTAGE_FORMS,
    CREDIT_FORMS,
    AblationConfig,
    analyze_critic_gae_ablation,
    render_ablation_report,
)
from eco_planner.experiments.objective_decomposition import _GRADIENT_GROUPS
from eco_planner.experiments.objective_decomposition_runner import verify_expected_calibration
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

_REFERENCE_ARM_INDICES = {"r0": 0, "energy_only": 4}
_VALUE_KEYS = (
    "reward",
    "raw_advantage",
    "center_advantage",
    "normalized_advantage",
    "value_target",
)


def verify_against_e035(
    arrays: dict[str, np.ndarray], reference: Path, study: AblationConfig
) -> dict[str, Any]:
    """Cross-check the standard-GAE arms against the recorded E-035 decomposition arrays."""
    with np.load(reference / "diagnostics.npz", allow_pickle=False) as historical:
        errors: dict[str, float] = {}
        for label, reference_index in _REFERENCE_ARM_INDICES.items():
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
                errors[f"{label}/{key}"] = float(
                    np.max(np.abs(ours.astype(np.float64) - theirs))
                )
            for advantage_form in ADVANTAGE_FORMS:
                for group in _GRADIENT_GROUPS:
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


def run(source: Path, reference: Path, config_path: Path, output: Path) -> dict:
    study = AblationConfig.model_validate(load_resolved_yaml_mapping(config_path))
    config = load_resolved_yaml_mapping(source / "resolved_config.yaml")
    source_summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    if not source_summary.get("arms") or source_summary["arms"][0]["lambda"] != 0:
        raise ValueError("source must be a fixed update-0 identifiability batch starting at R0")
    reference_summary = json.loads((reference / "summary.json").read_text(encoding="utf-8"))
    reference_labels = [arm["label"] for arm in reference_summary["arms"]]
    if reference_labels[0] != "r0" or reference_labels[-1] != "energy_only":
        raise ValueError(
            "reference must be the E-035 decomposition starting at R0 and ending at Energy-only"
        )
    if reference_summary.get("source_batch") != str(source):
        raise ValueError(
            f"reference decomposition used a different source batch: "
            f"{reference_summary.get('source_batch')!r} vs {str(source)!r}"
        )
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
        "reference_decomposition": str(reference),
        "device": str(updater.device),
        "actor_backward_precision": "float32, matching Tasks A/B/C (no rollout autocast)",
        "torch_version": str(torch.__version__),
        "initial_policy_hash": initial_hash,
        "planner": "not instantiated; original fixed contexts and values reused",
    }
    write_json(output / "runtime_metadata.json", metadata)
    write_tracked_diff(output / "tracked_diff.patch", REPOSITORY_ROOT)
    for file in (
        Path(__file__),
        Path(__file__).with_name("critic_gae_ablation.py"),
        REPOSITORY_ROOT / "src/eco_planner/experiments/lambda_identifiability/diagnostics.py",
        REPOSITORY_ROOT / "src/eco_planner/experiments/reward_calibration.py",
        REPOSITORY_ROOT / "src/eco_planner/experiments/reward_calibration_runner.py",
        REPOSITORY_ROOT / "src/eco_planner/experiments/objective_decomposition.py",
        REPOSITORY_ROOT / "src/eco_planner/experiments/objective_decomposition_runner.py",
        REPOSITORY_ROOT / "scripts/experiments/critic_gae_ablation.py",
    ):
        target = output / "source" / file.relative_to(REPOSITORY_ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(file, target)
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
    if policy_state_hash(policy) != initial_hash or updater.completed_optimizer_steps != 0:
        raise RuntimeError("backward-only ablation changed the policy or performed an update")
    reference_check = verify_against_e035(arrays, reference, study)
    write_npz(output / "diagnostics.npz", arrays)
    summary.update(
        {
            "status": "completed",
            "episode_count": len(episodes),
            "policy_unchanged": True,
            "source_batch": str(source),
            "reference_decomposition": str(reference),
            "calibrated_reward": calibrated.model_dump(),
            "calibration_verification": calibration_checks,
            "initial_policy_hash": initial_hash,
            "standard_gae_reference_check": reference_check,
        }
    )
    write_json(output / "summary.json", summary)
    (output / "report.md").write_text(render_ablation_report(summary), encoding="utf-8")
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
