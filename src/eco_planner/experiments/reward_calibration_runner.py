"""Restore the E-033 batch and run both reward calibrations without a simulator."""

from __future__ import annotations

import json
import shutil
from itertools import groupby
from pathlib import Path
from typing import cast

import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from eco_planner._repository import REPOSITORY_ROOT
from eco_planner.analysis.reporting.fixed import render_report
from eco_planner.analysis.runner import publish
from eco_planner.artifacts import (
    collect_repository_metadata,
    write_json,
    write_npz,
    write_tracked_diff,
)
from eco_planner.configuration import load_resolved_yaml_mapping
from eco_planner.experiments.lambda_identifiability.diagnostics import analyze
from eco_planner.experiments.reward_calibration import (
    CalibrationConfig,
    calibrate,
    dynamic_range_audit,
    raw_arrays,
    rescore,
)
from eco_planner.rl.artifacts import policy_state_hash
from eco_planner.rl.optimization import PPOUpdater, load_exploration_policy_checkpoint
from eco_planner.rl.optimization.config import PPOConfig
from eco_planner.rl.policy import ExplorationPolicy, ExplorationPolicyConfig
from eco_planner.rl.reward.config import PlannerRFTNoEnergyRewardConfig
from eco_planner.rl.rollout.contracts import (
    RewardProfileName,
    RolloutEpisode,
    TailKind,
    rollout_audit_keys,
)


def load_batch(source: Path) -> tuple[list[RolloutEpisode], list[dict]]:
    samples = json.loads((source / "sample_index.json").read_text(encoding="utf-8"))["samples"]
    payload = torch.load(source / "training-batch.pt", map_location="cpu", weights_only=True)
    groups = [
        (key, list(rows))
        for key, rows in groupby(samples, key=lambda s: (s["scenario_index"], s["episode_index"]))
    ]
    if len({key for key, _ in groups}) != len(groups):
        raise ValueError("sample index repeats an episode out of order")
    episodes = []
    for ((slot, number), rows), training_data in zip(groups, payload, strict=True):
        path = source / "updates/update-000" / f"slot-{slot}-episode-{number}.npz"
        with np.load(path, allow_pickle=False) as z:
            profile = cast(RewardProfileName, str(z["reward_profile"].item()))
            audit = TensorDict(
                {k: torch.from_numpy(z[k].copy()) for k in rollout_audit_keys(profile)},
                batch_size=[len(rows)],
            )
            episode = RolloutEpisode(
                training=TensorDict(training_data, batch_size=[len(rows)]),
                audit=audit,
                tail_kind=cast(TailKind, str(z["tail_kind"].item())),
                tail_bootstrap_value=torch.from_numpy(z["tail_bootstrap_value"].copy()),
                reward_profile=profile,
            )
        for key in ("planning_cycle_index", "map_seed"):
            np.testing.assert_array_equal(audit[key].numpy().reshape(-1), [s[key] for s in rows])
        for key in episode.training.keys():
            if key != "next":
                expected = audit[key]
                if key == "old_joint_guidance_log_prob":
                    expected = expected.squeeze(-1)
                torch.testing.assert_close(episode.training[key], expected, rtol=0, atol=0)
        for key, audit_key in (
            ("reward", "reward_total"),
            ("terminated", "terminated"),
            ("truncated", "truncated"),
        ):
            torch.testing.assert_close(
                episode.training["next", key], audit[audit_key], rtol=0, atol=0
            )
        episodes.append(episode)
    return episodes, samples


def verify_original_components(
    episodes: list[RolloutEpisode], base: PlannerRFTNoEnergyRewardConfig
) -> None:
    for episode in episodes:
        rebuilt = rescore(episode, base)
        for key in (
            "reward_component_progress",
            "reward_component_comfort",
            "reward_base_total",
            "reward_total",
        ):
            torch.testing.assert_close(rebuilt.audit[key], episode.audit[key], rtol=1e-6, atol=1e-7)


def run(source: Path, config_path: Path, output: Path, *, figures: bool = True) -> dict:
    study = CalibrationConfig.model_validate(load_resolved_yaml_mapping(config_path))
    config = load_resolved_yaml_mapping(source / "resolved_config.yaml")
    source_summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    if study.lambdas != [a["lambda"] for a in source_summary["arms"]]:
        raise ValueError("lambda axes must match the source diagnostic")
    base = PlannerRFTNoEnergyRewardConfig.model_validate(config["reward"])
    episodes, samples = load_batch(source)
    verify_original_components(episodes, base)
    raw = raw_arrays(episodes)
    calibrated = calibrate(raw, base, study)
    scenario_ids = np.asarray([s["scenario_index"] for s in samples], dtype=np.int64)
    cycles = np.asarray([s["planning_cycle_index"] for s in samples], dtype=np.int64)
    audit, audit_arrays = dynamic_range_audit(raw, base, calibrated, study, scenario_ids, cycles)
    torch.use_deterministic_algorithms(config["training"]["deterministic"])
    torch.set_float32_matmul_precision("high")
    policy = ExplorationPolicy(ExplorationPolicyConfig.model_validate(config["policy"]))
    load_exploration_policy_checkpoint(source / "policy-initial.pt", policy)
    policy.to(torch.device(config["runtime"]["accelerator"]))
    initial_hash = policy_state_hash(policy)
    if initial_hash != source_summary["initial_policy_hash"]:
        raise ValueError("loaded policy differs from source initial policy")
    updater = PPOUpdater(policy, PPOConfig.model_validate(config["ppo"]))
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "sample_index.json", {"samples": samples})
    write_json(output / "audit.json", audit)
    write_npz(output / "audit.npz", audit_arrays)
    OmegaConf.save(OmegaConf.create(study.model_dump()), output / "calibration_config.yaml")
    metadata = {
        **collect_repository_metadata(REPOSITORY_ROOT),
        "source_batch": str(source),
        "source_runtime": json.loads(
            (source / "runtime_metadata.json").read_text(encoding="utf-8")
        ),
        "device": str(updater.device),
        "actor_backward_precision": "float32, matching Task A (no rollout autocast)",
        "torch_version": str(torch.__version__),
        "initial_policy_hash": initial_hash,
        "planner": "not instantiated; original fixed contexts and values reused",
    }
    write_json(output / "runtime_metadata.json", metadata)
    write_tracked_diff(output / "tracked_diff.patch", REPOSITORY_ROOT)
    for file in (
        Path(__file__),
        Path(__file__).with_name("reward_calibration.py"),
        REPOSITORY_ROOT / "src/eco_planner/experiments/lambda_identifiability/diagnostics.py",
        REPOSITORY_ROOT / "src/eco_planner/experiments/lambda_identifiability/runner.py",
        REPOSITORY_ROOT / "src/eco_planner/rl/reward/components/progress.py",
        REPOSITORY_ROOT / "src/eco_planner/rl/reward/components/comfort.py",
        REPOSITORY_ROOT / "scripts/experiments/reward_calibration.py",
    ):
        target = output / "source" / file.relative_to(REPOSITORY_ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(file, target)
    results, original_errors = {}, {}
    for label, profile in (("original", base), ("calibrated", calibrated)):
        print(f"Running {label}: {len(samples)} fixed transitions, backward only.", flush=True)
        matched = episodes if label == "original" else [rescore(e, profile) for e in episodes]
        summary, arrays = analyze(
            updater, matched, profile, study.lambdas, study.quantiles, scenario_ids
        )
        if label == "original":
            with np.load(source / "diagnostics.npz", allow_pickle=False) as historical:
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
    if policy_state_hash(policy) != initial_hash or updater.completed_optimizer_steps != 0:
        raise RuntimeError("backward-only calibration changed the policy or performed an update")
    summary = {
        "status": "completed",
        "sample_count": len(samples),
        "episode_count": len(episodes),
        "optimizer_steps": updater.completed_optimizer_steps,
        "policy_unchanged": True,
        "source_batch": str(source),
        "calibrated_reward": calibrated.model_dump(),
        "original_replay_max_abs_errors": original_errors,
        "decision": "continuous evidence only; user decides whether to enter Task C",
        "comparisons": {label: result["pairs"] for label, result in results.items()},
    }
    write_json(output / "summary.json", summary)
    publish("reward-calibration", output, output, figures=figures)
    return {"status": "completed", "output_dir": str(output), "sample_count": len(samples)}
