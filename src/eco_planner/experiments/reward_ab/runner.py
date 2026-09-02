"""Launch the pre-registered matched PPO reward A/B without retries."""

from __future__ import annotations

import json
from pathlib import Path

from eco_planner.artifacts import write_json
from eco_planner.experiments.reward_ab.artifacts import write_study_manifest
from eco_planner.experiments.reward_ab.config import (
    PPORewardABConfig,
    RewardProfileSpec,
    load_ab_config,
)
from eco_planner.jobs import compose_job_config, run_training_job


def build_training_overrides(
    config: PPORewardABConfig,
    profile: RewardProfileSpec,
    training_seed: int,
    replay_id: int,
) -> tuple[str, ...]:
    """Build all explicit job overrides for one matched training run."""

    matched = config.matched_training
    return (
        f"components/reward={profile.reward_config}",
        f"runtime.seed={training_seed}",
        f"training.replay_id={replay_id}",
        f"training.update_count={matched.update_count}",
        f"training.transitions_per_environment={matched.transitions_per_environment}",
        f"ppo.scheduler_total_optimizer_steps={matched.scheduler_total_optimizer_steps}",
        f"name=ppo_reward_ab_{profile.id}",
    )


def run_ab(study_path: Path, output_root: Path) -> int:
    """Run every declared profile/seed/replay pair and report its comparison."""

    config = load_ab_config(study_path)
    output_root.mkdir(parents=True, exist_ok=False)
    write_study_manifest(study_path, output_root)
    runs: list[dict[str, object]] = []
    failed = False
    for training_seed in config.matched_training.training_seeds:
        for replay_id in config.matched_training.replay_ids:
            pair_dir = output_root / f"seed-{training_seed}-replay-{replay_id}"
            for profile in config.profiles:
                run_dir = pair_dir / profile.id
                raw = compose_job_config(
                    config.base_training_config,
                    build_training_overrides(config, profile, training_seed, replay_id),
                )
                summary = run_training_job(raw, run_dir)
                status = summary.status
                runs.append(
                    {
                        "training_seed": training_seed,
                        "replay_id": replay_id,
                        "profile": profile.id,
                        "reward_config": profile.reward_config,
                        "output_dir": str(run_dir),
                        "returncode": 0,
                        "status": status,
                    }
                )
                write_json(output_root / "launcher_summary.json", {"runs": runs})
                failed = failed or status != "completed"
    if failed:
        return 1
    from eco_planner.experiments.reward_ab.report import summarize_ab

    report = summarize_ab(output_root)
    write_json(output_root / "review_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["mechanical_status"] == "passed" else 1
