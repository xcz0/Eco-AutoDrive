"""Launch the pre-registered matched PPO reward A/B without retrying failed runs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Literal

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, model_validator

from eco_planner.artifacts import write_json
from eco_planner.configuration import load_resolved_yaml_mapping
from scripts._paths import CONFIG_ROOT, REPOSITORY_ROOT

DEFAULT_STUDY = CONFIG_ROOT / "studies" / "reward" / "ppo_ab.yaml"


class _StrictModel(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)


class RewardProfileSpec(_StrictModel):
    id: Literal["builtin", "energy"]
    reward_config: Literal["metadrive_builtin_v1", "plannerrft_energy_v1"]


class ScenarioSpec(_StrictModel):
    name: str
    map: str
    seed: StrictInt = Field(ge=0)


class MatchedTrainingSpec(_StrictModel):
    update_count: StrictInt = Field(ge=2)
    transitions_per_environment: StrictInt = Field(gt=0)
    scheduler_total_optimizer_steps: StrictInt = Field(gt=0)
    training_seeds: list[StrictInt] = Field(min_length=1)
    replay_ids: list[StrictInt] = Field(min_length=1)
    scenarios: list[ScenarioSpec] = Field(min_length=1)


class ReviewThresholds(_StrictModel):
    longitudinal_action_mean_deadband: StrictFloat = Field(ge=0.0)
    energy_intensity_deadband_fraction: StrictFloat = Field(ge=0.0)
    maximum_progress_drop_fraction: StrictFloat = Field(ge=0.0, le=1.0)
    maximum_mean_speed_drop_fraction: StrictFloat = Field(ge=0.0, le=1.0)
    maximum_collision_count_increase: StrictInt = Field(ge=0)
    maximum_out_of_road_count_increase: StrictInt = Field(ge=0)


class PPORewardABConfig(_StrictModel):
    version: Literal[2]
    base_training_config: str
    profiles: list[RewardProfileSpec]
    matched_training: MatchedTrainingSpec
    review_thresholds: ReviewThresholds

    @model_validator(mode="after")
    def validate_pair(self) -> PPORewardABConfig:
        if [(item.id, item.reward_config) for item in self.profiles] != [
            ("builtin", "metadrive_builtin_v1"),
            ("energy", "plannerrft_energy_v1"),
        ]:
            raise ValueError("PPO reward A/B profiles must be builtin then energy")
        if len(set(self.matched_training.training_seeds)) != len(
            self.matched_training.training_seeds
        ):
            raise ValueError("PPO reward A/B training seeds must be unique")
        if len(set(self.matched_training.replay_ids)) != len(self.matched_training.replay_ids):
            raise ValueError("PPO reward A/B replay ids must be unique")
        return self


def load_ab_config(path: Path) -> PPORewardABConfig:
    return PPORewardABConfig.model_validate(load_resolved_yaml_mapping(path))


def build_training_command(
    config: PPORewardABConfig,
    profile: RewardProfileSpec,
    training_seed: int,
    replay_id: int,
    run_dir: Path,
) -> list[str]:
    matched = config.matched_training
    return [
        sys.executable,
        "-m",
        "scripts.train",
        f"--config-name={config.base_training_config}",
        f"components/reward={profile.reward_config}",
        f"runtime.seed={training_seed}",
        f"training.replay_id={replay_id}",
        f"training.update_count={matched.update_count}",
        f"training.transitions_per_environment={matched.transitions_per_environment}",
        f"rl.scheduler_total_optimizer_steps={matched.scheduler_total_optimizer_steps}",
        f"name=ppo_reward_ab_{profile.id}",
        f"hydra.run.dir={run_dir.as_posix()}",
    ]


def run_ab(study_path: Path, output_root: Path) -> int:
    config = load_ab_config(study_path)
    output_root.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(OmegaConf.load(study_path), output_root / "study_manifest.yaml", resolve=True)
    runs: list[dict[str, object]] = []
    failed = False
    for training_seed in config.matched_training.training_seeds:
        for replay_id in config.matched_training.replay_ids:
            pair_dir = output_root / f"seed-{training_seed}-replay-{replay_id}"
            for profile in config.profiles:
                run_dir = pair_dir / profile.id
                command = build_training_command(config, profile, training_seed, replay_id, run_dir)
                completed = subprocess.run(command, cwd=REPOSITORY_ROOT, check=False)
                summary_path = run_dir / "summary.json"
                status = "launcher_failure"
                if summary_path.is_file():
                    payload = json.loads(summary_path.read_text(encoding="utf-8"))
                    status = str(payload.get("status", "invalid_summary"))
                record = {
                    "training_seed": training_seed,
                    "replay_id": replay_id,
                    "profile": profile.id,
                    "reward_config": profile.reward_config,
                    "output_dir": str(run_dir),
                    "returncode": completed.returncode,
                    "status": status,
                }
                runs.append(record)
                write_json(output_root / "launcher_summary.json", {"runs": runs})
                failed = failed or completed.returncode != 0 or status != "completed"
    if failed:
        return 1
    from scripts.analysis.ppo_reward_ab import summarize_ab

    report = summarize_ab(output_root)
    write_json(output_root / "review_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["mechanical_status"] == "passed" else 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run_ab(args.study.resolve(), args.output_root.resolve()))


if __name__ == "__main__":
    main()
