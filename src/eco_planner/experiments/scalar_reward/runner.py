from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from eco_planner.analysis.runner import publish_scalar_run
from eco_planner.experiments.scalar_reward.composition import (
    ArmName,
    CheckpointLabel,
    arm_config,
    compose_a0_evaluation_config,
    compose_arm_training_config,
    compose_policy_evaluation_config,
)
from eco_planner.experiments.scalar_reward.config import (
    load_scalar_reward_protocol,
)
from eco_planner.jobs import run_evaluation_job, run_training_job


def run_command(
    action: Literal["evaluate-a0", "train", "evaluate-policy"],
    protocol_path: Path,
    output_dir: Path,
    *,
    arm: ArmName | None = None,
    training_seed: int | None = None,
    checkpoint_label: CheckpointLabel | None = None,
    checkpoint_path: Path | None = None,
    overrides: Sequence[str] = (),
    figures: bool = True,
) -> dict[str, object]:
    """Dispatch one matched-protocol CLI action."""

    protocol = load_scalar_reward_protocol(protocol_path)
    if action == "evaluate-a0":
        config, _ = compose_a0_evaluation_config(protocol)
        summary = run_evaluation_job(config, output_dir)
        publish_scalar_run(output_dir, training=False, figures=figures)
        return {
            "action": action,
            "arm": "a0",
            "status": summary.status,
            "output_dir": str(output_dir),
        }
    if arm is None:
        raise ValueError(f"{action} requires --arm a1 or a2")
    if action == "train":
        if training_seed is None:
            raise ValueError("train requires --training-seed")
        config, _ = compose_arm_training_config(protocol, arm, training_seed, overrides)
        summary = run_training_job(config, output_dir)
        publish_scalar_run(output_dir, training=True, figures=figures)
        return {
            "action": action,
            "arm": arm,
            "reward_profile": arm_config(protocol, arm).reward_profile,
            "training_seed": training_seed,
            "status": summary.status,
            "output_dir": str(output_dir),
        }
    if checkpoint_label is None or checkpoint_path is None:
        raise ValueError("evaluate-policy requires --checkpoint and --checkpoint-path")
    config, _ = compose_policy_evaluation_config(protocol, arm, checkpoint_label, checkpoint_path)
    summary = run_evaluation_job(config, output_dir)
    publish_scalar_run(output_dir, training=False, figures=figures)
    return {
        "action": action,
        "arm": arm,
        "checkpoint_label": checkpoint_label,
        "status": summary.status,
        "output_dir": str(output_dir),
    }
