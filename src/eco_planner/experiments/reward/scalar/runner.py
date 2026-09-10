from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from eco_planner.analysis.runner import publish_scalar_run
from eco_planner.jobs import run_evaluation_job, run_training_job

from .composition import (
    CheckpointLabel,
    arm_config,
    compose_a0_evaluation_config,
    compose_arm_training_config,
    compose_policy_evaluation_config,
)
from .config import load_scalar_reward_protocol


def run_command(
    action: Literal["train", "evaluate"],
    protocol_path: Path,
    output_dir: Path,
    *,
    arm: Literal["a0", "a1", "a2"],
    training_seed: int | None = None,
    checkpoint_label: CheckpointLabel | None = None,
    checkpoint_path: Path | None = None,
    overrides: Sequence[str] = (),
    figures: bool = True,
) -> dict[str, object]:
    """Dispatch one matched-protocol CLI action."""

    protocol = load_scalar_reward_protocol(protocol_path)
    if action not in ("train", "evaluate"):
        raise ValueError(f"unsupported scalar operation: {action}")
    if action == "train" and (
        arm == "a0" or checkpoint_label is not None or checkpoint_path is not None
    ):
        raise ValueError("train requires a1/a2 and does not accept checkpoints")
    if action == "evaluate" and (training_seed is not None or overrides):
        raise ValueError("evaluate does not accept training arguments")
    if arm == "a0":
        if checkpoint_label is not None or checkpoint_path is not None:
            raise ValueError("a0 does not accept checkpoints")
        config, _ = compose_a0_evaluation_config(protocol)
        summary = run_evaluation_job(config, output_dir)
        publish_scalar_run(output_dir, training=False, figures=figures)
        return {
            "action": "evaluate-a0",
            "arm": "a0",
            "status": summary.status,
            "output_dir": str(output_dir),
        }
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
        "action": "evaluate-policy",
        "arm": arm,
        "checkpoint_label": checkpoint_label,
        "status": summary.status,
        "output_dir": str(output_dir),
    }
