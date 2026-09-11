"""Composition and protocol guards for the Task G positive-control training runs."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from omegaconf import DictConfig

from eco_planner.experiments.reward.scalar.config import ScalarRewardProtocolConfig
from eco_planner.jobs import compose_job_config
from eco_planner.models import Ddim5SamplerConfig
from eco_planner.rl.config import TrainingJobConfig, parse_training_config

ArmProfileName = str


def compose_positive_control_training_config(
    protocol: ScalarRewardProtocolConfig,
    reward_profile: ArmProfileName,
    seed: int,
    overrides: Sequence[str] = (),
) -> tuple[DictConfig, TrainingJobConfig]:
    """Compose one Task G arm's PPO job on the matched scalar protocol."""

    if any(override.startswith("runtime.seed") for override in overrides):
        raise ValueError("runtime.seed is selected by the seed argument, not overrides")
    if any(override.startswith("components/reward") for override in overrides):
        raise ValueError("the reward profile is selected by the arm, not overrides")
    config = compose_job_config(
        protocol.training.base_job,
        [
            f"runtime.seed={seed}",
            "training.replay_id=0",
            f"components/reward={reward_profile}",
            *overrides,
        ],
    )
    parsed = parse_training_config(config)
    _require_positive_control_protocol(protocol, reward_profile, parsed, seed)
    return config, parsed


def _require_positive_control_protocol(
    protocol: ScalarRewardProtocolConfig,
    reward_profile: str,
    parsed: TrainingJobConfig,
    seed: int,
) -> None:
    if seed not in protocol.training.seeds:
        namespace = sorted(protocol.training.seeds)
        raise ValueError(f"training seed {seed} is outside the protocol namespace {namespace}")
    if parsed.reward.name != reward_profile:
        raise ValueError(f"positive-control arm must use reward profile {reward_profile}")
    if parsed.runtime.seed != seed:
        raise ValueError(f"matched training requires runtime.seed={seed}")
    if parsed.training.replay_id != 0:
        raise ValueError("matched training pins training.replay_id=0")
    if not isinstance(parsed.sampler, Ddim5SamplerConfig):
        raise ValueError("matched training requires the ddim5 sampler")
    actual = {(item.map, item.seed) for item in parsed.scenarios}
    outside = actual - protocol.training_pairs()
    if outside:
        raise ValueError(f"training scenarios outside the protocol pool: {sorted(outside)}")
    maximum_seed = max(protocol.training.map_seeds)
    num_scenarios = parsed.env.get("num_scenarios")
    if type(num_scenarios) is not int or num_scenarios <= maximum_seed:
        raise ValueError(f"training env.num_scenarios must exceed map seed {maximum_seed}")


def compose_frozen_ppo_overrides(
    *,
    learning_rate: float,
    epochs: int,
    max_gradient_norm: float,
    update_count: int,
) -> list[str]:
    """Render the Task F frozen optimizer region as Hydra overrides."""

    return [
        f"ppo.learning_rate={learning_rate:.8g}",
        f"ppo.epochs={epochs}",
        f"ppo.max_gradient_norm={max_gradient_norm:g}",
        f"training.update_count={update_count}",
        f"ppo.scheduler_total_optimizer_steps={update_count * epochs}",
    ]


def run_label(arm_key: str, seed: int) -> str:
    """Name one training run directory after its arm and training seed."""

    return f"{arm_key}-seed-{seed}"


def heldout_dir(output_dir: Path, seed: int, arm_key: str) -> Path:
    """Locate one matched held-out evaluation directory for a seed and arm."""

    return output_dir / "heldout" / f"seed-{seed}" / arm_key
