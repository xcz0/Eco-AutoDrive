"""Composition and protocol guards for the matched scalar-reward study."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from omegaconf import DictConfig

from eco_planner.evaluation import EvaluationJobConfig, parse_evaluation_config
from eco_planner.evaluation.artifacts.models import JobSummary
from eco_planner.experiments.protocol.config import (
    ComparisonProtocol,
    TrainedPolicyArmConfig,
)
from eco_planner.jobs import compose_job_config
from eco_planner.models import Ddim5SamplerConfig
from eco_planner.rl.config import TrainingJobConfig, parse_training_config

ArmName = str
CheckpointLabel = Literal["initial", "final"]


def compose_a0_evaluation_config(
    protocol: ComparisonProtocol,
) -> tuple[DictConfig, EvaluationJobConfig]:
    """Compose the frozen-planner held-out job and verify the protocol."""

    config = compose_job_config(protocol.evaluation.job)
    parsed = parse_evaluation_config(config)
    _require_evaluation_protocol(protocol, parsed)
    return config, parsed


def compose_arm_training_config(
    protocol: ComparisonProtocol,
    arm: ArmName,
    seed: int,
    overrides: Sequence[str] = (),
) -> tuple[DictConfig, TrainingJobConfig]:
    """Compose one trained arm's PPO job with its pinned matched-protocol values."""

    selected_arm = arm_config(protocol, arm)
    if selected_arm.reward_profile is None:
        raise ValueError("frozen planner cannot be trained")
    if any(override.startswith("runtime.seed") for override in overrides):
        raise ValueError("runtime.seed is selected by the seed argument, not overrides")
    config = compose_job_config(
        protocol.training.base_job,
        [
            *protocol.training.overrides,
            f"runtime.seed={seed}",
            "training.replay_id=0",
            f"components/reward={selected_arm.reward_profile}",
            *overrides,
        ],
    )
    config.tracking.tags = {
        **config.tracking.tags,
        "arm": arm.upper(),
        "protocol": protocol.study_name,
    }
    parsed = parse_training_config(config)
    _require_training_protocol(protocol, selected_arm, parsed, seed)
    return config, parsed


def compose_policy_evaluation_config(
    protocol: ComparisonProtocol,
    arm: ArmName,
    checkpoint_label: CheckpointLabel,
    checkpoint_path: Path,
    overrides: Sequence[str] = (),
) -> tuple[DictConfig, EvaluationJobConfig]:
    """Compose a trained arm's held-out policy-checkpoint evaluation job."""

    config = compose_job_config(
        protocol.evaluation.policy_job,
        [
            f"evaluation.policy_checkpoint.label={checkpoint_label}",
            f"evaluation.policy_checkpoint.path={checkpoint_path.as_posix()}",
            *overrides,
        ],
    )
    parsed = parse_evaluation_config(config)
    _require_evaluation_protocol(protocol, parsed)
    _, training = compose_arm_training_config(protocol, arm, protocol.training.seeds[0])
    if parsed.policy != training.policy:
        raise ValueError("policy evaluation must use the training policy architecture")
    if parsed.model != training.model:
        raise ValueError("policy evaluation must use the training planner model paths")
    if parsed.map_query_radius_m != training.map_query_radius_m:
        raise ValueError("policy evaluation must use the training map query radius")
    return config, parsed


def arm_config(protocol: ComparisonProtocol, arm: ArmName) -> TrainedPolicyArmConfig:
    return protocol.arms[arm]


def _require_evaluation_protocol(protocol: ComparisonProtocol, parsed: EvaluationJobConfig) -> None:
    if parsed.runtime.seed != protocol.evaluation.seed:
        raise ValueError(f"held-out evaluation requires runtime.seed={protocol.evaluation.seed}")
    if parsed.evaluation.history_warmup_steps != 0:
        raise ValueError("matched held-out evaluation requires zero history warmup steps")
    if parsed.evaluation.evaluated_horizon_steps != protocol.evaluation.horizon_steps:
        raise ValueError(
            f"held-out evaluation requires horizon={protocol.evaluation.horizon_steps} steps"
        )
    if not isinstance(parsed.sampler, Ddim5SamplerConfig):
        raise ValueError("matched held-out evaluation requires the ddim5 sampler")
    actual = {(item.map, item.seed) for item in parsed.scenarios}
    if actual != protocol.held_out_pairs():
        raise ValueError("held-out evaluation scenarios must match the protocol pool")
    maximum_seed = max(protocol.evaluation.map_seeds)
    num_scenarios = parsed.env.get("num_scenarios")
    if type(num_scenarios) is not int or num_scenarios <= maximum_seed:
        raise ValueError(f"held-out env.num_scenarios must exceed map seed {maximum_seed}")


def _require_training_protocol(
    protocol: ComparisonProtocol,
    arm_config: TrainedPolicyArmConfig,
    parsed: TrainingJobConfig,
    seed: int,
) -> None:
    if seed not in protocol.training.seeds:
        namespace = sorted(protocol.training.seeds)
        raise ValueError(f"training seed {seed} is outside the protocol namespace {namespace}")
    if parsed.reward.name != arm_config.reward_profile:
        raise ValueError(f"trained arm must use reward profile {arm_config.reward_profile}")
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


def validate_evaluation(protocol: ComparisonProtocol, summary: JobSummary) -> None:
    if {
        (e.scenario.map_sequence, e.scenario.seed) for e in summary.episodes
    } != protocol.held_out_pairs():
        raise ValueError("evaluation does not cover the held-out pool")
    if summary.runtime.seed != protocol.evaluation.seed:
        raise ValueError("evaluation seed differs from protocol")
    if (
        summary.workload.evaluated_horizon_steps != protocol.evaluation.horizon_steps
        or summary.sampler.name != protocol.evaluation.sampler
        or summary.sampler.ddim_stochasticity != 0
    ):
        raise ValueError("evaluation horizon/sampler differs from protocol")
