"""Protocol validation for explicit scalar-reward comparisons."""

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field

from eco_planner.analysis.evaluation import PolicyComparison, PolicyComparisonRun
from eco_planner.configuration import load_resolved_yaml_mapping
from eco_planner.evaluation.artifacts import load_job_summary
from eco_planner.experiments.protocol.composition import validate_evaluation
from eco_planner.experiments.protocol.config import ComparisonProtocol, load_protocol
from eco_planner.rl.artifacts import TrainingRunSummary


class ComparisonRun(BaseModel):
    model_config = ConfigDict(extra="forbid")
    arm: str
    training_summary: Path
    checkpoint_label: str
    evaluation_dir: Path


class ComparisonConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    protocol: Path
    baseline_evaluation_dir: Path | None
    runs: list[ComparisonRun] = Field(min_length=1)


def load_comparison(config_path: Path) -> PolicyComparison:
    config = ComparisonConfig.model_validate(OmegaConf.to_container(OmegaConf.load(config_path)))
    root = config_path.parent
    protocol = load_protocol(root / config.protocol)
    baseline = None
    source_directories = {root.resolve(), (root / config.protocol).resolve().parent}
    if config.baseline_evaluation_dir is not None:
        if protocol.frozen_arm is None:
            raise ValueError("protocol has no frozen baseline arm")
        baseline = load_job_summary(root / config.baseline_evaluation_dir / "summary.json")
        source_directories.add((root / config.baseline_evaluation_dir).resolve())
        validate_evaluation(protocol, baseline)
        if baseline.policy_checkpoint is not None:
            raise ValueError("frozen baseline must not contain a policy checkpoint")
    elif protocol.frozen_arm is not None:
        raise ValueError("comparison requires the declared frozen baseline")
    runs: list[PolicyComparisonRun] = []
    seen = set()
    initial_by_seed: dict[int, TrainingRunSummary] = {}
    conditions_by_seed: dict[int, dict[str, Any]] = {}
    for run in config.runs:
        source_directories.update(
            [
                (root / run.training_summary).resolve().parent,
                (root / run.evaluation_dir).resolve(),
            ]
        )
        if run.arm not in protocol.arms or protocol.arms[run.arm].reward_profile is None:
            raise ValueError("comparison arm must declare a trained reward profile")
        training_path = root / run.training_summary
        training = TrainingRunSummary.model_validate_json(training_path.read_text(encoding="utf-8"))
        key = (run.arm, training.training_seed, run.checkpoint_label)
        if key in seen:
            raise ValueError(f"duplicate scalar-reward run: {key}")
        seen.add(key)
        if training.training_seed not in protocol.training.seeds:
            raise ValueError("training seed absent from protocol")
        reference = initial_by_seed.setdefault(training.training_seed, training)
        if any(
            getattr(training, key) != getattr(reference, key)
            for key in (
                "initial_policy_hash",
                "noise_seeds",
                "policy_action_seeds",
                "frozen_planner_hash_before",
                "probe_before",
            )
        ):
            raise ValueError("trained arms do not share matched initial policy, probes and seeds")
        if training.reward_profile != protocol.arms[run.arm].reward_profile:
            raise ValueError("training reward differs from declared arm")
        conditions = _load_matched_training_conditions(
            training_path.with_name("resolved_config.yaml"), protocol, run.arm, training
        )
        reference_conditions = conditions_by_seed.setdefault(training.training_seed, conditions)
        if conditions != reference_conditions:
            raise ValueError("trained arms do not share matched resolved training conditions")
        summary = load_job_summary(root / run.evaluation_dir / "summary.json")
        validate_evaluation(protocol, summary)
        checkpoint = summary.policy_checkpoint
        if checkpoint is None or checkpoint.label != run.checkpoint_label:
            raise ValueError("evaluation checkpoint label differs from comparison config")
        expected_hash = (
            training.initial_policy_hash
            if run.checkpoint_label == "initial"
            else (training.final_policy_hash if run.checkpoint_label == "final" else None)
        )
        if expected_hash is None or checkpoint.policy_hash != expected_hash:
            raise ValueError(
                "evaluation checkpoint is not the declared training initial/final state"
            )
        runs.append(PolicyComparisonRun(run.arm, run.checkpoint_label, training, summary))
    return PolicyComparison(
        baseline,
        tuple(runs),
        protocol.bootstrap,
        tuple(protocol.training.seeds),
        tuple((pair[0], pair[1]) for pair in protocol.contrasts),
        protocol.frozen_arm,
        tuple(sorted(source_directories)),
    )


def _load_matched_training_conditions(
    path: Path,
    protocol: ComparisonProtocol,
    arm: str,
    summary: TrainingRunSummary,
) -> dict[str, Any]:
    resolved = load_resolved_yaml_mapping(path)
    try:
        scenarios = {(item["map"], item["seed"]) for item in resolved["scenarios"]}
        runtime_seed = resolved["runtime"]["seed"]
        replay_id = resolved["training"]["replay_id"]
        sampler = resolved["sampler"]["name"]
        reward_profile = resolved["reward"]["name"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"training resolved config is missing matched fields: {path}") from error
    if scenarios != protocol.training_pairs():
        raise ValueError("training resolved scenarios must match the protocol pool")
    if runtime_seed != summary.training_seed or replay_id != summary.replay_id:
        raise ValueError("training resolved seed/replay differs from the typed summary")
    if sampler != "ddim5":
        raise ValueError("matched training requires the ddim5 sampler")
    if reward_profile != protocol.arms[arm].reward_profile:
        raise ValueError("training resolved reward differs from the declared arm")
    conditions = dict(resolved)
    conditions.pop("reward", None)
    conditions.pop("tracking", None)
    return conditions
