"""Protocol validation for explicit scalar-reward comparisons."""

from pathlib import Path

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field

from eco_planner.analysis.evaluation import ScalarComparison, ScalarComparisonRun
from eco_planner.evaluation.artifacts import load_job_summary
from eco_planner.rl.artifacts import TrainingRunSummary

from .config import load_scalar_reward_protocol


class ComparisonRun(BaseModel):
    model_config = ConfigDict(extra="forbid")
    arm: str
    training_summary: Path
    checkpoint_label: str
    evaluation_dir: Path


class ComparisonConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    protocol: Path
    a0_evaluation_dir: Path
    runs: list[ComparisonRun] = Field(min_length=1)


def load_comparison(config_path: Path) -> ScalarComparison:
    config = ComparisonConfig.model_validate(OmegaConf.to_container(OmegaConf.load(config_path)))
    root = config_path.parent
    protocol = load_scalar_reward_protocol(root / config.protocol)
    baseline = load_job_summary(root / config.a0_evaluation_dir / "summary.json")
    expected = protocol.held_out_pairs()
    if {(e.scenario.map_sequence, e.scenario.seed) for e in baseline.episodes} != expected:
        raise ValueError("A0 does not cover the protocol held-out pool")
    if baseline.policy_checkpoint is not None or baseline.runtime.seed != protocol.evaluation.seed:
        raise ValueError("A0 checkpoint/seed differs from protocol")
    if (
        baseline.workload.evaluated_horizon_steps != protocol.evaluation.horizon_steps
        or baseline.sampler.name != protocol.evaluation.sampler
        or baseline.sampler.ddim_stochasticity != 0
    ):
        raise ValueError("A0 horizon/sampler differs from protocol")
    runs: list[ScalarComparisonRun] = []
    seen = set()
    for run in config.runs:
        if run.arm not in ("a1", "a2"):
            raise ValueError("comparison arm must be a1 or a2")
        training = TrainingRunSummary.model_validate_json(
            (root / run.training_summary).read_text(encoding="utf-8")
        )
        key = (run.arm, training.training_seed, run.checkpoint_label)
        if key in seen:
            raise ValueError(f"duplicate scalar-reward run: {key}")
        seen.add(key)
        if training.training_seed not in protocol.training.seeds:
            raise ValueError("training seed absent from protocol")
        if training.reward_profile != getattr(protocol.arms, run.arm).reward_profile:
            raise ValueError("training reward differs from declared arm")
        summary = load_job_summary(root / run.evaluation_dir / "summary.json")
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
        runs.append(ScalarComparisonRun(run.arm, run.checkpoint_label, training, summary))
    return ScalarComparison(
        baseline, tuple(runs), protocol.bootstrap, tuple(protocol.training.seeds)
    )
