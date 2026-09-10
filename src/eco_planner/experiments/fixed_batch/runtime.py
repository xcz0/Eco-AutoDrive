"""Restore the initial policy for diagnostics that only run actor backwards."""

from dataclasses import dataclass
from pathlib import Path

import torch

from eco_planner._repository import REPOSITORY_ROOT
from eco_planner.artifacts import collect_repository_metadata, write_json, write_tracked_diff
from eco_planner.experiments.fixed_batch.artifacts import FixedBatch
from eco_planner.rl.artifacts import policy_state_hash
from eco_planner.rl.optimization import PPOUpdater, load_exploration_policy_checkpoint
from eco_planner.rl.policy import ExplorationPolicy


@dataclass(frozen=True)
class DiagnosticRuntime:
    updater: PPOUpdater
    initial_policy_hash: str

    def verify_unchanged(self) -> None:
        if (
            policy_state_hash(self.updater.policy) != self.initial_policy_hash
            or self.updater.completed_optimizer_steps != 0
        ):
            raise RuntimeError("backward-only diagnostic changed the policy or performed an update")


def restore_runtime(source: Path, batch: FixedBatch) -> DiagnosticRuntime:
    config = batch.config
    torch.use_deterministic_algorithms(config.training.deterministic)
    torch.set_float32_matmul_precision("high")
    policy = ExplorationPolicy(config.policy)
    load_exploration_policy_checkpoint(source / "policy-initial.pt", policy)
    policy.to(torch.device(config.runtime.accelerator))
    initial_hash = policy_state_hash(policy)
    if initial_hash != batch.summary["initial_policy_hash"]:
        raise ValueError("loaded policy differs from source initial policy")
    return DiagnosticRuntime(PPOUpdater(policy, config.ppo), initial_hash)


def write_runtime_metadata(
    output: Path, source: Path, batch: FixedBatch, runtime: DiagnosticRuntime
) -> None:
    write_json(
        output / "runtime_metadata.json",
        {
            **collect_repository_metadata(REPOSITORY_ROOT),
            "source_batch": str(source.resolve()),
            "source_runtime": batch.runtime_metadata,
            "device": str(runtime.updater.device),
            "actor_backward_precision": "float32 (no rollout autocast)",
            "torch_version": str(torch.__version__),
            "initial_policy_hash": runtime.initial_policy_hash,
            "planner": "not instantiated; original fixed contexts and values reused",
        },
    )
    write_tracked_diff(output / "tracked_diff.patch", REPOSITORY_ROOT)
