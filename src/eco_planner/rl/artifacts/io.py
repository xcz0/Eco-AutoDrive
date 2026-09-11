"""RL-specific summaries and TensorDict-to-NPZ artifact adapters."""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
from hydra.utils import to_absolute_path
from tensordict import TensorDictBase

from eco_planner.artifacts import collect_repository_metadata, write_json, write_npz
from eco_planner.rl.artifacts.schema import rollout_artifact_fields
from eco_planner.rl.policy import ExplorationPolicy
from eco_planner.rl.rollout.contracts import (
    RolloutEpisode,
    rollout_audit_keys,
)
from eco_planner.runtime.resources import ResourceProfileConfig

if TYPE_CHECKING:
    from eco_planner.rl.rollout.runtime import FabricRolloutRuntime


def policy_state_hash(policy: ExplorationPolicy) -> str:
    """Hash one policy state dict in stable name order."""

    digest = hashlib.sha256()
    for name, value in sorted(policy.state_dict().items()):
        host = value.detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(host.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def write_rollout_episode(path: Path, episode: RolloutEpisode) -> None:
    """Persist the complete audit trajectory through the stable NumPy artifact boundary."""

    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = _trajectory_arrays(episode)
    arrays.update(
        {
            "reward_profile": np.asarray(episode.reward_profile),
            "tail_kind": np.asarray(episode.tail_kind),
            "tail_bootstrap_value": episode.tail_bootstrap_value.cpu().numpy(),
        }
    )
    expected_fields = set(rollout_artifact_fields(episode.reward_profile))
    if set(arrays) != expected_fields:
        raise RuntimeError("rollout artifact payload does not match its explicit schema")
    write_npz(path, arrays)


def write_training_runtime_metadata(
    path: Path, runtime: FabricRolloutRuntime, resources: ResourceProfileConfig
) -> None:
    """Record common reproducibility metadata plus the RL runtime selections."""

    repository_root = Path(to_absolute_path("."))
    metadata = {
        **collect_repository_metadata(repository_root),
        "runtime": asdict(runtime.report),
        "checkpoint": asdict(runtime.checkpoint_report),
        "sampler": asdict(runtime.sampler_report),
        "guidance": asdict(runtime.guidance_config),
        "resources": resources.model_dump(mode="json"),
    }
    write_json(path, metadata)


def _trajectory_arrays(episode: RolloutEpisode) -> dict[str, np.ndarray]:
    return {
        name: _tensor(episode.audit, name).detach().cpu().numpy()
        for name in rollout_audit_keys(episode.reward_profile)
    }


def _tensor(trajectory: TensorDictBase, key: str) -> torch.Tensor:
    return cast(torch.Tensor, trajectory[key])
