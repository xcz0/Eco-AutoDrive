"""RL-specific summaries and TensorDict-to-NPZ artifact adapters."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
from hydra.utils import to_absolute_path
from tensordict import TensorDict, TensorDictBase

from eco_planner.artifacts import collect_repository_metadata, write_json, write_npz
from eco_planner.planning.policy.inputs import POLICY_CONTEXT_KEYS
from eco_planner.rl.artifacts.schema import rollout_artifact_fields
from eco_planner.rl.reward.result import RewardProfileName
from eco_planner.rl.rollout.contracts import (
    RolloutEpisode,
    TailKind,
    rollout_audit_keys,
)
from eco_planner.runtime.resources import ResourceProfileConfig

if TYPE_CHECKING:
    from eco_planner.rl.rollout.runtime import FabricRolloutRuntime


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


def read_rollout_episode(path: Path) -> RolloutEpisode:
    """Rebuild one persisted episode from its stable NumPy audit artifact.

    The stored audit plus ``tail_kind``/``tail_bootstrap_value`` are the complete
    source of the compact PPO trajectory, so no separate training payload is needed.
    """

    with np.load(path, allow_pickle=False) as data:
        profile = cast(RewardProfileName, str(data["reward_profile"].item()))
        audit = TensorDict(
            {name: torch.from_numpy(data[name].copy()) for name in rollout_audit_keys(profile)},
            batch_size=[int(data["state_value"].shape[0])],
        )
        tail_kind = cast(TailKind, str(data["tail_kind"].item()))
        bootstrap = torch.from_numpy(data["tail_bootstrap_value"].copy())
    training = TensorDict(
        {
            **{key: audit[key] for key in POLICY_CONTEXT_KEYS},
            "guidance_action": audit["guidance_action"],
            "old_joint_guidance_log_prob": audit["old_joint_guidance_log_prob"].reshape(-1),
            "state_value": audit["state_value"],
        },
        batch_size=audit.batch_size,
    )
    next_transition = audit.select("reward_total", "terminated", "truncated").clone()
    next_transition.rename_key_("reward_total", "reward")
    next_transition["state_value"] = torch.cat((audit["state_value"][1:], bootstrap.reshape(1, 1)))
    done = next_transition["terminated"] | next_transition["truncated"]
    done[-1] = True
    next_transition["done"] = done
    training["next"] = next_transition
    return RolloutEpisode(training, audit, tail_kind, bootstrap, profile)


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
