"""Fixed-batch storage and explicit source provenance for experiment runs."""

from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from eco_planner._repository import REPOSITORY_ROOT
from eco_planner.artifacts import write_json
from eco_planner.configuration import ScenarioConfig, load_resolved_yaml_mapping
from eco_planner.rl.artifacts import write_rollout_episode
from eco_planner.rl.config import TrainingJobConfig, parse_training_config
from eco_planner.rl.rollout.contracts import (
    RewardProfileName,
    RolloutEpisode,
    TailKind,
    rollout_audit_keys,
)

SHARED_SOURCES = tuple(
    Path(__file__).with_name(name + ".py")
    for name in (
        "artifacts",
        "config",
        "runtime",
        "rewards",
        "calibration",
        "gradients",
    )
)


@dataclass(frozen=True)
class FixedBatch:
    episodes: list[RolloutEpisode]
    samples: list[dict[str, Any]]
    config: TrainingJobConfig
    resolved_config: dict[str, Any]
    summary: dict[str, Any]
    runtime_metadata: dict[str, Any]

    @property
    def scenario_ids(self) -> np.ndarray:
        return np.asarray([s["scenario_index"] for s in self.samples], dtype=np.int64)


def load_fixed_batch(source: Path) -> FixedBatch:
    resolved = load_resolved_yaml_mapping(source / "resolved_config.yaml")
    config = parse_training_config(OmegaConf.create(resolved))
    summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    if summary["kind"] != "fixed-batch" or summary["optimizer_steps"] != 0:
        raise ValueError("source must be a collected update-0 fixed batch")
    episodes, samples = load_batch(source)
    if summary["sample_count"] != len(samples) or summary["episode_count"] != len(episodes):
        raise ValueError("batch counts differ from collection summary")
    return FixedBatch(
        episodes,
        samples,
        config,
        resolved,
        summary,
        json.loads((source / "runtime_metadata.json").read_text(encoding="utf-8")),
    )


def load_batch(source: Path) -> tuple[list[RolloutEpisode], list[dict]]:
    samples = json.loads((source / "sample_index.json").read_text(encoding="utf-8"))["samples"]
    payload = torch.load(source / "training-batch.pt", map_location="cpu", weights_only=True)
    groups = [
        (key, list(rows))
        for key, rows in groupby(samples, key=lambda s: (s["scenario_index"], s["episode_index"]))
    ]
    if len({key for key, _ in groups}) != len(groups):
        raise ValueError("sample index repeats an episode out of order")
    episodes = []
    for ((slot, number), rows), training_data in zip(groups, payload, strict=True):
        path = source / "updates/update-000" / f"slot-{slot}-episode-{number}.npz"
        with np.load(path, allow_pickle=False) as z:
            profile = cast(RewardProfileName, str(z["reward_profile"].item()))
            audit = TensorDict(
                {k: torch.from_numpy(z[k].copy()) for k in rollout_audit_keys(profile)},
                batch_size=[len(rows)],
            )
            episode = RolloutEpisode(
                training=TensorDict(training_data, batch_size=[len(rows)]),
                audit=audit,
                tail_kind=cast(TailKind, str(z["tail_kind"].item())),
                tail_bootstrap_value=torch.from_numpy(z["tail_bootstrap_value"].copy()),
                reward_profile=profile,
            )
        for key in ("planning_cycle_index", "map_seed"):
            np.testing.assert_array_equal(audit[key].numpy().reshape(-1), [s[key] for s in rows])
        for key in episode.training.keys():
            if key != "next":
                expected = audit[key]
                if key == "old_joint_guidance_log_prob":
                    expected = expected.squeeze(-1)
                torch.testing.assert_close(episode.training[key], expected, rtol=0, atol=0)
        for key, audit_key in (
            ("reward", "reward_total"),
            ("terminated", "terminated"),
            ("truncated", "truncated"),
        ):
            torch.testing.assert_close(
                episode.training["next", key], audit[audit_key], rtol=0, atol=0
            )
        episodes.append(episode)
    return episodes, samples


def write_batch(
    output: Path,
    slots: Sequence[Sequence[RolloutEpisode]],
    scenarios: Sequence[ScenarioConfig],
) -> tuple[list[RolloutEpisode], list[dict[str, Any]]]:
    episodes, samples, training = [], [], []
    for slot, slot_episodes in enumerate(slots):
        for number, episode in enumerate(slot_episodes):
            write_rollout_episode(
                output / "updates/update-000" / f"slot-{slot}-episode-{number}.npz", episode
            )
            episodes.append(episode)
            training.append(episode.training.cpu().to_dict())
            samples.extend(
                {
                    "scenario_index": slot,
                    "scenario": scenarios[slot].name,
                    "map": scenarios[slot].map,
                    "map_seed": scenarios[slot].seed,
                    "episode_index": number,
                    "planning_cycle_index": int(cycle),
                }
                for cycle in episode.audit["planning_cycle_index"].reshape(-1).tolist()
            )
    torch.save(training, output / "training-batch.pt")
    write_json(output / "sample_index.json", {"samples": samples})
    return episodes, samples


def verify_reference(reference: Path, source: Path, batch: FixedBatch) -> dict[str, Any]:
    summary = json.loads((reference / "summary.json").read_text(encoding="utf-8"))
    if Path(summary["source_batch"]).resolve() != source.resolve():
        raise ValueError("reference used a different source batch")
    if summary["initial_policy_hash"] != batch.summary["initial_policy_hash"]:
        raise ValueError("reference used a different initial policy")
    samples = json.loads((reference / "sample_index.json").read_text(encoding="utf-8"))["samples"]
    if samples != batch.samples:
        raise ValueError("reference sample order differs from the source batch")
    return summary


def copy_sources(output: Path, files: Sequence[Path]) -> None:
    for source in files:
        target = output / "source" / source.relative_to(REPOSITORY_ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
