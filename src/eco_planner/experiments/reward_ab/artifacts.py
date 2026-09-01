"""Experiment-specific persistence boundary for PPO reward A/B runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from eco_planner.rl.artifacts import (
    BUILTIN_ROLLOUT_ARTIFACT_FIELDS,
    ENERGY_ROLLOUT_ARTIFACT_FIELDS,
    TrainingRunSummary,
)

COMMON_PAIR_FIELDS = (
    "guidance_action",
    "route_completion_delta",
    "speed_mps",
    "stopped",
    "crash_vehicle",
    "crash_object",
    "crash_building",
    "crash_human",
    "crash_sidewalk",
    "out_of_road",
    "step_distance_m",
    "native_step_energy_ml",
    "executed_fuel_proxy_step_energy_ml",
)


@dataclass(frozen=True, slots=True)
class RewardABRunArtifacts:
    path: Path
    summary: TrainingRunSummary
    initial_update: dict[str, np.ndarray]


def write_study_manifest(study_path: Path, output_root: Path) -> None:
    """Persist the resolved experiment manifest selected for this run."""

    OmegaConf.save(OmegaConf.load(study_path), output_root / "study_manifest.yaml", resolve=True)


def load_run(path: Path, expected_profile: str) -> RewardABRunArtifacts:
    """Load typed training output and the paired pre-update rollout audit."""

    summary = TrainingRunSummary.model_validate_json(
        (path / "summary.json").read_text(encoding="utf-8")
    )
    if summary.reward_profile != expected_profile:
        raise ValueError(f"{path} uses reward profile {summary.reward_profile!r}")
    return RewardABRunArtifacts(path, summary, load_update(path, 0, expected_profile))


def load_update(path: Path, update_index: int, expected_profile: str) -> dict[str, np.ndarray]:
    """Load one rollout update through the reward-profile-specific schema."""

    files = sorted((path / "updates" / f"update-{update_index:03d}").glob("*.npz"))
    if not files:
        raise ValueError(f"{path} update {update_index} has no rollout artifacts")
    values: dict[str, list[np.ndarray]] = {name: [] for name in COMMON_PAIR_FIELDS}
    expected_fields = set(
        ENERGY_ROLLOUT_ARTIFACT_FIELDS
        if expected_profile == "plannerrft_energy_v1"
        else BUILTIN_ROLLOUT_ARTIFACT_FIELDS
    )
    for artifact in files:
        with np.load(artifact, allow_pickle=False) as arrays:
            if str(arrays["reward_profile"]) != expected_profile:
                raise ValueError(f"{artifact} reward profile disagrees with its run")
            if set(arrays.files) != expected_fields:
                raise ValueError(f"{artifact} does not match its strict reward artifact schema")
            missing = set(COMMON_PAIR_FIELDS) - set(arrays.files)
            if missing:
                raise ValueError(f"{artifact} is missing common A/B fields: {sorted(missing)}")
            for name in arrays.files:
                value = arrays[name]
                if value.dtype.kind in "fc" and not np.isfinite(value).all():
                    raise ValueError(f"{artifact}:{name} contains non-finite values")
            for name in COMMON_PAIR_FIELDS:
                values[name].append(arrays[name])
    return {name: np.concatenate(items, axis=0) for name, items in values.items()}
