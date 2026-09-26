"""RL rollout adapter for reward calibration, reweighting, and offline rescoring.

Pure reward math, configuration, calibration, and objective scalarization are
owned by ``eco_planner.reward``. This adapter reads and writes ``RolloutEpisode``
audit / training TensorDicts: it extracts per-substep reward inputs, delegates the
numeric work to ``eco_planner.reward``, and writes component scores and scalar
reward back into the episode.

The canonical closed-loop cadence gives one transition several simulator
substeps. Online evaluation stores the per-substep component scores, safety gates
and recalibration inputs in the audit, so offline reweighting/rescoring rebuild
the same per-substep objective and reduce it with the shared
``aggregate_substep_rewards`` rule instead of assuming ``min(gate) * sum(base)``.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import replace
from typing import cast

import numpy as np
import torch
from tensordict import TensorDictBase, cat

from eco_planner.reward import (
    MOTION_LIMITS,
    EnergyBandConfig,
    PlannerRFTEnergyRewardConfig,
    PlannerRFTNoEnergyRewardConfig,
    PlannerRFTObjectiveResult,
    RewardComponents,
    RewardProfileConfig,
    energy_band_thresholds,
    energy_only_prefix,
    energy_score_from_fuel,
    recompose_reward_prefix,
    scored_arrays,
)
from eco_planner.reward.components import EnergyRewardConfig
from eco_planner.rl import RolloutEpisode

COMPONENTS = ("ttc", "progress", "comfort", "speed", "energy")


def reward_profile(base: PlannerRFTNoEnergyRewardConfig, weight: float) -> RewardProfileConfig:
    if weight == 0:
        return base
    payload = base.model_dump()
    payload["name"] = "plannerrft_energy_v1"
    payload["weights"]["energy"] = weight
    return PlannerRFTEnergyRewardConfig.model_validate(payload)


def reweight(episode: RolloutEpisode, profile: RewardProfileConfig) -> RolloutEpisode:
    """Reuse stored scores with new weights; component settings are not reevaluated."""
    audit = episode.audit.clone()
    results = [
        recompose_reward_prefix(profile.weights.model_dump(), components, gates, count)
        for components, gates, count in _substep_scores(audit)
    ]
    for name in COMPONENTS:
        audit[f"reward_component_{name}"] = torch.tensor(
            [[getattr(result.components, name)] for result in results], dtype=torch.float32
        )
    return replace(_write_objectives(episode, audit, results), reward_profile=profile.name)


def energy_only_reward(episode: RolloutEpisode) -> RolloutEpisode:
    """Objective endpoint: per-substep safety gate times the audited energy score."""
    audit = episode.audit.clone()
    results = [energy_only_prefix(*inputs) for inputs in _substep_scores(audit)]
    return _write_objectives(episode, audit, results)


def _substep_scores(
    audit: TensorDictBase,
) -> Iterator[tuple[list[RewardComponents], list[float], int]]:
    """Convert stored float32 scores to Python doubles; reward owns prefix reduction."""

    components = [_substep_field(audit, f"reward_substep_component_{name}") for name in COMPONENTS]
    gates = _substep_field(audit, "reward_substep_safety_gate").tolist()
    counts = _substep_field(audit, "reward_substep_count").reshape(-1).tolist()
    for row, (gate_row, count) in enumerate(zip(gates, counts, strict=True)):
        scores = [
            RewardComponents(*values)
            for values in zip(*(field[row].tolist() for field in components), strict=True)
        ]
        yield scores, gate_row, count


def _write_objectives(
    episode: RolloutEpisode, audit: TensorDictBase, results: Sequence[PlannerRFTObjectiveResult]
) -> RolloutEpisode:
    for name in ("base_total", "total"):
        audit[f"reward_{name}"] = torch.tensor(
            [[getattr(result, name)] for result in results], dtype=torch.float32
        )
    training = episode.training.clone()
    training["next", "reward"] = audit["reward_total"].to(training["next", "reward"])
    return replace(episode, training=training, audit=audit)


def raw_arrays(episodes: list[RolloutEpisode]) -> dict[str, np.ndarray]:
    """Flatten the per-substep recalibration measurements across episodes."""

    audit = cat([e.audit for e in episodes])
    mask = _substep_mask(audit).numpy()
    result: dict[str, np.ndarray] = {}
    for key in ("route_progress_delta_m", *MOTION_LIMITS):
        values = _substep_field(audit, f"reward_substep_{key}").numpy().astype(np.float64)
        result[key] = values[mask]
    return result


def substep_counts(episodes: Sequence[RolloutEpisode]) -> np.ndarray:
    """Return the executed substep count of every transition, in episode order."""

    audit = cat([e.audit for e in episodes])
    return _substep_field(audit, "reward_substep_count").reshape(-1).numpy().astype(np.int64)


def rescore_energy(episode: RolloutEpisode, energy: EnergyRewardConfig) -> RolloutEpisode:
    """Replace the audited energy score with the calibrated-band rescore per substep."""
    if energy.mode != "calibrated_band":
        return episode
    audit = episode.audit.clone()
    mask = _substep_mask(audit).numpy()
    fuel = _substep_field(audit, "reward_substep_executed_fuel_proxy_step_energy_ml")
    distance = _substep_field(audit, "reward_substep_step_distance_m")
    scores = np.zeros(tuple(fuel.shape), dtype=np.float64)
    fuel_values = fuel.numpy().astype(np.float64)
    distance_values = distance.numpy().astype(np.float64)
    for row, column in zip(*np.nonzero(mask), strict=True):
        scores[row, column] = energy_score_from_fuel(
            energy, float(fuel_values[row, column]), float(distance_values[row, column])
        )[0]
    audit["reward_substep_component_energy"] = torch.from_numpy(scores).to(
        _substep_field(audit, "reward_substep_component_energy")
    )
    return replace(episode, audit=audit)


def rescore(episode: RolloutEpisode, profile: PlannerRFTNoEnergyRewardConfig) -> RolloutEpisode:
    audit = episode.audit.clone()
    mask = _substep_mask(audit).numpy()
    scores = scored_arrays(raw_arrays([episode]), profile)
    for name in ("progress", "comfort"):
        field = np.zeros(mask.shape, dtype=np.float64)
        field[mask] = scores[name]
        audit[f"reward_substep_component_{name}"] = torch.from_numpy(field).to(
            _substep_field(audit, f"reward_substep_component_{name}")
        )
    episode = rescore_energy(replace(episode, audit=audit), profile.energy)
    return reweight(episode, profile)


def verify_original_components(
    episodes: list[RolloutEpisode], base: PlannerRFTNoEnergyRewardConfig
) -> None:
    for episode in episodes:
        rebuilt = rescore(episode, base)
        for key in (
            "reward_component_progress",
            "reward_component_comfort",
            "reward_base_total",
            "reward_total",
        ):
            torch.testing.assert_close(rebuilt.audit[key], episode.audit[key], rtol=1e-6, atol=1e-7)


def apply_energy_band(
    calibrated: PlannerRFTNoEnergyRewardConfig,
    episodes: Sequence[RolloutEpisode],
    band: EnergyBandConfig,
) -> PlannerRFTNoEnergyRewardConfig:
    """Derive the energy representation from this batch's per-substep intensity distribution."""
    audit = cat([episode.audit for episode in episodes])
    mask = _substep_mask(audit).numpy()
    fuel = _substep_field(audit, "reward_substep_executed_fuel_proxy_step_energy_ml")
    distance = _substep_field(audit, "reward_substep_step_distance_m")
    valid = _substep_field(audit, "reward_substep_energy_distance_valid").numpy().astype(bool)
    fuel_values = fuel.numpy().astype(np.float64)[mask]
    distance_values = distance.numpy().astype(np.float64)[mask]
    valid_values = valid[mask]
    intensity = np.where(
        valid_values & (distance_values > 0.0), fuel_values * 1_000.0 / distance_values, 0.0
    )
    full, zero = energy_band_thresholds(intensity, band)
    payload = calibrated.model_dump()
    payload["energy"]["mode"] = "calibrated_band"
    payload["energy"]["band_full_score_ml_per_km"] = full
    payload["energy"]["band_zero_score_ml_per_km"] = zero
    return PlannerRFTNoEnergyRewardConfig.model_validate(payload)


def _substep_mask(audit: TensorDictBase) -> torch.Tensor:
    count = _substep_field(audit, "reward_substep_count").reshape(-1).long()
    limit = _substep_field(audit, "reward_substep_safety_gate").shape[1]
    return torch.arange(limit).unsqueeze(0) < count.unsqueeze(1)


def _substep_field(audit: TensorDictBase, key: str) -> torch.Tensor:
    return cast(torch.Tensor, audit[key])


__all__ = [
    "COMPONENTS",
    "apply_energy_band",
    "energy_only_reward",
    "raw_arrays",
    "rescore",
    "rescore_energy",
    "reward_profile",
    "reweight",
    "substep_counts",
    "verify_original_components",
]
