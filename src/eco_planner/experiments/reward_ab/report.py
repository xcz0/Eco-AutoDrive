"""Validate and summarize matched builtin/energy PPO training artifacts for human review."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np

from eco_planner.artifacts import write_json
from eco_planner.experiments.reward_ab.artifacts import (
    COMMON_PAIR_FIELDS,
    RewardABRunArtifacts,
    load_run,
)
from eco_planner.experiments.reward_ab.config import PPORewardABConfig, load_ab_config
from eco_planner.rl.artifacts import TrainingRunSummary, TrainingUpdateSummary


def summarize_ab(root: Path) -> dict[str, object]:
    config = load_ab_config(root / "study_manifest.yaml")
    pairs: list[dict[str, object]] = []
    mechanical_checks: list[dict[str, object]] = []
    for seed in config.matched_training.training_seeds:
        for replay in config.matched_training.replay_ids:
            pair_root = root / f"seed-{seed}-replay-{replay}"
            builtin = load_run(pair_root / "builtin", "metadrive_builtin_v1")
            energy = load_run(pair_root / "energy", "plannerrft_energy_v1")
            checks = _pair_checks(config, builtin, energy)
            mechanical_checks.extend(
                {"pair": f"seed-{seed}-replay-{replay}", **item} for item in checks
            )
            pairs.append(_pair_report(config, seed, replay, builtin, energy))
    passed = all(bool(item["passed"]) for item in mechanical_checks)
    return {
        "mechanical_status": "passed" if passed else "failed",
        "review_status": "pending_human_review" if passed else "mechanical_failure",
        "pairs": pairs,
        "cross_seed_summary": aggregate_pair_reports(pairs),
        "mechanical_checks": mechanical_checks,
        "review_questions": [
            "Did longitudinal guidance change beyond the configured deadband?",
            (
                "Did execution fuel-proxy intensity change, and is any decrease "
                "confounded by progress?"
            ),
            "Did mean speed or route progress cross a configured degradation threshold?",
            "Did collision or out-of-road counts increase?",
            "Are PPO loss, value loss, entropy, approximate KL, and gradients interpretable?",
        ],
    }


def _pair_checks(
    config: PPORewardABConfig,
    builtin: RewardABRunArtifacts,
    energy: RewardABRunArtifacts,
) -> list[dict[str, object]]:
    left = builtin.summary
    right = energy.summary
    expected_updates = config.matched_training.update_count
    checks = [
        {
            "name": "matched_training_seeds_and_replays",
            "passed": (
                left.training_seed == right.training_seed and left.replay_id == right.replay_id
            ),
        },
        {
            "name": "initial_policy_hash_matches",
            "passed": left.initial_policy_hash == right.initial_policy_hash,
        },
        {"name": "noise_seeds_match", "passed": left.noise_seeds == right.noise_seeds},
        {
            "name": "policy_action_seeds_match",
            "passed": left.policy_action_seeds == right.policy_action_seeds,
        },
        {
            "name": "configured_update_count_completed",
            "passed": len(left.updates) == len(right.updates) == expected_updates,
        },
        {
            "name": "both_policies_updated",
            "passed": (
                left.initial_policy_hash != left.final_policy_hash
                and right.initial_policy_hash != right.final_policy_hash
            ),
        },
        {
            "name": "both_frozen_planners_unchanged",
            "passed": (
                left.frozen_planner_hash_before == left.frozen_planner_hash_after
                and right.frozen_planner_hash_before == right.frozen_planner_hash_after
            ),
        },
        {
            "name": "pre_update_collection_is_exactly_paired",
            "passed": _initial_collection_equal(builtin, energy),
        },
    ]
    return checks


def _pair_report(
    config: PPORewardABConfig,
    seed: int,
    replay: int,
    builtin: RewardABRunArtifacts,
    energy: RewardABRunArtifacts,
) -> dict[str, object]:
    builtin_metrics = _effect_metrics(builtin)
    energy_metrics = _effect_metrics(energy)
    changes = _metric_changes(builtin_metrics, energy_metrics)
    thresholds = config.review_thresholds
    flags = {
        "longitudinal_action_changed": (
            abs(changes["longitudinal_action_mean_delta"])
            > thresholds.longitudinal_action_mean_deadband
        ),
        "energy_intensity_changed": (
            abs(changes["energy_intensity_change_fraction"])
            > thresholds.energy_intensity_deadband_fraction
        ),
        "progress_regressed": (
            changes["progress_change_fraction"] < -thresholds.maximum_progress_drop_fraction
        ),
        "mean_speed_regressed": (
            changes["mean_speed_change_fraction"] < -thresholds.maximum_mean_speed_drop_fraction
        ),
        "collision_regressed": (
            changes["collision_count_delta"] > thresholds.maximum_collision_count_increase
        ),
        "out_of_road_regressed": (
            changes["out_of_road_count_delta"] > thresholds.maximum_out_of_road_count_increase
        ),
        "energy_drop_confounded_by_progress_drop": (
            changes["energy_intensity_change_fraction"]
            < -thresholds.energy_intensity_deadband_fraction
            and changes["progress_change_fraction"] < 0.0
        ),
    }
    return {
        "training_seed": seed,
        "replay_id": replay,
        "effect_window": "updates 1..N-1; update 0 is the matched pre-update collection",
        "longitudinal_action_component": "guidance_action[:, 1]",
        "builtin": builtin_metrics,
        "energy": energy_metrics,
        "changes": changes,
        "update_series": _effect_update_series(builtin, energy),
        "review_flags": flags,
        "ppo_diagnostics": {
            "builtin": _ppo_diagnostics(builtin.summary),
            "energy": _ppo_diagnostics(energy.summary),
        },
    }


def _metric_changes(
    builtin_metrics: dict[str, float | int], energy_metrics: dict[str, float | int]
) -> dict[str, float | int]:
    return {
        "longitudinal_action_mean_delta": (
            energy_metrics["longitudinal_action_mean"] - builtin_metrics["longitudinal_action_mean"]
        ),
        "energy_intensity_change_fraction": _fraction_change(
            builtin_metrics["fuel_proxy_ml_per_km"], energy_metrics["fuel_proxy_ml_per_km"]
        ),
        "progress_change_fraction": _fraction_change(
            builtin_metrics["route_progress_delta"], energy_metrics["route_progress_delta"]
        ),
        "mean_speed_change_fraction": _fraction_change(
            builtin_metrics["mean_speed_mps"], energy_metrics["mean_speed_mps"]
        ),
        "collision_count_delta": (
            energy_metrics["collision_count"] - builtin_metrics["collision_count"]
        ),
        "out_of_road_count_delta": (
            energy_metrics["out_of_road_count"] - builtin_metrics["out_of_road_count"]
        ),
    }


def _effect_metrics(run: RewardABRunArtifacts) -> dict[str, float | int]:
    updates = run.summary.updates[1:]
    if not updates:
        raise ValueError("PPO reward A/B effect window requires at least one post-update rollout")
    return _effect_metrics_for_updates(updates)


def _effect_metrics_for_updates(
    updates: tuple[TrainingUpdateSummary, ...],
) -> dict[str, float | int]:
    sample_count = sum(item.sample_count for item in updates)
    if not sample_count:
        raise ValueError("PPO reward A/B effect window requires training samples")
    action_mean = sum(item.action_mean[1] * item.sample_count for item in updates) / sample_count
    action_variance = (
        sum(
            item.sample_count * (item.action_std[1] ** 2 + (item.action_mean[1] - action_mean) ** 2)
            for item in updates
        )
        / sample_count
    )
    distance = float(sum(item.executed_fuel_proxy_distance_m for item in updates))
    fuel = float(sum(item.executed_fuel_proxy_total_ml for item in updates))
    if distance <= 0.0:
        raise ValueError("PPO reward A/B effect window has zero execution distance")
    return {
        "sample_count": sample_count,
        "longitudinal_action_mean": float(action_mean),
        "longitudinal_action_variance": float(action_variance),
        "fuel_proxy_total_ml": fuel,
        "distance_m": distance,
        "fuel_proxy_ml_per_km": fuel * 1000.0 / distance,
        "mean_speed_mps": float(
            sum(item.mean_speed_mps * item.sample_count for item in updates) / sample_count
        ),
        "route_progress_delta": float(sum(item.route_completion_delta for item in updates)),
        "stopped_fraction": float(
            sum(item.stopped_fraction * item.sample_count for item in updates) / sample_count
        ),
        "collision_count": sum(item.collision_count for item in updates),
        "out_of_road_count": sum(item.out_of_road_count for item in updates),
    }


def _effect_update_series(
    builtin: RewardABRunArtifacts, energy: RewardABRunArtifacts
) -> list[dict[str, object]]:
    builtin_updates = builtin.summary.updates[1:]
    energy_updates = energy.summary.updates[1:]
    if len(builtin_updates) != len(energy_updates):
        raise ValueError("matched PPO reward runs have different effect-window lengths")
    series: list[dict[str, object]] = []
    for update_index, (builtin_update, energy_update) in enumerate(
        zip(builtin_updates, energy_updates, strict=True), start=1
    ):
        builtin_metrics = _effect_metrics_for_updates((builtin_update,))
        energy_metrics = _effect_metrics_for_updates((energy_update,))
        series.append(
            {
                "update_index": update_index,
                "builtin": builtin_metrics,
                "energy": energy_metrics,
            }
        )
    return series


def aggregate_pair_reports(pairs: list[dict[str, object]]) -> dict[str, object]:
    """Aggregate one effect estimate per matched seed/replay pair."""

    if not pairs:
        raise ValueError("PPO reward A/B aggregation requires at least one matched pair")
    changes = [cast(dict[str, float | int], pair["changes"]) for pair in pairs]
    flags = [cast(dict[str, bool], pair["review_flags"]) for pair in pairs]
    change_names = tuple(changes[0])
    if any(tuple(item) != change_names for item in changes[1:]):
        raise ValueError("PPO reward A/B pairs expose different change metrics")
    flag_names = tuple(flags[0])
    if any(tuple(item) != flag_names for item in flags[1:]):
        raise ValueError("PPO reward A/B pairs expose different review flags")

    statistics: dict[str, object] = {}
    for name in change_names:
        values = np.asarray([item[name] for item in changes], dtype=np.float64)
        statistics[name] = {
            "mean": float(values.mean()),
            "sample_standard_deviation": (float(values.std(ddof=1)) if values.size > 1 else None),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
        }
    return {
        "aggregation_unit": "matched training seed/replay pair",
        "pair_count": len(pairs),
        "training_seed_count": len({_integer_field(pair, "training_seed") for pair in pairs}),
        "replay_id_count": len({_integer_field(pair, "replay_id") for pair in pairs}),
        "change_statistics": statistics,
        "review_flag_counts": {
            name: sum(bool(item[name]) for item in flags) for name in flag_names
        },
    }


def _integer_field(payload: dict[str, object], field: str) -> int:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"PPO reward A/B pair {field!r} must be an integer")
    return value


def _ppo_diagnostics(summary: TrainingRunSummary) -> list[dict[str, float | int]]:
    return [
        {
            "update_index": item.update_index,
            "policy_loss": item.mean_policy_loss,
            "value_loss": item.mean_value_loss,
            "entropy": item.mean_entropy,
            "approximate_kl": item.mean_approximate_kl,
            "maximum_pre_clip_gradient_norm": item.maximum_pre_clip_gradient_norm,
        }
        for item in summary.updates
    ]


def _initial_collection_equal(left: RewardABRunArtifacts, right: RewardABRunArtifacts) -> bool:
    first = left.initial_update
    second = right.initial_update
    return all(np.array_equal(first[name], second[name]) for name in COMMON_PAIR_FIELDS)


def _fraction_change(baseline: float | int, candidate: float | int) -> float:
    baseline_value = float(baseline)
    if baseline_value <= 0.0:
        raise ValueError("PPO reward A/B fractional comparison requires a positive baseline")
    return (float(candidate) - baseline_value) / abs(baseline_value)


def summarize_and_write_ab(root: Path) -> dict[str, object]:
    report = summarize_ab(root)
    write_json(root / "review_report.json", report)
    return report
