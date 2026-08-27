"""Validate and summarize matched builtin/energy PPO training artifacts for human review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import numpy as np

from eco_planner.artifacts import write_json
from eco_planner.configuration import load_resolved_yaml_mapping
from eco_planner.rl.artifacts import TrainingRunSummary
from eco_planner.rl.rollout import AUDIT_FIELD_KEYS, ENERGY_REWARD_AUDIT_FIELD_KEYS
from scripts.studies.ppo_reward_ab import PPORewardABConfig, load_ab_config

_COMMON_PAIR_FIELDS = (
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


def summarize_ab(root: Path) -> dict[str, object]:
    config = load_ab_config(root / "study_manifest.yaml")
    pairs: list[dict[str, object]] = []
    mechanical_checks: list[dict[str, object]] = []
    for seed in config.matched_training.training_seeds:
        for replay in config.matched_training.replay_ids:
            pair_root = root / f"seed-{seed}-replay-{replay}"
            builtin = _load_run(pair_root / "builtin", "metadrive_builtin_v1")
            energy = _load_run(pair_root / "energy", "plannerrft_energy_v1")
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


def _load_run(path: Path, expected_profile: str) -> dict[str, object]:
    summary = TrainingRunSummary.model_validate_json(
        (path / "summary.json").read_text(encoding="utf-8")
    )
    if summary.reward_profile != expected_profile:
        raise ValueError(f"{path} uses reward profile {summary.reward_profile!r}")
    resolved = load_resolved_yaml_mapping(path / "resolved_config.yaml")
    updates = tuple(
        _load_update(path, index, expected_profile) for index in range(len(summary.updates))
    )
    return {"path": path, "summary": summary, "resolved": resolved, "updates": updates}


def _load_update(path: Path, update_index: int, expected_profile: str) -> dict[str, np.ndarray]:
    files = sorted((path / "updates" / f"update-{update_index:03d}").glob("*.npz"))
    if not files:
        raise ValueError(f"{path} update {update_index} has no rollout artifacts")
    values: dict[str, list[np.ndarray]] = {name: [] for name in _COMMON_PAIR_FIELDS}
    profile_fields = (
        ENERGY_REWARD_AUDIT_FIELD_KEYS
        if expected_profile == "plannerrft_energy_v1"
        else AUDIT_FIELD_KEYS
    )
    expected_fields = {*profile_fields, "reward_profile", "tail_kind", "tail_bootstrap_value"}
    for artifact in files:
        with np.load(artifact, allow_pickle=False) as arrays:
            if str(arrays["reward_profile"]) != expected_profile:
                raise ValueError(f"{artifact} reward profile disagrees with its run")
            if set(arrays.files) != expected_fields:
                raise ValueError(f"{artifact} does not match its strict reward artifact schema")
            missing = set(_COMMON_PAIR_FIELDS) - set(arrays.files)
            if missing:
                raise ValueError(f"{artifact} is missing common A/B fields: {sorted(missing)}")
            for name in arrays.files:
                value = arrays[name]
                if value.dtype.kind in "fc" and not np.isfinite(value).all():
                    raise ValueError(f"{artifact}:{name} contains non-finite values")
            for name in _COMMON_PAIR_FIELDS:
                value = arrays[name]
                values[name].append(value)
    return {name: np.concatenate(items, axis=0) for name, items in values.items()}


def _pair_checks(
    config: PPORewardABConfig, builtin: dict[str, object], energy: dict[str, object]
) -> list[dict[str, object]]:
    left = _summary(builtin)
    right = _summary(energy)
    expected_updates = config.matched_training.update_count
    checks = [
        {
            "name": "resolved_configs_match_except_reward",
            "passed": _matched_configs(builtin, energy),
        },
        {
            "name": "resolved_config_matches_matrix",
            "passed": _resolved_matches_matrix(config, builtin),
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
    builtin: dict[str, object],
    energy: dict[str, object],
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
            "builtin": _ppo_diagnostics(_summary(builtin)),
            "energy": _ppo_diagnostics(_summary(energy)),
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


def _effect_metrics(run: dict[str, object]) -> dict[str, float | int]:
    updates = _updates(run)[1:]
    if not updates:
        raise ValueError("PPO reward A/B effect window requires at least one post-update rollout")
    return _effect_metrics_for_updates(updates)


def _effect_metrics_for_updates(
    updates: tuple[dict[str, np.ndarray], ...],
) -> dict[str, float | int]:
    actions = np.concatenate([item["guidance_action"] for item in updates], axis=0)
    distance = float(sum(item["step_distance_m"].sum() for item in updates))
    fuel = float(sum(item["executed_fuel_proxy_step_energy_ml"].sum() for item in updates))
    if distance <= 0.0:
        raise ValueError("PPO reward A/B effect window has zero execution distance")
    speed = np.concatenate([item["speed_mps"] for item in updates], axis=0)
    progress = float(sum(item["route_completion_delta"].sum() for item in updates))
    collision = sum(
        int(
            (
                item["crash_vehicle"]
                | item["crash_object"]
                | item["crash_building"]
                | item["crash_human"]
                | item["crash_sidewalk"]
            ).sum()
        )
        for item in updates
    )
    return {
        "sample_count": int(actions.shape[0]),
        "longitudinal_action_mean": float(actions[:, 1].mean()),
        "longitudinal_action_variance": float(actions[:, 1].var()),
        "fuel_proxy_total_ml": fuel,
        "distance_m": distance,
        "fuel_proxy_ml_per_km": fuel * 1000.0 / distance,
        "mean_speed_mps": float(speed.mean()),
        "route_progress_delta": progress,
        "stopped_fraction": float(
            np.concatenate([item["stopped"] for item in updates], axis=0).mean()
        ),
        "collision_count": collision,
        "out_of_road_count": int(sum(item["out_of_road"].sum() for item in updates)),
    }


def _effect_update_series(
    builtin: dict[str, object], energy: dict[str, object]
) -> list[dict[str, object]]:
    builtin_updates = _updates(builtin)[1:]
    energy_updates = _updates(energy)[1:]
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
        "training_seed_count": len({int(pair["training_seed"]) for pair in pairs}),
        "replay_id_count": len({int(pair["replay_id"]) for pair in pairs}),
        "change_statistics": statistics,
        "review_flag_counts": {
            name: sum(bool(item[name]) for item in flags) for name in flag_names
        },
    }


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


def _matched_configs(left: dict[str, object], right: dict[str, object]) -> bool:
    first = dict(_resolved(left))
    second = dict(_resolved(right))
    for payload in (first, second):
        payload.pop("reward", None)
        payload.pop("name", None)
    return first == second


def _resolved_matches_matrix(config: PPORewardABConfig, run: dict[str, object]) -> bool:
    resolved = _resolved(run)
    training = resolved.get("training")
    rl = resolved.get("rl")
    scenarios = resolved.get("scenarios")
    if (
        not isinstance(training, dict)
        or not isinstance(rl, dict)
        or not isinstance(scenarios, list)
    ):
        return False
    expected_scenarios = [
        item.model_dump(mode="python") for item in config.matched_training.scenarios
    ]
    return (
        training.get("update_count") == config.matched_training.update_count
        and training.get("transitions_per_environment")
        == config.matched_training.transitions_per_environment
        and rl.get("scheduler_total_optimizer_steps")
        == config.matched_training.scheduler_total_optimizer_steps
        and scenarios == expected_scenarios
    )


def _initial_collection_equal(left: dict[str, object], right: dict[str, object]) -> bool:
    first = _updates(left)[0]
    second = _updates(right)[0]
    return all(np.array_equal(first[name], second[name]) for name in _COMMON_PAIR_FIELDS)


def _fraction_change(baseline: float | int, candidate: float | int) -> float:
    baseline_value = float(baseline)
    if baseline_value <= 0.0:
        raise ValueError("PPO reward A/B fractional comparison requires a positive baseline")
    return (float(candidate) - baseline_value) / abs(baseline_value)


def _summary(run: dict[str, object]) -> TrainingRunSummary:
    value = run["summary"]
    if not isinstance(value, TrainingRunSummary):
        raise TypeError("A/B run summary has an invalid internal type")
    return value


def _resolved(run: dict[str, object]) -> dict[str, object]:
    value = run["resolved"]
    if not isinstance(value, dict):
        raise TypeError("A/B resolved config has an invalid internal type")
    return value


def _updates(run: dict[str, object]) -> tuple[dict[str, np.ndarray], ...]:
    value = run["updates"]
    if not isinstance(value, tuple):
        raise TypeError("A/B update artifacts have an invalid internal type")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    report = summarize_ab(args.root.resolve())
    write_json(args.root.resolve() / "review_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["mechanical_status"] == "passed" else 1)


if __name__ == "__main__":
    main()
