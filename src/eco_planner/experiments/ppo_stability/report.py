"""Aggregation, ranking, and reporting for PPO stability stages."""

from __future__ import annotations

import math

import optuna
from optuna.importance import get_param_importances
from optuna.trial import FrozenTrial, TrialState

from eco_planner.experiments.ppo_stability.config import PPOStabilityStudyConfig


def summarize_stage_a(study: optuna.Study, config: PPOStabilityStudyConfig) -> dict[str, object]:
    """Produce the stable Stage A report from persistent Optuna state."""

    completed = sorted(
        (trial for trial in study.trials if trial.state == TrialState.COMPLETE),
        key=lambda item: (-(item.value or -math.inf), item.number),
    )
    importances: dict[str, float] | None = None
    importance_error: str | None = None
    if len(completed) >= 2:
        try:
            importances = get_param_importances(study)
        except (ValueError, ImportError) as error:
            importance_error = str(error)
    counts = {
        state.name.lower(): sum(trial.state == state for trial in study.trials)
        for state in (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL)
    }
    return {
        "study_name": study.study_name,
        "trial_count": len(study.trials),
        "state_counts": counts,
        "stability_counts": {
            "stable": counts["complete"],
            "unstable": counts["pruned"] + counts["fail"],
        },
        "top_configs": [
            trial_payload(item) for item in completed[: config.stage_b.top_config_count]
        ],
        "parameter_importances": importances,
        "parameter_importance_error": importance_error,
        "stability_region": [trial_payload(item) for item in study.trials],
    }


def rank_validation_configs(
    records: list[dict[str, object]], required_seed_count: int
) -> list[int]:
    """Rank candidates that completed and passed across every required seed."""

    by_config: dict[int, list[dict[str, object]]] = {}
    for record in records:
        config_id = record.get("config_id")
        if isinstance(config_id, bool) or not isinstance(config_id, int):
            raise ValueError("validation record config_id must be an integer")
        by_config.setdefault(config_id, []).append(record)
    ranked: list[tuple[float, float, int]] = []
    for config_id, items in by_config.items():
        if len(items) != required_seed_count or any(item["state"] != "complete" for item in items):
            continue
        evaluations = [evaluation_record(item) for item in items]
        if any(not value["passed"] for value in evaluations):
            continue
        worst_training = min(
            finite_number(item, "minimum_episode_length_retention") for item in items
        )
        mean_route = sum(
            finite_number(value, "route_progress_retention") for value in evaluations
        ) / len(evaluations)
        ranked.append((-worst_training, -mean_route, config_id))
    return [item[2] for item in sorted(ranked)]


def trial_payload(trial: FrozenTrial) -> dict[str, object]:
    return {
        "trial_number": trial.number,
        "state": trial.state.name.lower(),
        "value": trial.value,
        "parameters": dict(trial.params),
        "user_attributes": dict(trial.user_attrs),
    }


def evaluation_record(record: dict[str, object]) -> dict[str, object]:
    value = record.get("evaluation")
    if not isinstance(value, dict):
        raise ValueError("validation record evaluation must be a mapping")
    return value


def finite_number(record: dict[str, object], field: str) -> float:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"validation record {field!r} must be numeric")
    return float(value)
