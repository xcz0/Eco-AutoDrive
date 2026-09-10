"""Aggregation, ranking, and reporting for PPO stability stages."""

from __future__ import annotations

import optuna

from eco_planner.analysis.stability import summarize_search

from .config import PPOStabilityStudyConfig


def summarize_stage_a(study: optuna.Study, config: PPOStabilityStudyConfig) -> dict[str, object]:
    """Produce the stable Stage A report from persistent Optuna state."""

    return summarize_search(study, config.sampler_seed, config.stage_b.top_config_count)


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
