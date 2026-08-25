"""Matrix report orchestration for validated evaluation artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eco_planner.evaluation.analysis.statistics import (
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    build_matrix_statistics,
)
from eco_planner.evaluation.analysis.validation import (
    HEADING_ERROR_LIMIT_RAD,
    POSITION_ERROR_LIMIT_M,
    ValidatedMatrix,
    validate_matrix_artifacts,
)


def summarize_matrix(matrix_root: Path, *, partial: bool = False) -> dict[str, Any]:
    """Validate, summarize, and persist a complete or explicitly partial matrix."""

    matrix_root = matrix_root.resolve()
    report_path = matrix_root / ("partial_matrix_report.json" if partial else "matrix_report.json")
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {report_path}")
    report = build_matrix_report(matrix_root, partial=partial)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def build_matrix_report(matrix_root: Path, *, partial: bool = False) -> dict[str, Any]:
    """Validate artifacts and build a report without writing or overwriting an artifact."""

    return _build_report(validate_matrix_artifacts(matrix_root, partial=partial))


def _build_report(validated: ValidatedMatrix) -> dict[str, Any]:
    episodes = validated.episodes
    episode_rows = [
        {
            "scenario": episode.scenario.name,
            "seed": episode.noise_seed,
            "traffic_density": episode.traffic_density,
            "terminal_reason": (episode.terminal_reason if episode.status == "completed" else None),
            "status": episode.status,
            "termination": episode.termination.model_dump(mode="json"),
            "simulated_seconds": (
                episode.simulated_seconds if episode.status == "completed" else None
            ),
            "distance_m": episode.distance_m if episode.status == "completed" else None,
            "energy_total_ml": episode.energy.total_ml if episode.status == "completed" else None,
            "energy_ml_per_km": episode.energy.ml_per_km if episode.status == "completed" else None,
            "route_completion": (
                episode.route_completion if episode.status == "completed" else None
            ),
            "mean_speed_mps": episode.speed_mps.mean if episode.status == "completed" else None,
            "total_reward": episode.total_reward if episode.status == "completed" else None,
        }
        for episode in sorted(
            episodes,
            key=lambda item: (item.scenario.name, item.traffic_density, item.noise_seed),
        )
    ]
    return {
        "matrix_root": str(validated.matrix_root),
        "matrix_complete": not validated.partial,
        "matrix_successful": all(episode.status == "completed" for episode in episodes),
        "observed_job_grid": [
            {"seed": seed, "traffic_density": density}
            for seed, density in sorted(validated.observed_jobs)
        ],
        "expected_job_grid": [
            {"seed": seed, "traffic_density": density}
            for seed, density in sorted(validated.expected_jobs)
        ],
        "expected_episode_count": validated.scenario_count * len(validated.expected_jobs),
        "validated_episode_count": len(episodes),
        "status_counts": {
            status: sum(episode.status == status for episode in episodes)
            for status in ("completed", "failed")
        },
        "termination_type_counts": {
            kind: sum(episode.termination.type == kind for episode in episodes)
            for kind in sorted({episode.termination.type for episode in episodes})
        },
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "interface_limits": {
            "trajectory_position_error_m": POSITION_ERROR_LIMIT_M,
            "trajectory_heading_error_rad": HEADING_ERROR_LIMIT_RAD,
        },
        "episodes": episode_rows,
        "statistics": build_matrix_statistics(episodes),
    }
