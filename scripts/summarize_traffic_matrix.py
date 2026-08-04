"""Validate and summarize the fixed 20-episode MetaDrive traffic matrix."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

EXPECTED_SEEDS = frozenset(range(5))
EXPECTED_DENSITIES = frozenset({0.05, 0.10})
EXPECTED_SCENARIOS = frozenset({"long_straight", "long_mixed"})
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 0
POSITION_ERROR_LIMIT_M = 1e-3
HEADING_ERROR_LIMIT_RAD = 1e-4


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix_root", type=Path)
    parser.add_argument(
        "--partial",
        action="store_true",
        help="summarize only jobs with complete job-level summaries without claiming full coverage",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"required non-empty JSON file does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return payload


def _require_nonempty(path: Path) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"required non-empty artifact does not exist: {path}")


def _validate_trace(path: Path, episode: dict[str, Any]) -> None:
    _require_nonempty(path)
    with np.load(path, allow_pickle=False) as trace:
        required = {
            "warmup_states",
            "initial_noise",
            "predictions_local",
            "observation_neighbor_agents_past",
            "executed_states",
            "executed_terminated",
            "executed_truncated",
            "executed_plan_indices",
            "trajectory_position_errors_m",
            "trajectory_heading_errors_rad",
            "traffic_selected_ids",
            "traffic_participant_counts",
            "traffic_nearest_distance_m",
            "traffic_has_nearest",
        }
        missing = sorted(required - set(trace.files))
        if missing:
            raise ValueError(f"trace {path} is missing arrays: {missing}")
        for name in trace.files:
            value = trace[name]
            if value.dtype.kind in "fc" and not np.isfinite(value).all():
                raise ValueError(f"trace {path} array {name!r} contains non-finite values")
        warmup_states = trace["warmup_states"]
        executed_states = trace["executed_states"]
        noises = trace["initial_noise"]
        observations = trace["observation_neighbor_agents_past"]
        plan_indices = trace["executed_plan_indices"]
        if warmup_states.shape != (20, 7):
            raise ValueError(f"trace {path} must contain exactly 20 warmup states")
        if executed_states.ndim != 2 or executed_states.shape[1] != 7:
            raise ValueError(f"trace {path} executed states must have shape [N, 7]")
        if executed_states.shape[0] != int(episode["simulator_steps"]):
            raise ValueError(f"trace {path} simulator step count disagrees with summary")
        if noises.shape[0] != int(episode["plan_cycles"]):
            raise ValueError(f"trace {path} planning cycle count disagrees with summary")
        if observations.shape != (noises.shape[0], 32, 21, 11):
            raise ValueError(f"trace {path} has an invalid traffic observation shape")
        if plan_indices.shape != (executed_states.shape[0],):
            raise ValueError(f"trace {path} has an invalid executed plan-index time axis")
        if not np.array_equal(np.unique(plan_indices), np.arange(noises.shape[0])):
            raise ValueError(f"trace {path} plan indices are not contiguous")
        if trace["traffic_participant_counts"].shape != (noises.shape[0],):
            raise ValueError(f"trace {path} traffic counts are not planning-cycle aligned")
        if not np.any(trace["traffic_participant_counts"] > 0):
            raise ValueError(f"trace {path} never observed traffic within the query radius")
        if float(trace["trajectory_position_errors_m"].max()) >= POSITION_ERROR_LIMIT_M:
            raise ValueError(f"trace {path} exceeds the trajectory position error limit")
        if float(trace["trajectory_heading_errors_rad"].max()) >= HEADING_ERROR_LIMIT_RAD:
            raise ValueError(f"trace {path} exceeds the trajectory heading error limit")


def _bootstrap(values: np.ndarray) -> dict[str, float | list[float]]:
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("bootstrap values must be a non-empty finite vector")
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    indices = generator.integers(0, values.size, size=(BOOTSTRAP_SAMPLES, values.size))
    means = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "mean_bootstrap_95_percentile_interval": [
            float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)),
        ],
    }


def summarize_matrix(matrix_root: Path, *, partial: bool = False) -> dict[str, Any]:
    matrix_root = matrix_root.resolve()
    if not matrix_root.is_dir():
        raise NotADirectoryError(f"matrix root does not exist: {matrix_root}")
    report_path = matrix_root / ("partial_matrix_report.json" if partial else "matrix_report.json")
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {report_path}")

    jobs = sorted(
        (path for path in matrix_root.iterdir() if path.is_dir() and path.name.isdigit()),
        key=lambda path: int(path.name),
    )
    if not partial and len(jobs) != 10:
        raise ValueError(f"expected exactly 10 Hydra jobs, found {len(jobs)}")
    if partial:
        jobs = [job for job in jobs if (job / "summary.json").is_file()]
        if not jobs:
            raise ValueError("partial matrix has no complete Hydra jobs")

    observed_jobs: set[tuple[int, float]] = set()
    episodes: list[dict[str, Any]] = []
    for job_dir in jobs:
        _require_nonempty(job_dir / "resolved_config.yaml")
        _require_nonempty(job_dir / ".hydra" / "overrides.yaml")
        _require_nonempty(job_dir / "runtime_metadata.json")
        _require_nonempty(job_dir / "tracked_diff.patch")
        job_summary = _read_json(job_dir / "summary.json")
        config = job_summary["config"]
        seed = int(config["seed"])
        density = float(config["env"]["traffic_density"])
        job_key = (seed, density)
        if job_key in observed_jobs:
            raise ValueError(f"duplicate matrix job: seed={seed}, density={density}")
        observed_jobs.add(job_key)
        job_episodes = job_summary["episodes"]
        if not isinstance(job_episodes, list) or len(job_episodes) != 2:
            raise ValueError(f"job {job_dir} must contain exactly two episode summaries")
        scenario_names = {episode["scenario"]["name"] for episode in job_episodes}
        if scenario_names != EXPECTED_SCENARIOS:
            raise ValueError(f"job {job_dir} has unexpected scenarios: {scenario_names}")
        for episode in job_episodes:
            if episode["noise_seed"] != seed or episode["scenario"]["seed"] != seed:
                raise ValueError(f"job {job_dir} does not use paired map/noise seeds")
            if float(episode["traffic_density"]) != density:
                raise ValueError(f"job {job_dir} summary density disagrees with config")
            if not 2_000.0 <= float(episode["route_length_m"]) <= 5_000.0:
                raise ValueError(f"job {job_dir} route length is outside [2000, 5000] m")
            episode_dir = job_dir / episode["scenario"]["name"]
            _require_nonempty(episode_dir / "summary.json")
            _require_nonempty(episode_dir / "closed_loop.gif")
            _validate_trace(episode_dir / "trace.npz", episode)
            episodes.append(episode)

    expected_jobs = {(seed, density) for seed in EXPECTED_SEEDS for density in EXPECTED_DENSITIES}
    if not partial and observed_jobs != expected_jobs:
        raise ValueError(
            f"matrix job grid mismatch: missing={sorted(expected_jobs - observed_jobs)}, "
            f"unexpected={sorted(observed_jobs - expected_jobs)}"
        )
    if not partial and len(episodes) != 20:
        raise RuntimeError(f"expected 20 episodes, found {len(episodes)}")
    if partial and len(episodes) != 2 * len(observed_jobs):
        raise RuntimeError("partial matrix episode count does not match complete job count")

    grouped: defaultdict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        grouped[(episode["scenario"]["name"], float(episode["traffic_density"]))].append(episode)
    statistics: dict[str, Any] = {}
    for (scenario, density), group in sorted(grouped.items()):
        label = f"{scenario}/density_{density:.2f}"
        metrics = {
            "simulated_seconds": np.asarray(
                [episode["simulated_seconds"] for episode in group], dtype=np.float64
            ),
            "distance_m": np.asarray(
                [episode["distance_m"] for episode in group], dtype=np.float64
            ),
            "route_completion": np.asarray(
                [episode["route_completion"] for episode in group], dtype=np.float64
            ),
            "mean_speed_mps": np.asarray(
                [episode["speed_mps"]["mean"] for episode in group], dtype=np.float64
            ),
            "total_reward": np.asarray(
                [episode["total_reward"] for episode in group], dtype=np.float64
            ),
        }
        statistics[label] = {
            "episode_count": len(group),
            "metrics": {name: _bootstrap(values) for name, values in metrics.items()},
            "arrive_rate": float(np.mean([episode["arrive_dest"] for episode in group])),
            "collision_rate": float(
                np.mean(
                    [
                        episode["crash_vehicle"]
                        or episode["crash_object"]
                        or episode["crash_building"]
                        or episode["crash_human"]
                        for episode in group
                    ]
                )
            ),
            "out_of_road_rate": float(np.mean([episode["out_of_road"] for episode in group])),
        }

    episode_rows = [
        {
            "scenario": episode["scenario"]["name"],
            "seed": episode["noise_seed"],
            "traffic_density": episode["traffic_density"],
            "terminal_reason": episode["terminal_reason"],
            "simulated_seconds": episode["simulated_seconds"],
            "distance_m": episode["distance_m"],
            "route_completion": episode["route_completion"],
            "mean_speed_mps": episode["speed_mps"]["mean"],
            "total_reward": episode["total_reward"],
        }
        for episode in sorted(
            episodes,
            key=lambda item: (
                item["scenario"]["name"],
                item["traffic_density"],
                item["noise_seed"],
            ),
        )
    ]
    report = {
        "matrix_root": str(matrix_root),
        "matrix_complete": not partial,
        "observed_job_grid": [
            {"seed": seed, "traffic_density": density} for seed, density in sorted(observed_jobs)
        ],
        "expected_episode_count": 20,
        "validated_episode_count": len(episodes),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "interface_limits": {
            "trajectory_position_error_m": POSITION_ERROR_LIMIT_M,
            "trajectory_heading_error_rad": HEADING_ERROR_LIMIT_RAD,
        },
        "episodes": episode_rows,
        "statistics": statistics,
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def main() -> None:
    args = _parse_args()
    report = summarize_matrix(args.matrix_root, partial=args.partial)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
