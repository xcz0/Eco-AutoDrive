"""Validation and statistics for the fixed MetaDrive traffic matrix."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from eco_planner.evaluation.trace import validate_trace_arrays

EXPECTED_SEEDS = frozenset(range(5))
EXPECTED_DENSITIES = frozenset({0.05, 0.10})
EXPECTED_SCENARIOS = frozenset({"long_straight", "long_mixed"})
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 0
POSITION_ERROR_LIMIT_M = 1e-3
HEADING_ERROR_LIMIT_RAD = 1e-4


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
    """Validate and build a report without writing or overwriting an artifact."""

    matrix_root = matrix_root.resolve()
    if not matrix_root.is_dir():
        raise NotADirectoryError(f"matrix root does not exist: {matrix_root}")
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
        metadata = _read_json(job_dir / "runtime_metadata.json")
        _require_file(job_dir / "tracked_diff.patch")
        job_summary = _read_json(job_dir / "summary.json")
        runtime = _required_dict(job_summary, "runtime", job_dir / "summary.json")
        _validate_runtime_report(runtime, job_dir / "summary.json")
        metadata_runtime = _required_dict(
            metadata, "inference_runtime", job_dir / "runtime_metadata.json"
        )
        if metadata_runtime != runtime:
            raise ValueError(f"job {job_dir} runtime metadata disagrees with summary")
        checkpoint = _required_dict(job_summary, "checkpoint", job_dir / "summary.json")
        _required_int(checkpoint, "ema_tensor_count", job_dir / "summary.json")
        _required_int(checkpoint, "parameter_count", job_dir / "summary.json")
        seed = _required_int(runtime, "seed", job_dir / "summary.json")
        job_episodes = job_summary.get("episodes")
        if not isinstance(job_episodes, list) or len(job_episodes) != 2:
            raise ValueError(f"job {job_dir} must contain exactly two episode summaries")
        if not all(isinstance(episode, dict) for episode in job_episodes):
            raise TypeError(f"job {job_dir} episode summaries must be objects")
        density = _required_finite_float(
            job_episodes[0], "traffic_density", job_dir / "summary.json"
        )
        job_key = (seed, density)
        if job_key in observed_jobs:
            raise ValueError(f"duplicate matrix job: seed={seed}, density={density}")
        if job_key not in _expected_jobs():
            raise ValueError(f"unexpected matrix job: seed={seed}, density={density}")
        observed_jobs.add(job_key)
        scenario_names = {_scenario_name(episode, job_dir) for episode in job_episodes}
        if scenario_names != EXPECTED_SCENARIOS:
            raise ValueError(f"job {job_dir} has unexpected scenarios: {scenario_names}")
        for episode in job_episodes:
            _validate_episode_summary(episode, job_dir, seed, density)
            scenario_name = _scenario_name(episode, job_dir)
            episode_dir = job_dir / scenario_name
            persisted_episode = _read_json(episode_dir / "summary.json")
            if persisted_episode != episode:
                raise ValueError(f"episode summary copy disagrees with job summary: {episode_dir}")
            _require_nonempty(episode_dir / "closed_loop.gif")
            _validate_trace(episode_dir / "trace.npz", episode)
            episodes.append(episode)

    expected_jobs = _expected_jobs()
    if not partial and observed_jobs != expected_jobs:
        raise ValueError(
            f"matrix job grid mismatch: missing={sorted(expected_jobs - observed_jobs)}, "
            f"unexpected={sorted(observed_jobs - expected_jobs)}"
        )
    expected_episode_count = 2 * len(observed_jobs)
    if len(episodes) != expected_episode_count:
        raise RuntimeError("matrix episode count does not match validated job count")
    if not partial and len(episodes) != 20:
        raise RuntimeError(f"expected 20 episodes, found {len(episodes)}")

    return _build_report(matrix_root, partial, observed_jobs, episodes)


def _validate_trace(path: Path, episode: dict[str, Any]) -> None:
    _require_nonempty(path)
    with np.load(path, allow_pickle=False) as trace:
        arrays = {name: trace[name] for name in trace.files}
    validate_trace_arrays(
        arrays,
        expected_plan_cycles=_required_int(episode, "plan_cycles", path),
        expected_simulator_steps=_required_int(episode, "simulator_steps", path),
        expected_warmup_steps=20,
        require_traffic=True,
    )
    position_errors = arrays["trajectory_position_errors_m"]
    heading_errors = arrays["trajectory_heading_errors_rad"]
    if float(position_errors.max()) >= POSITION_ERROR_LIMIT_M:
        raise ValueError(f"trace {path} exceeds the trajectory position error limit")
    if float(heading_errors.max()) >= HEADING_ERROR_LIMIT_RAD:
        raise ValueError(f"trace {path} exceeds the trajectory heading error limit")


def _validate_episode_summary(episode: object, job_dir: Path, seed: int, density: float) -> None:
    if not isinstance(episode, dict):
        raise TypeError(f"job {job_dir} episode summary must be an object")
    scenario = _required_dict(episode, "scenario", job_dir / "summary.json")
    if _required_int(episode, "noise_seed", job_dir / "summary.json") != seed:
        raise ValueError(f"job {job_dir} does not use the configured noise seed")
    if _required_int(scenario, "seed", job_dir / "summary.json") != seed:
        raise ValueError(f"job {job_dir} does not use paired map/noise seeds")
    if _required_finite_float(episode, "traffic_density", job_dir / "summary.json") != density:
        raise ValueError(f"job {job_dir} summary density disagrees with config")
    route_length_m = _required_finite_float(episode, "route_length_m", job_dir / "summary.json")
    if not 2_000.0 <= route_length_m <= 5_000.0:
        raise ValueError(f"job {job_dir} route length is outside [2000, 5000] m")
    for name in (
        "simulated_seconds",
        "distance_m",
        "route_completion",
        "total_reward",
    ):
        _required_finite_float(episode, name, job_dir / "summary.json")
    speed = _required_dict(episode, "speed_mps", job_dir / "summary.json")
    _required_finite_float(speed, "mean", job_dir / "summary.json")
    for name in (
        "arrive_dest",
        "out_of_road",
        "crash_vehicle",
        "crash_object",
        "crash_building",
        "crash_human",
    ):
        if type(episode.get(name)) is not bool:
            raise TypeError(f"job {job_dir} episode field {name!r} must be boolean")


def _build_report(
    matrix_root: Path,
    partial: bool,
    observed_jobs: set[tuple[int, float]],
    episodes: list[dict[str, Any]],
) -> dict[str, Any]:
    grouped: defaultdict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        grouped[(_scenario_name(episode, matrix_root), float(episode["traffic_density"]))].append(
            episode
        )
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
    return {
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


def _read_json(path: Path) -> dict[str, Any]:
    _require_nonempty(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return payload


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"required artifact does not exist: {path}")


def _require_nonempty(path: Path) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"required non-empty artifact does not exist: {path}")


def _required_dict(mapping: dict[str, Any], name: str, path: Path) -> dict[str, Any]:
    value = mapping.get(name)
    if not isinstance(value, dict):
        raise TypeError(f"{path} field {name!r} must be an object")
    return value


def _required_int(mapping: dict[str, Any], name: str, path: Path) -> int:
    value = mapping.get(name)
    if type(value) is not int:
        raise TypeError(f"{path} field {name!r} must be an integer")
    return value


def _required_finite_float(mapping: dict[str, Any], name: str, path: Path) -> float:
    value = mapping.get(name)
    if type(value) not in {int, float} or not np.isfinite(value):
        raise TypeError(f"{path} field {name!r} must be finite and numeric")
    return float(value)


def _validate_runtime_report(runtime: dict[str, Any], path: Path) -> None:
    for name in (
        "requested_accelerator",
        "resolved_accelerator",
        "requested_precision",
        "resolved_precision",
        "device",
    ):
        value = runtime.get(name)
        if not isinstance(value, str) or not value:
            raise TypeError(f"{path} runtime field {name!r} must be a non-empty string")
    _required_int(runtime, "seed", path)
    if _required_int(runtime, "world_size", path) != 1:
        raise ValueError(f"{path} runtime world_size must be 1")


def _scenario_name(episode: object, path: Path) -> str:
    if not isinstance(episode, dict):
        raise TypeError(f"job {path} episode summary must be an object")
    scenario = _required_dict(episode, "scenario", path)
    name = scenario.get("name")
    if not isinstance(name, str) or not name:
        raise TypeError(f"job {path} episode scenario name must be a non-empty string")
    return name


def _expected_jobs() -> set[tuple[int, float]]:
    return {(seed, density) for seed in EXPECTED_SEEDS for density in EXPECTED_DENSITIES}
