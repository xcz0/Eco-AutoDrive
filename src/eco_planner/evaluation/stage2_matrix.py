"""Strict validation and reporting for the stage-2 paired guidance matrix."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from omegaconf import OmegaConf

from eco_planner.evaluation.artifact_reader import load_json_artifact, load_trace_artifact
from eco_planner.evaluation.trace import validate_trace_arrays

PROFILE_ACTIONS = {
    "unguided": None,
    "neutral": (0.0, 0.0),
    "lateral_positive": (1.0, 0.0),
    "lateral_negative": (-1.0, 0.0),
    "longitudinal_positive": (0.0, 1.0),
    "longitudinal_negative": (0.0, -1.0),
}
EXPECTED_SCENARIOS = frozenset({"straight", "gentle_curve"})
POSITION_ERROR_LIMIT_M = 1e-3
HEADING_ERROR_LIMIT_RAD = 1e-4


@dataclass(frozen=True)
class GuidanceTrend:
    """Signed first-cycle offsets relative to the saved reference trajectory."""

    mean_lateral_offset_m: float
    mean_longitudinal_speed_delta_mps: float


def measure_first_cycle_guidance_trend(
    arrays: dict[str, np.ndarray],
) -> GuidanceTrend:
    """Measure first-cycle lateral and along-track velocity changes at 10 Hz."""

    reference = arrays.get("reference_predictions_local")
    prediction = arrays.get("predictions_local")
    observation = arrays.get("observation_ego_current_state")
    if reference is None or reference.shape[0] < 1 or reference.shape[1:] != (11, 80, 4):
        raise ValueError("guided trace must contain reference_predictions_local [P, 11, 80, 4]")
    if prediction is None or prediction.shape != reference.shape:
        raise ValueError("prediction and reference trace shapes must match")
    if (
        observation is None
        or observation.shape[0] != reference.shape[0]
        or observation.shape[1:] != (10,)
    ):
        raise ValueError("guided trace must contain planning-aligned ego observations")
    reference_ego = reference[0, 0].astype(np.float64)
    prediction_ego = prediction[0, 0].astype(np.float64)
    heading = reference_ego[:, 2:4]
    norms = np.linalg.norm(heading, axis=-1)
    if not np.isfinite(norms).all() or np.any(norms <= 1e-6):
        raise ValueError("first-cycle reference heading is non-finite or degenerate")
    tangent = heading / norms[:, None]
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    lateral = np.sum(normal * (prediction_ego[:, :2] - reference_ego[:, :2]), axis=-1)
    current_position = observation[0, :2].astype(np.float64)
    reference_points = np.vstack((current_position, reference_ego[:, :2]))
    prediction_points = np.vstack((current_position, prediction_ego[:, :2]))
    reference_velocity = np.diff(reference_points, axis=0) / 0.1
    prediction_velocity = np.diff(prediction_points, axis=0) / 0.1
    longitudinal = np.sum(tangent * (prediction_velocity - reference_velocity), axis=-1)
    if not np.isfinite(lateral).all() or not np.isfinite(longitudinal).all():
        raise ValueError("first-cycle guidance trend must be finite")
    return GuidanceTrend(
        mean_lateral_offset_m=float(lateral.mean()),
        mean_longitudinal_speed_delta_mps=float(longitudinal.mean()),
    )


def summarize_stage2_matrix(
    matrix_root: Path,
    stage1_ddim_root: Path,
    *,
    expected_seeds: tuple[int, ...],
    expected_accelerator: Literal["cpu", "cuda"],
    expected_precision: Literal["32-true", "16-mixed", "bf16-mixed"],
) -> dict[str, Any]:
    """Validate one explicit stage-2 matrix and write a non-overwriting report."""

    matrix_root = matrix_root.resolve()
    report_path = matrix_root / "matrix_report.json"
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {report_path}")
    report = build_stage2_matrix_report(
        matrix_root,
        stage1_ddim_root.resolve(),
        expected_seeds=expected_seeds,
        expected_accelerator=expected_accelerator,
        expected_precision=expected_precision,
    )
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def build_stage2_matrix_report(
    matrix_root: Path,
    stage1_ddim_root: Path,
    *,
    expected_seeds: tuple[int, ...],
    expected_accelerator: Literal["cpu", "cuda"],
    expected_precision: Literal["32-true", "16-mixed", "bf16-mixed"],
) -> dict[str, Any]:
    """Validate the pre-registered profile/seed/scenario grid without writing."""

    if not matrix_root.is_dir():
        raise NotADirectoryError(f"stage-2 matrix root does not exist: {matrix_root}")
    if not stage1_ddim_root.is_dir():
        raise NotADirectoryError(f"stage-1 DDIM root does not exist: {stage1_ddim_root}")
    seed_set = _validate_expected_seeds(expected_seeds)
    traces: dict[tuple[str, int, str], Path] = {}
    rows: list[dict[str, Any]] = []
    for profile, expected_action in PROFILE_ACTIONS.items():
        profile_root = matrix_root / profile
        jobs = _numbered_jobs(profile_root)
        if {int(job.name) for job in jobs} != seed_set:
            raise ValueError(f"profile {profile!r} has unexpected job seeds")
        for job in jobs:
            seed = int(job.name)
            _require_nonempty(job / "resolved_config.yaml")
            _require_nonempty(job / ".hydra" / "overrides.yaml")
            _require_nonempty(job / "runtime_metadata.json")
            _require_file(job / "tracked_diff.patch")
            config = OmegaConf.load(job / "resolved_config.yaml")
            if int(config.runtime.seed) != seed:
                raise ValueError(f"profile {profile!r} job {seed} has an unexpected runtime seed")
            if config.runtime.accelerator != expected_accelerator:
                raise ValueError(f"profile {profile!r} job {seed} has an unexpected accelerator")
            if config.runtime.precision != expected_precision:
                raise ValueError(f"profile {profile!r} job {seed} has an unexpected precision")
            if config.sampler.name != "ddim5" or float(config.sampler.initial_noise_scale) != 1.0:
                raise ValueError(f"profile {profile!r} job {seed} is not standard-Gaussian DDIM-5")
            _validate_profile_config(profile, expected_action, config.guidance)
            summary = _read_json(job / "summary.json")
            runtime = summary.get("runtime")
            if not isinstance(runtime, dict):
                raise TypeError(f"profile {profile!r} job {seed} has no runtime summary")
            _validate_runtime(
                runtime,
                expected_accelerator=expected_accelerator,
                expected_precision=expected_precision,
            )
            metadata = _read_json(job / "runtime_metadata.json")
            metadata_runtime = metadata.get("inference_runtime")
            if metadata_runtime != runtime:
                raise ValueError(f"profile {profile!r} job {seed} runtime metadata disagrees")
            episodes = summary.get("episodes")
            if not isinstance(episodes, list) or len(episodes) != 2:
                raise ValueError(f"profile {profile!r} job {seed} must contain two episodes")
            names = {_scenario_name(episode) for episode in episodes}
            if names != EXPECTED_SCENARIOS:
                raise ValueError(f"profile {profile!r} job {seed} scenarios are {names}")
            for episode in episodes:
                scenario = _scenario_name(episode)
                if episode.get("noise_seed") != seed:
                    raise ValueError("episode noise seed disagrees with the Hydra job")
                episode_dir = job / scenario
                if _read_json(episode_dir / "summary.json") != episode:
                    raise ValueError(f"episode summary copy disagrees: {episode_dir}")
                _require_nonempty(episode_dir / "closed_loop.gif")
                trace_path = episode_dir / "trace.npz"
                arrays = _load_trace(trace_path)
                validate_trace_arrays(
                    arrays,
                    expected_plan_cycles=int(episode["plan_cycles"]),
                    expected_simulator_steps=int(episode["simulator_steps"]),
                    expected_warmup_steps=0,
                )
                _validate_execution_error(trace_path, arrays)
                guided = expected_action is not None
                if guided != ("reference_predictions_local" in arrays):
                    raise ValueError(f"profile {profile!r} trace guidance schema disagrees")
                traces[(profile, seed, scenario)] = trace_path
                row = {
                    "profile": profile,
                    "seed": seed,
                    "scenario": scenario,
                    "terminal_reason": episode["terminal_reason"],
                    "simulated_seconds": episode["simulated_seconds"],
                    "distance_m": episode["distance_m"],
                    "route_completion": episode["route_completion"],
                    "mean_speed_mps": episode["speed_mps"]["mean"],
                    "total_reward": episode["total_reward"],
                }
                if guided:
                    row["first_cycle_guidance_trend"] = asdict(
                        measure_first_cycle_guidance_trend(arrays)
                    )
                rows.append(row)

    comparisons = _validate_paired_predictions(traces, stage1_ddim_root, seed_set)
    _validate_direction_signs(rows)
    expected_episode_count = len(PROFILE_ACTIONS) * len(seed_set) * len(EXPECTED_SCENARIOS)
    if len(rows) != expected_episode_count:
        raise RuntimeError(
            f"expected {expected_episode_count} validated episodes, found {len(rows)}"
        )
    return {
        "matrix_root": str(matrix_root),
        "stage1_ddim_root": str(stage1_ddim_root),
        "validated_episode_count": len(rows),
        "profiles": list(PROFILE_ACTIONS),
        "seeds": sorted(seed_set),
        "runtime": {
            "accelerator": expected_accelerator,
            "precision": expected_precision,
        },
        "scenarios": sorted(EXPECTED_SCENARIOS),
        "interface_limits": {
            "trajectory_position_error_m": POSITION_ERROR_LIMIT_M,
            "trajectory_heading_error_rad": HEADING_ERROR_LIMIT_RAD,
        },
        "paired_comparisons": comparisons,
        "episodes": sorted(rows, key=lambda row: (row["profile"], row["seed"], row["scenario"])),
    }


def _validate_paired_predictions(
    traces: dict[tuple[str, int, str], Path], stage1_root: Path, expected_seeds: frozenset[int]
) -> dict[str, int]:
    stage1_matches = 0
    neutral_matches = 0
    for seed in expected_seeds:
        for scenario in EXPECTED_SCENARIOS:
            unguided = _load_trace(traces[("unguided", seed, scenario)])
            neutral = _load_trace(traces[("neutral", seed, scenario)])
            stage1 = _load_trace(stage1_root / str(seed) / scenario / "trace.npz")
            for name in ("initial_noise", "predictions_local", "executed_states"):
                if not np.array_equal(unguided[name], stage1[name]):
                    raise ValueError(f"stage-2 unguided {name} disagrees with E-007")
                if not np.array_equal(neutral[name], unguided[name]):
                    raise ValueError(f"neutral {name} does not exactly match unguided")
            for name in ("executed_terminated", "executed_truncated"):
                if not np.array_equal(neutral[name], unguided[name]):
                    raise ValueError(f"neutral {name} does not exactly match unguided")
            if not np.array_equal(
                neutral["reference_predictions_local"], neutral["predictions_local"]
            ):
                raise ValueError("neutral prediction does not exactly equal its reference")
            stage1_matches += 1
            neutral_matches += 1
    return {
        "unguided_episode_matches_with_stage1_anchor": stage1_matches,
        "neutral_episode_matches_with_unguided": neutral_matches,
    }


def _validate_expected_seeds(expected_seeds: tuple[int, ...]) -> frozenset[int]:
    if not isinstance(expected_seeds, tuple) or not expected_seeds:
        raise ValueError("expected seeds must be a non-empty tuple")
    if any(type(seed) is not int or seed < 0 for seed in expected_seeds):
        raise TypeError("expected seeds must be non-negative integers")
    result = frozenset(expected_seeds)
    if len(result) != len(expected_seeds):
        raise ValueError("expected seeds must be unique")
    return result


def _validate_runtime(
    runtime: dict[str, Any],
    *,
    expected_accelerator: Literal["cpu", "cuda"],
    expected_precision: Literal["32-true", "16-mixed", "bf16-mixed"],
) -> None:
    expected = {
        "requested_accelerator": expected_accelerator,
        "resolved_accelerator": expected_accelerator,
        "requested_precision": expected_precision,
        "resolved_precision": expected_precision,
    }
    for name, value in expected.items():
        if runtime.get(name) != value:
            raise ValueError(f"runtime {name} must equal {value!r}")


def _validate_direction_signs(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        trend = row.get("first_cycle_guidance_trend")
        if not isinstance(trend, dict):
            continue
        profile = row["profile"]
        lateral = float(trend["mean_lateral_offset_m"])
        longitudinal = float(trend["mean_longitudinal_speed_delta_mps"])
        if profile == "lateral_positive" and lateral <= 0.0:
            raise ValueError("positive lateral guidance did not move left")
        if profile == "lateral_negative" and lateral >= 0.0:
            raise ValueError("negative lateral guidance did not move right")
        if profile == "longitudinal_positive" and longitudinal <= 0.0:
            raise ValueError("positive longitudinal guidance did not increase speed")
        if profile == "longitudinal_negative" and longitudinal >= 0.0:
            raise ValueError("negative longitudinal guidance did not decrease speed")


def _validate_profile_config(profile: str, action: tuple[float, float] | None, config: Any) -> None:
    if action is None:
        if config.name != "none":
            raise ValueError(f"profile {profile!r} must disable guidance")
        return
    if config.name != "orthogonal_reference":
        raise ValueError(f"profile {profile!r} must enable orthogonal reference guidance")
    actual = (float(config.lateral_scale), float(config.longitudinal_scale))
    if actual != action:
        raise ValueError(f"profile {profile!r} action {actual} does not equal {action}")


def _validate_execution_error(path: Path, arrays: dict[str, np.ndarray]) -> None:
    if float(arrays["trajectory_position_errors_m"].max()) >= POSITION_ERROR_LIMIT_M:
        raise ValueError(f"trace {path} exceeds the position execution-error limit")
    if float(arrays["trajectory_heading_errors_rad"].max()) >= HEADING_ERROR_LIMIT_RAD:
        raise ValueError(f"trace {path} exceeds the heading execution-error limit")


def _numbered_jobs(root: Path) -> list[Path]:
    if not root.is_dir():
        raise NotADirectoryError(f"stage-2 profile root does not exist: {root}")
    return sorted(
        (path for path in root.iterdir() if path.is_dir() and path.name.isdigit()),
        key=lambda path: int(path.name),
    )


def _load_trace(path: Path) -> dict[str, np.ndarray]:
    _require_nonempty(path)
    return load_trace_artifact(path).arrays


def _read_json(path: Path) -> dict[str, Any]:
    _require_nonempty(path)
    return load_json_artifact(path)


def _scenario_name(episode: object) -> str:
    if not isinstance(episode, dict):
        raise TypeError("episode summary must be an object")
    scenario = episode.get("scenario")
    if not isinstance(scenario, dict) or not isinstance(scenario.get("name"), str):
        raise TypeError("episode scenario name must be a string")
    return str(scenario["name"])


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"required artifact does not exist: {path}")


def _require_nonempty(path: Path) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"required non-empty artifact does not exist: {path}")
