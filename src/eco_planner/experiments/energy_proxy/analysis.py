"""Issue #113 Task 1: fuel-proxy structural proof and E-060 speed attribution.

Offline diagnostic over the existing E-060 matched evaluation artifacts. No
simulator, model or training run is involved: the study re-derives the
MetaDrive fuel proxy from raw execution traces, demonstrates its structural
invariants on frozen synthetic counterfactuals, and decomposes the paired
A2-A1 intensity effect into the distance-weighted speed level and the
speed-profile shape factor using the exact identity

    ln I = ln 32.5 + 0.036 * mu_d + ln( sum_i w_i exp[0.036 (v_i - mu_d)] )

with w_i = step_distance_i / total_distance and mu_d = sum_i w_i v_i. The
identity is exact bookkeeping, not a causal mediation claim.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt
from scipy.stats import spearmanr

from eco_planner.contracts import SIMULATOR_STEP_S
from eco_planner.envs.domain.energy import EnergyTrace, MetaDriveFuelProxyProvider
from eco_planner.evaluation.artifacts import (
    CompletedEpisodeSummary,
    load_episode_summary,
    load_trace_artifact,
)
from eco_planner.rl.artifacts.summaries import TrainingRunSummary
from eco_planner.statistics import ScenarioBootstrapConfig, scenario_effect

PROXY_SCALE_ML_PER_KM = 32.5
PROXY_SPEED_EXP_COEFFICIENT = 0.036
IDENTITY_RESIDUAL_LIMIT = 1e-12
_SPEED_QUANTILES = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)

# Paired delta metrics that receive a per-seed scenario bootstrap interval.
_BOOTSTRAP_METRICS = (
    "intensity_ml_per_km",
    "ln_intensity",
    "speed_level_term",
    "speed_shape_term",
    "mu_d",
    "time_mean_speed_mps",
    "total_ml",
    "distance_m",
    "route_completion",
)
# All delta metrics reported per pair, per family and across seeds.
_DELTA_METRICS = (
    *_BOOTSTRAP_METRICS,
    "stopped_fraction",
    "accel_rms_mps2",
    "jerk_rms_mps3",
    "accel_decel_fraction",
    "identity_residual",
)
_LEAVE_ONE_OUT_METRICS = ("ln_intensity", "speed_level_term", "speed_shape_term")
_ACCELERATION_ASSOCIATIONS = (
    ("accel_rms_mps2", "speed_shape_term"),
    ("accel_rms_mps2", "ln_intensity"),
    ("jerk_rms_mps3", "speed_shape_term"),
)


class ParityConfig(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)

    rtol: StrictFloat = Field(gt=0.0)
    atol: StrictFloat = Field(ge=0.0)


class CounterfactualConfig(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)

    timestep_s: StrictFloat = Field(gt=0.0)
    step_count: StrictInt = Field(gt=0)
    base_speed_mps: StrictFloat = Field(ge=0.0)
    fluctuation_amplitude_mps: StrictFloat = Field(ge=0.0)
    stationary_step_count: StrictInt = Field(gt=0)


class AnalysisConfig(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)

    version: StrictInt
    study_name: str = Field(min_length=1)
    checkpoint_labels: list[str] = Field(min_length=1)
    contrast_primary: list[str] = Field(min_length=2, max_length=2)
    contrast_context: list[str] = Field(min_length=2, max_length=2)
    parity_include_baseline: StrictBool
    bootstrap: ScenarioBootstrapConfig
    parity: ParityConfig
    counterfactuals: CounterfactualConfig


def load_analysis_config(path: Path) -> AnalysisConfig:
    return AnalysisConfig.model_validate(OmegaConf.to_container(OmegaConf.load(path)))


def proxy_step_fuel_ml(speed_mps: np.ndarray, step_distance_m: np.ndarray) -> np.ndarray:
    """Independent folded proxy formula: 32.5 * exp(0.036 * v) * ds / 1000 mL.

    ``v`` is the end-of-substep speed in m/s (the provider's only speed input)
    and ``ds`` the executed center displacement in meters.
    """

    speed = np.asarray(speed_mps, dtype=np.float64)
    distance = np.asarray(step_distance_m, dtype=np.float64)
    return PROXY_SCALE_ML_PER_KM * np.exp(PROXY_SPEED_EXP_COEFFICIENT * speed) * distance / 1000.0


def intensity_decomposition(speed_mps: np.ndarray, step_distance_m: np.ndarray) -> dict[str, Any]:
    """Exact ln-intensity split into distance-weighted speed level and shape.

    Returns the recomputed proxy totals plus ``mu_d`` (distance-weighted mean
    speed), the speed-profile shape factor and the floating-point residual of
    the identity. Zero total distance yields undefined intensity entries
    (``None``) instead of an effective zero.
    """

    speed = np.asarray(speed_mps, dtype=np.float64)
    distance = np.asarray(step_distance_m, dtype=np.float64)
    total_distance = float(distance.sum(dtype=np.float64))
    total_ml = float(proxy_step_fuel_ml(speed, distance).sum(dtype=np.float64))
    if total_distance <= 0.0:
        return {
            "distance_m": total_distance,
            "total_ml": total_ml,
            "intensity_ml_per_km": None,
            "ln_intensity": None,
            "mu_d": None,
            "speed_shape_ln": None,
            "identity_residual": None,
        }
    weights = distance / total_distance
    mu_d = float((weights * speed).sum(dtype=np.float64))
    shape = float(
        (weights * np.exp(PROXY_SPEED_EXP_COEFFICIENT * (speed - mu_d))).sum(dtype=np.float64)
    )
    if shape <= 0.0:
        raise ValueError("speed-profile shape factor must be positive for a non-empty trace")
    shape_ln = math.log(shape)
    intensity = 1000.0 * total_ml / total_distance
    ln_intensity = math.log(intensity)
    residual = ln_intensity - (
        math.log(PROXY_SCALE_ML_PER_KM) + PROXY_SPEED_EXP_COEFFICIENT * mu_d + shape_ln
    )
    return {
        "distance_m": total_distance,
        "total_ml": total_ml,
        "intensity_ml_per_km": intensity,
        "ln_intensity": ln_intensity,
        "mu_d": mu_d,
        "speed_shape_ln": shape_ln,
        "identity_residual": residual,
    }


def kinematics_descriptives(speed_mps: np.ndarray) -> dict[str, Any]:
    """Descriptive acceleration/jerk statistics from end-of-substep speed norms.

    These are finite differences of the speed norm at the simulator cadence,
    not signed longitudinal dynamics; they carry no braking-loss, grade or
    powertrain information and never enter the proxy formula.
    """

    speed = np.asarray(speed_mps, dtype=np.float64)
    accel = np.diff(speed) / SIMULATOR_STEP_S if speed.size >= 2 else np.zeros(0)
    jerk = np.diff(accel) / SIMULATOR_STEP_S if accel.size >= 2 else np.zeros(0)

    def _stats(values: np.ndarray) -> dict[str, Any]:
        if values.size == 0:
            return {"mean": None, "std": None, "minimum": None, "maximum": None, "rms": None}
        return {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "rms": float(np.sqrt(np.mean(np.square(values)))),
        }

    accel_stats = _stats(accel)
    return {
        "accel_mean_mps2": accel_stats["mean"],
        "accel_std_mps2": accel_stats["std"],
        "accel_minimum_mps2": accel_stats["minimum"],
        "accel_maximum_mps2": accel_stats["maximum"],
        "accel_rms_mps2": accel_stats["rms"],
        "accel_decel_fraction": float(np.mean(accel < 0.0)) if accel.size else None,
        "jerk_mean_mps3": _stats(jerk)["mean"],
        "jerk_std_mps3": _stats(jerk)["std"],
        "jerk_max_abs_mps3": (float(np.max(np.abs(jerk))) if jerk.size else None),
        "jerk_rms_mps3": _stats(jerk)["rms"],
    }


def parity_checks(
    speed: np.ndarray,
    step_distance: np.ndarray,
    stored_fuel: np.ndarray,
    state_speed: np.ndarray,
    state_distance: np.ndarray,
    stopped_fraction: float,
    warmup_steps: int,
    summary: CompletedEpisodeSummary,
    parity: ParityConfig,
) -> dict[str, Any]:
    """Compare the stored proxy stream with an independent formula recomputation."""

    expected_fuel = proxy_step_fuel_ml(speed, step_distance)
    step_ok = np.isclose(expected_fuel, stored_fuel, rtol=parity.rtol, atol=parity.atol)
    with np.errstate(divide="ignore", invalid="ignore"):
        relative = np.where(
            stored_fuel > 0.0,
            np.abs(expected_fuel - stored_fuel) / stored_fuel,
            np.abs(expected_fuel - stored_fuel),
        )
    recomputed_total_ml = float(expected_fuel.sum(dtype=np.float64))
    recomputed_distance_m = float(step_distance.sum(dtype=np.float64))
    recomputed_intensity = (
        None
        if recomputed_distance_m == 0.0
        else 1000.0 * recomputed_total_ml / recomputed_distance_m
    )
    metrics = summary.metrics
    intensity_matches = (recomputed_intensity is None and metrics.energy.ml_per_km is None) or (
        recomputed_intensity is not None
        and metrics.energy.ml_per_km is not None
        and bool(
            np.isclose(
                recomputed_intensity, metrics.energy.ml_per_km, rtol=parity.rtol, atol=parity.atol
            )
        )
    )
    checks = {
        "step_formula_matches_stored_stream": bool(step_ok.all()),
        "speed_endpoint_matches_state_velocity": bool(
            np.allclose(speed, state_speed, rtol=0.0, atol=1e-12)
        ),
        "distance_matches_state_displacement": bool(
            np.allclose(step_distance, state_distance, rtol=0.0, atol=1e-12)
        ),
        "total_ml_matches_summary": bool(
            np.isclose(
                recomputed_total_ml, metrics.energy.total_ml, rtol=parity.rtol, atol=parity.atol
            )
        ),
        "distance_m_matches_summary": bool(
            np.isclose(
                recomputed_distance_m, metrics.energy.distance_m, rtol=parity.rtol, atol=parity.atol
            )
            and np.isclose(
                recomputed_distance_m, metrics.distance_m, rtol=parity.rtol, atol=parity.atol
            )
        ),
        "intensity_matches_summary": intensity_matches,
        "speed_summary_matches_trace": bool(
            np.isclose(float(speed.mean()), metrics.speed_mps.mean, rtol=1e-12, atol=1e-12)
            and np.isclose(float(speed.min()), metrics.speed_mps.minimum, rtol=1e-12, atol=1e-12)
            and np.isclose(float(speed.max()), metrics.speed_mps.maximum, rtol=1e-12, atol=1e-12)
        ),
        "stopped_fraction_matches_trace": bool(
            np.isclose(stopped_fraction, summary.metrics.stopped_fraction, rtol=1e-12, atol=1e-12)
        ),
        "warmup_steps_match_summary": bool(
            warmup_steps == summary.history_warmup.simulator_steps
            and summary.environment_steps_including_warmup == summary.simulator_steps + warmup_steps
        ),
    }
    return {
        "step_count": int(speed.size),
        "step_mismatch_indices": np.flatnonzero(~step_ok).tolist(),
        "max_step_relative_error": float(np.max(relative)) if relative.size else 0.0,
        "warmup_steps": warmup_steps,
        **checks,
        "passed": all(checks.values()),
    }


def audit_episode(
    arm: str,
    checkpoint_label: str,
    training_seed: int | None,
    episode_dir: Path,
    parity: ParityConfig,
) -> dict[str, Any]:
    """Load one episode, verify the proxy measurement chain, derive attribution facts."""

    summary = load_episode_summary(episode_dir / "summary.json")
    trace = load_trace_artifact(episode_dir / "trace.npz")
    arrays = trace.arrays
    record: dict[str, Any] = {
        "arm": arm,
        "checkpoint_label": checkpoint_label,
        "training_seed": training_seed,
        "scenario": summary.scenario.name,
        "map_sequence": summary.scenario.map_sequence,
        "scenario_seed": summary.scenario.seed,
        "noise_seed": summary.noise_seed,
        "status": summary.status,
        "trace_status": trace.trace_status,
        "episode_dir": str(episode_dir),
    }
    if not isinstance(summary, CompletedEpisodeSummary) or trace.trace_status != "complete":
        record["facts"] = None
        record["parity"] = {
            "audited": False,
            "reason": (
                f"status_{summary.status}"
                if not isinstance(summary, CompletedEpisodeSummary)
                else f"trace_status_{trace.trace_status}"
            ),
        }
        return record
    states = arrays["executed_states"]
    previous_positions = np.vstack([arrays["initial_state"][None, :2], states[:-1, :2]])
    record["parity"] = {
        "audited": True,
        **parity_checks(
            speed=arrays["executed_speed_mps"],
            step_distance=arrays["executed_step_distance_m"],
            stored_fuel=arrays["executed_fuel_proxy_step_energy_ml"],
            state_speed=np.linalg.norm(states[:, 3:5], axis=1),
            state_distance=np.linalg.norm(states[:, :2] - previous_positions, axis=1),
            stopped_fraction=float(arrays["executed_stopped"].mean()),
            warmup_steps=int(arrays["warmup_states"].shape[0]),
            summary=summary,
            parity=parity,
        ),
    }
    speed = arrays["executed_speed_mps"]
    decomposition = intensity_decomposition(speed, arrays["executed_step_distance_m"])
    metrics = summary.metrics
    record["facts"] = {
        "step_count": int(speed.size),
        "simulated_seconds": metrics.simulated_seconds,
        "distance_m": decomposition["distance_m"],
        "total_ml": decomposition["total_ml"],
        "intensity_ml_per_km": decomposition["intensity_ml_per_km"],
        "ln_intensity": decomposition["ln_intensity"],
        "mu_d": decomposition["mu_d"],
        "speed_shape_ln": decomposition["speed_shape_ln"],
        "identity_residual": decomposition["identity_residual"],
        "time_mean_speed_mps": float(speed.mean()),
        "speed_quantiles_mps": {
            str(q): float(v)
            for q, v in zip(
                _SPEED_QUANTILES,
                np.quantile(speed, np.asarray(_SPEED_QUANTILES)),
                strict=True,
            )
        },
        "route_completion": metrics.route_completion,
        "arrive_dest": metrics.arrive_dest,
        "terminated": summary.terminated,
        "truncated": summary.truncated,
        "terminal_reason": summary.terminal_reason,
        "stopped_fraction": float(arrays["executed_stopped"].mean()),
        **kinematics_descriptives(speed),
    }
    return record


def _audit_evaluation_directory(
    arm: str,
    checkpoint_label: str,
    training_seed: int | None,
    directory: Path,
    parity: ParityConfig,
) -> list[dict[str, Any]]:
    summary = _read_json_mapping(directory / "summary.json")
    records = []
    for scenario in summary["workload"]["scenarios"]:
        episode_dir = directory / str(scenario["name"])
        if (
            not (episode_dir / "trace.npz").is_file()
            or not (episode_dir / "summary.json").is_file()
        ):
            raise ValueError(f"evaluation episode artifact is missing: {episode_dir}")
        records.append(audit_episode(arm, checkpoint_label, training_seed, episode_dir, parity))
    return records


_MODEL_SKEW_NOTE = (
    "The historical E-057/E-058/E-060 evaluation summaries predate the "
    "policy_action provenance requirement added to JobSummary after E-060 "
    "(75d4326), so the current load_comparison cannot validate them. This "
    "study therefore performs targeted raw-JSON matched-key and checkpoint-pin "
    "checks over the same guard scope; the E-060-time full validation ran under "
    "its own pins."
)


@dataclass(frozen=True)
class SourceRun:
    """One manifest run with its training identity and evaluation directory."""

    arm: str
    checkpoint_label: str
    training_seed: int
    evaluation_dir: Path
    training: TrainingRunSummary
    policy_hash: str


@dataclass(frozen=True)
class SourceStudy:
    """Matched E-060 source study resolved from its manifest."""

    runs: tuple[SourceRun, ...]
    baseline_arm: str | None
    baseline_dir: Path | None
    scenario_keys: tuple[tuple[str, str, int], ...]
    noise_seed: int
    runtime_seed: int


def _read_json_mapping(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON mapping: {path}")
    return data


def _expected_policy_hash(entry: dict[str, Any], training: TrainingRunSummary) -> str:
    label = str(entry["checkpoint_label"])
    if label == "initial":
        return training.initial_policy_hash
    if label == "final":
        return training.final_policy_hash
    explicit = entry.get("checkpoint_hash")
    if not isinstance(explicit, str):
        raise ValueError(f"checkpoint label {label!r} requires an explicit checkpoint_hash")
    return explicit


def _evaluation_provenance(
    directory: Path, *, label: str | None, expected_hash: str | None
) -> dict[str, Any]:
    """Matched-key and checkpoint-pin guard over one raw evaluation summary."""

    summary = _read_json_mapping(directory / "summary.json")
    if summary.get("status") != "completed":
        raise ValueError(f"evaluation job is not completed: {directory}")
    episodes = summary.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError(f"evaluation summary has no episodes: {directory}")
    workload_scenarios = tuple(
        (item["name"], item["map"], item["seed"]) for item in summary["workload"]["scenarios"]
    )
    result_scenarios = tuple(
        (item["scenario"]["name"], item["scenario"]["map_sequence"], item["scenario"]["seed"])
        for item in episodes
    )
    if result_scenarios != workload_scenarios:
        raise ValueError(f"evaluation episodes must match the declared workload: {directory}")
    noise_seeds = {item["noise_seed"] for item in episodes}
    if len(noise_seeds) != 1:
        raise ValueError(f"evaluation episodes do not share one noise seed: {directory}")
    sampler = summary["sampler"]
    if sampler["name"] != "ddim5" or float(sampler["ddim_stochasticity"]) != 0.0:
        raise ValueError(
            f"matched evaluation requires the deterministic ddim5 sampler: {directory}"
        )
    checkpoint = summary.get("policy_checkpoint")
    if expected_hash is None:
        if checkpoint is not None:
            raise ValueError(f"frozen evaluation must not carry a policy checkpoint: {directory}")
    elif (
        not isinstance(checkpoint, dict)
        or checkpoint.get("label") != label
        or checkpoint.get("policy_hash") != expected_hash
    ):
        raise ValueError(
            f"evaluation checkpoint does not match the declared training state: {directory}"
        )
    return {
        "scenario_keys": workload_scenarios,
        "noise_seed": noise_seeds.pop(),
        "runtime_seed": int(summary["runtime"]["seed"]),
    }


def _load_source(source: Path) -> SourceStudy:
    """Resolve the E-060 manifest with matched-condition and checkpoint-pin guards."""

    container = OmegaConf.to_container(OmegaConf.load(source / "comparison.yaml"))
    if not isinstance(container, dict):
        raise ValueError("comparison manifest must be a mapping")
    manifest = cast(dict[str, Any], container)
    entries = manifest.get("runs")
    if not isinstance(entries, list) or not entries:
        raise ValueError("comparison manifest declares no runs")
    protocol = OmegaConf.load(source / Path(str(manifest["protocol"])))
    frozen_arms = [
        str(name) for name, arm in protocol.arms.items() if arm.get("reward_profile") is None
    ]
    if len(frozen_arms) != 1:
        raise ValueError("comparison protocol must declare exactly one frozen baseline arm")
    baseline_arm = frozen_arms[0]
    seeds_by_arm: dict[str, set[int]] = {}
    conditions_by_seed: dict[int, tuple[Any, ...]] = {}
    scenario_keys: tuple[tuple[str, str, int], ...] | None = None
    noise_seed: int | None = None
    runtime_seed: int | None = None
    runs: list[SourceRun] = []
    seen: set[tuple[str, int, str]] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("manifest run entries must be mappings")
        arm, label = str(entry["arm"]), str(entry["checkpoint_label"])
        training = TrainingRunSummary.model_validate_json(
            (source / Path(str(entry["training_summary"]))).read_text(encoding="utf-8")
        )
        policy_hash = _expected_policy_hash(entry, training)
        directory = (source / Path(str(entry["evaluation_dir"]))).resolve()
        provenance = _evaluation_provenance(directory, label=label, expected_hash=policy_hash)
        key = (arm, training.training_seed, label)
        if key in seen:
            raise ValueError(f"duplicate manifest run identity: {key}")
        seen.add(key)
        conditions = (
            training.initial_policy_hash,
            training.frozen_planner_hash_before,
            tuple(training.noise_seeds),
            tuple(training.policy_action_seeds),
            training.probe_before,
        )
        reference = conditions_by_seed.setdefault(training.training_seed, conditions)
        if conditions != reference:
            raise ValueError(
                f"trained runs do not share matched initial policy, probes and seeds: {key}"
            )
        if scenario_keys is None:
            scenario_keys = provenance["scenario_keys"]
        elif scenario_keys != provenance["scenario_keys"]:
            raise ValueError(f"matched scenario sets differ: {key}")
        if noise_seed is None:
            noise_seed = provenance["noise_seed"]
        elif noise_seed != provenance["noise_seed"]:
            raise ValueError(f"matched evaluation noise seeds differ: {key}")
        if runtime_seed is None:
            runtime_seed = provenance["runtime_seed"]
        elif runtime_seed != provenance["runtime_seed"]:
            raise ValueError(f"matched evaluation runtime seeds differ: {key}")
        seeds_by_arm.setdefault(arm, set()).add(training.training_seed)
        runs.append(
            SourceRun(
                arm=arm,
                checkpoint_label=label,
                training_seed=training.training_seed,
                evaluation_dir=directory,
                training=training,
                policy_hash=policy_hash,
            )
        )
    if len({frozenset(seeds) for seeds in seeds_by_arm.values()}) != 1:
        raise ValueError("comparison arms do not share matched training seeds")
    if set(protocol.training.seeds) != next(iter(seeds_by_arm.values())):
        raise ValueError("trained seeds differ from the protocol training pool")
    baseline_dir = (source / Path(str(manifest["baseline_evaluation_dir"]))).resolve()
    baseline_provenance = _evaluation_provenance(baseline_dir, label=None, expected_hash=None)
    if baseline_provenance["scenario_keys"] != scenario_keys:
        raise ValueError("baseline scenario set differs from the matched runs")
    if baseline_provenance["noise_seed"] != noise_seed:
        raise ValueError("baseline noise seed differs from the matched runs")
    if baseline_provenance["runtime_seed"] != runtime_seed:
        raise ValueError("baseline runtime seed differs from the matched runs")
    assert scenario_keys is not None and noise_seed is not None and runtime_seed is not None
    return SourceStudy(
        runs=tuple(runs),
        baseline_arm=baseline_arm,
        baseline_dir=baseline_dir,
        scenario_keys=scenario_keys,
        noise_seed=noise_seed,
        runtime_seed=runtime_seed,
    )


def _parity_summary(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    audited = [e for e in episodes if e["parity"].get("audited")]
    failed_checks = []
    step_mismatches = []
    for episode in audited:
        parity = episode["parity"]
        failed = sorted(k for k, v in parity.items() if k not in {"audited"} and v is False)
        if failed:
            failed_checks.append(
                {
                    "arm": episode["arm"],
                    "training_seed": episode["training_seed"],
                    "scenario": episode["scenario"],
                    "failed_checks": failed,
                }
            )
        if parity["step_mismatch_indices"]:
            step_mismatches.append(
                {
                    "arm": episode["arm"],
                    "training_seed": episode["training_seed"],
                    "scenario": episode["scenario"],
                    "step_mismatch_indices": parity["step_mismatch_indices"],
                }
            )
    residuals = [
        abs(e["facts"]["identity_residual"])
        for e in audited
        if e["facts"]["identity_residual"] is not None
    ]
    return {
        "episode_count": len(episodes),
        "audited_count": len(audited),
        "not_audited": [
            {
                "arm": e["arm"],
                "training_seed": e["training_seed"],
                "scenario": e["scenario"],
                "reason": e["parity"]["reason"],
            }
            for e in episodes
            if not e["parity"].get("audited")
        ],
        "max_step_relative_error": (
            max(e["parity"]["max_step_relative_error"] for e in audited) if audited else None
        ),
        "max_episode_identity_residual": max(residuals) if residuals else None,
        "undefined_intensity_episodes": [
            {
                "arm": e["arm"],
                "training_seed": e["training_seed"],
                "scenario": e["scenario"],
            }
            for e in audited
            if e["facts"]["identity_residual"] is None
        ],
        "warmup_steps_total": sum(e["parity"]["warmup_steps"] for e in audited),
        "episodes": [
            {
                "arm": e["arm"],
                "training_seed": e["training_seed"],
                "scenario": e["scenario"],
                "status": e["status"],
                "passed": e["parity"]["passed"],
                "max_step_relative_error": e["parity"]["max_step_relative_error"],
                "warmup_steps": e["parity"]["warmup_steps"],
            }
            for e in audited
        ],
        "step_mismatch_episodes": step_mismatches,
        "failed_check_episodes": failed_checks,
        "parity_passed": len(audited) == len(episodes)
        and not step_mismatches
        and not failed_checks,
    }


def _synthetic_energy_trace(step_speeds: np.ndarray, timestep_s: float) -> EnergyTrace:
    speeds = np.asarray(step_speeds, dtype=np.float64)
    points = np.concatenate([[speeds[0]], speeds])
    times = np.arange(points.size, dtype=np.float64) * timestep_s
    return EnergyTrace(time_s=times, speed_mps=points, step_distance_m=speeds * timestep_s)


def counterfactual_study(config: CounterfactualConfig) -> dict[str, Any]:
    """Frozen synthetic counterfactuals measured through the real provider."""

    provider = MetaDriveFuelProxyProvider()
    count = config.step_count
    pattern = np.array([-1.0, 0.0, 1.0, 0.0])
    fluctuating = (
        config.base_speed_mps + config.fluctuation_amplitude_mps * pattern[np.arange(count) % 4]
    )
    constant = np.full(count, config.base_speed_mps)
    profiles: dict[str, np.ndarray] = {
        "constant": constant,
        "fluctuating": fluctuating,
        "reordered": fluctuating[::-1].copy(),
        "constant_with_stationary_tail": np.concatenate(
            [constant, np.zeros(config.stationary_step_count)]
        ),
        "stationary": np.zeros(count),
    }
    results: dict[str, Any] = {}
    for name, speeds in profiles.items():
        trace = _synthetic_energy_trace(speeds, config.timestep_s)
        metrics = provider.measure(trace)
        decomposition = intensity_decomposition(speeds, trace.step_distance_m)
        provider_matches = (
            metrics.fuel_ml_per_km is None and decomposition["intensity_ml_per_km"] is None
        ) or (
            metrics.fuel_ml_per_km is not None
            and decomposition["intensity_ml_per_km"] is not None
            and bool(
                np.isclose(
                    metrics.fuel_ml_per_km,
                    decomposition["intensity_ml_per_km"],
                    rtol=1e-12,
                    atol=1e-15,
                )
            )
        )
        results[name] = {
            "step_count": int(speeds.size),
            "time_mean_speed_mps": float(speeds.mean()),
            "distance_m": metrics.distance_m,
            "total_ml": metrics.fuel_ml,
            "intensity_ml_per_km": metrics.fuel_ml_per_km,
            "mu_d": decomposition["mu_d"],
            "speed_shape_ln": decomposition["speed_shape_ln"],
            "ln_intensity": decomposition["ln_intensity"],
            "provider_matches_independent_formula": provider_matches,
        }

    def _intensity(name: str) -> float:
        value = results[name]["intensity_ml_per_km"]
        if value is None:
            raise ValueError(f"counterfactual profile {name!r} has undefined intensity")
        return float(value)

    fluctuation_intensity = _intensity("fluctuating")
    constant_intensity = _intensity("constant")
    return {
        "profiles": results,
        "invariants": {
            "reordering_invariance": {
                "profiles": ["fluctuating", "reordered"],
                "note": "identical distance-weighted speed distribution with different segment"
                " ordering and acceleration timing; intensity must not change",
                "total_ml_absolute_difference": abs(
                    results["reordered"]["total_ml"] - results["fluctuating"]["total_ml"]
                ),
                "intensity_relative_difference": abs(
                    _intensity("reordered") - fluctuation_intensity
                )
                / fluctuation_intensity,
                "invariant": bool(
                    np.isclose(
                        results["reordered"]["total_ml"],
                        results["fluctuating"]["total_ml"],
                        rtol=1e-12,
                        atol=1e-15,
                    )
                ),
            },
            "fluctuation_sensitivity": {
                "profiles": ["constant", "fluctuating"],
                "note": "identical plain (time-weighted) mean speed; higher fluctuation must"
                " raise intensity through the speed-profile shape factor, so mean speed alone"
                " does not determine the proxy",
                "time_mean_speed_equal": bool(
                    np.isclose(
                        results["constant"]["time_mean_speed_mps"],
                        results["fluctuating"]["time_mean_speed_mps"],
                        rtol=0.0,
                        atol=1e-12,
                    )
                ),
                "intensity_ratio_fluctuating_over_constant": fluctuation_intensity
                / constant_intensity,
                "intensity_strictly_increases": bool(fluctuation_intensity > constant_intensity),
            },
            "stationary_tail_invariance": {
                "profiles": ["constant", "constant_with_stationary_tail"],
                "note": "stopped time is invisible to proxy totals and intensity; only"
                " time-based statistics change",
                "total_ml_equal": bool(
                    np.isclose(
                        results["constant"]["total_ml"],
                        results["constant_with_stationary_tail"]["total_ml"],
                        rtol=0.0,
                        atol=1e-15,
                    )
                ),
                "intensity_equal": bool(
                    np.isclose(
                        constant_intensity,
                        _intensity("constant_with_stationary_tail"),
                        rtol=1e-12,
                        atol=1e-15,
                    )
                ),
                "time_mean_speed_changes": bool(
                    results["constant_with_stationary_tail"]["time_mean_speed_mps"]
                    < results["constant"]["time_mean_speed_mps"]
                ),
            },
            "stationary_undefined_intensity": {
                "profile": "stationary",
                "note": "zero distance: total is a defined 0.0 mL while per-km intensity is"
                " undefined (None), not an effective zero",
                "total_ml": results["stationary"]["total_ml"],
                "intensity_ml_per_km": results["stationary"]["intensity_ml_per_km"],
                "undefined_intensity_confirmed": (
                    results["stationary"]["intensity_ml_per_km"] is None
                ),
            },
        },
    }


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 2 or np.ptp(left) == 0.0 or np.ptp(right) == 0.0:
        return None
    return float(cast(Any, spearmanr(left, right))[0])


def _pair_deltas(reference: dict[str, Any], comparison: dict[str, Any]) -> dict[str, Any]:
    ref, cmp = reference["facts"], comparison["facts"]
    mu_d_delta = cmp["mu_d"] - ref["mu_d"]
    level_term = PROXY_SPEED_EXP_COEFFICIENT * mu_d_delta
    shape_term = cmp["speed_shape_ln"] - ref["speed_shape_ln"]
    ln_intensity_delta = cmp["ln_intensity"] - ref["ln_intensity"]
    deltas = {
        metric: cmp[metric] - ref[metric]
        for metric in (
            "intensity_ml_per_km",
            "total_ml",
            "distance_m",
            "route_completion",
            "time_mean_speed_mps",
            "mu_d",
            "stopped_fraction",
            "accel_rms_mps2",
            "jerk_rms_mps3",
            "accel_decel_fraction",
        )
    }
    deltas.update(
        {
            "ln_intensity": ln_intensity_delta,
            "speed_level_term": level_term,
            "speed_shape_term": shape_term,
            "identity_residual": ln_intensity_delta - level_term - shape_term,
        }
    )
    return deltas


def _leave_one_out(values: np.ndarray) -> dict[str, Any] | None:
    if values.size < 2:
        return None
    total = float(values.sum())
    full_mean = total / values.size
    loo_means = (total - values) / (values.size - 1)
    return {
        "sample_count": int(values.size),
        "full_mean": full_mean,
        "loo_mean_min": float(np.min(loo_means)),
        "loo_mean_max": float(np.max(loo_means)),
        "sign_change_count": int(sum(np.sign(mean) != np.sign(full_mean) for mean in loo_means)),
    }


def _paired_contrast(
    episodes: list[dict[str, Any]],
    reference_arm: str,
    comparison_arm: str,
    checkpoint_label: str,
    config: AnalysisConfig,
) -> dict[str, Any]:
    def _rows(arm: str) -> list[dict[str, Any]]:
        rows = [
            e for e in episodes if e["arm"] == arm and e["checkpoint_label"] == checkpoint_label
        ]
        if not rows:
            # The frozen-planner baseline carries its own label and pairs with
            # every checkpoint label; no other arm uses that label.
            rows = [e for e in episodes if e["arm"] == arm and e["checkpoint_label"] == "baseline"]
        return rows

    reference_rows, comparison_rows = _rows(reference_arm), _rows(comparison_arm)
    if not reference_rows or not comparison_rows:
        raise ValueError(
            f"contrast {reference_arm}->{comparison_arm} has no audited episodes at"
            f" label {checkpoint_label!r}"
        )
    noise_seeds = {e["noise_seed"] for e in reference_rows + comparison_rows}
    if len(noise_seeds) != 1:
        raise ValueError("contrast episodes do not share a single noise seed")
    scenarios = sorted({e["scenario"] for e in comparison_rows})
    for arm, rows in ((reference_arm, reference_rows), (comparison_arm, comparison_rows)):
        if sorted({e["scenario"] for e in rows}) != scenarios:
            raise ValueError(f"arm {arm} scenario set differs from the comparison arm")
    reference_index = {(e["scenario"], e["training_seed"]): e for e in reference_rows}
    comparison_index = {(e["scenario"], e["training_seed"]): e for e in comparison_rows}
    if len(reference_index) != len(reference_rows) or len(comparison_index) != len(comparison_rows):
        raise ValueError("duplicate episode identity within a contrast arm")
    reference_seeds = sorted(
        {e["training_seed"] for e in reference_rows}, key=lambda s: (s is None, s)
    )
    comparison_seeds = sorted(
        {e["training_seed"] for e in comparison_rows}, key=lambda s: (s is None, s)
    )
    if reference_seeds != comparison_seeds and reference_seeds != [None]:
        raise ValueError("contrast arms do not share matched training seeds")

    per_seed: dict[str, Any] = {}
    for comparison_seed in comparison_seeds:
        reference_seed = (
            comparison_seed if (scenarios[0], comparison_seed) in reference_index else None
        )
        pairs = []
        for scenario in scenarios:
            reference = reference_index[(scenario, reference_seed)]
            comparison = comparison_index[(scenario, comparison_seed)]
            if reference["map_sequence"] != comparison["map_sequence"]:
                raise ValueError(f"map family mismatch for scenario {scenario}")
            available = (
                reference["facts"] is not None
                and comparison["facts"] is not None
                and reference["facts"]["intensity_ml_per_km"] is not None
                and comparison["facts"]["intensity_ml_per_km"] is not None
            )
            reason = None
            if not available:
                if reference["facts"] is None or comparison["facts"] is None:
                    reason = "incomplete_episode"
                else:
                    reason = "undefined_intensity"
            pairs.append(
                {
                    "scenario": scenario,
                    "map_sequence": comparison["map_sequence"],
                    "training_seed": comparison_seed,
                    "available": available,
                    "unavailable_reason": reason,
                    "reference": reference["facts"],
                    "comparison": comparison["facts"],
                    "deltas": _pair_deltas(reference, comparison) if available else None,
                }
            )
        valid = [p for p in pairs if p["available"]]
        effects = {}
        for metric in _BOOTSTRAP_METRICS:
            values = np.asarray([p["deltas"][metric] for p in valid], dtype=np.float64)
            effects[metric] = scenario_effect(values, config.bootstrap)
        families: dict[str, dict[str, Any]] = {}
        for metric in _DELTA_METRICS:
            families[metric] = {
                family: {
                    "mean": float(
                        np.mean([p["deltas"][metric] for p in valid if p["map_sequence"] == family])
                    ),
                    "count": sum(1 for p in valid if p["map_sequence"] == family),
                }
                for family in sorted({p["map_sequence"] for p in valid})
            }
        leave_one_out = {
            metric: _leave_one_out(
                np.asarray([p["deltas"][metric] for p in valid], dtype=np.float64)
            )
            for metric in _LEAVE_ONE_OUT_METRICS
        }
        accel_associations = {
            f"{left}_vs_{right}": _spearman(
                np.asarray([p["deltas"][left] for p in valid], dtype=np.float64),
                np.asarray([p["deltas"][right] for p in valid], dtype=np.float64),
            )
            for left, right in _ACCELERATION_ASSOCIATIONS
        }
        per_seed[str(comparison_seed)] = {
            "pair_count": len(pairs),
            "available_pair_count": len(valid),
            "unavailable_pairs": [
                {"scenario": p["scenario"], "reason": p["unavailable_reason"]}
                for p in pairs
                if not p["available"]
            ],
            "pairs": pairs,
            "effects": effects,
            "families": families,
            "leave_one_out": leave_one_out,
            "acceleration_descriptives": accel_associations,
        }

    across_seeds = {}
    for metric in _BOOTSTRAP_METRICS:
        estimates = {
            seed: row["effects"][metric]["estimate"]
            for seed, row in per_seed.items()
            if row["effects"][metric]["estimate"] is not None
        }
        across_seeds[metric] = {
            "per_seed_estimates": estimates,
            "mean_of_estimates": (float(np.mean(list(estimates.values()))) if estimates else None),
            "per_seed_ci95": {
                seed: row["effects"][metric]["ci95"]
                for seed, row in per_seed.items()
                if row["effects"][metric]["ci95"] is not None
            },
            "note": "training seeds are not resampled; this is not a cross-seed"
            " population interval",
        }

    def _lowest(key: str, count: int = 3) -> list[dict[str, Any]]:
        valid_pairs = [p for seed in per_seed for p in per_seed[seed]["pairs"] if p["available"]]
        selected = sorted(
            valid_pairs,
            key=lambda p: min(p["reference"][key], p["comparison"][key]),
        )[:count]
        return [
            {
                "scenario": p["scenario"],
                "training_seed": p["training_seed"],
                "reference": p["reference"][key],
                "comparison": p["comparison"][key],
                "ln_intensity": p["deltas"]["ln_intensity"],
            }
            for p in selected
        ]

    extremes = {
        "non_completed_episodes": [
            {
                "arm": e["arm"],
                "training_seed": e["training_seed"],
                "scenario": e["scenario"],
                "status": e["status"],
            }
            for e in reference_rows + comparison_rows
            if e["facts"] is None
        ],
        "truncated_or_non_arrival_episodes": [
            {
                "arm": e["arm"],
                "training_seed": e["training_seed"],
                "scenario": e["scenario"],
                "arrive_dest": e["facts"]["arrive_dest"],
                "truncated": e["facts"]["truncated"],
                "terminal_reason": e["facts"]["terminal_reason"],
            }
            for e in reference_rows + comparison_rows
            if e["facts"] is not None and (not e["facts"]["arrive_dest"] or e["facts"]["truncated"])
        ],
        "lowest_distance_pairs": _lowest("distance_m"),
        "lowest_route_completion_pairs": _lowest("route_completion"),
    }
    max_identity_residual = max(
        abs(p["deltas"]["identity_residual"])
        for seed in per_seed
        for p in per_seed[seed]["pairs"]
        if p["available"]
    )
    return {
        "reference_arm": reference_arm,
        "comparison_arm": comparison_arm,
        "difference_direction": "comparison - reference",
        "checkpoint_label": checkpoint_label,
        "noise_seed": noise_seeds.pop(),
        "reference_has_training_seed": reference_seeds != [None],
        "per_seed": per_seed,
        "across_seeds": across_seeds,
        "extremes": extremes,
        "max_identity_residual": max_identity_residual,
    }


def _gate_inputs(
    parity: dict[str, Any], attribution: dict[str, Any], config: AnalysisConfig
) -> dict[str, Any]:
    label = config.checkpoint_labels[0]
    primary = attribution[label]["primary"]
    decomposition_exact = primary["max_identity_residual"] <= IDENTITY_RESIDUAL_LIMIT and (
        parity["max_episode_identity_residual"] is None
        or parity["max_episode_identity_residual"] <= IDENTITY_RESIDUAL_LIMIT
    )
    per_seed_shares = {}
    for seed, row in primary["per_seed"].items():
        valid = [p for p in row["pairs"] if p["available"]]
        ln_total = float(np.mean([p["deltas"]["ln_intensity"] for p in valid]))
        level = float(np.mean([p["deltas"]["speed_level_term"] for p in valid]))
        shape = float(np.mean([p["deltas"]["speed_shape_term"] for p in valid]))
        per_seed_shares[seed] = {
            "mean_ln_intensity_delta": ln_total,
            "mean_speed_level_term": level,
            "mean_speed_shape_term": shape,
            "level_share_of_total": None if ln_total == 0.0 else level / ln_total,
            "shape_share_of_total": None if ln_total == 0.0 else shape / ln_total,
        }
    return {
        "verdict_1_trace_formula_summary_consistency": parity["parity_passed"],
        "verdict_2_decomposition_exact": bool(decomposition_exact),
        "verdict_2_max_pair_identity_residual": primary["max_identity_residual"],
        "verdict_2_primary_contrast_per_seed": per_seed_shares,
        "verdict_3_structural_inputs": {
            "proxy_inputs": "end-of-substep speed norm and executed center displacement only",
            "explicit_terms_absent": [
                "acceleration",
                "braking/regeneration loss",
                "idle fuel",
                "road grade",
                "powertrain state",
            ],
        },
    }


def run(source: Path, config_path: Path, output: Path, *, figures: bool = True) -> dict[str, Any]:
    """Execute the frozen diagnostic over one E-060 matched comparison source."""

    from eco_planner.reporting.artifacts import separate_output

    source, output = separate_output(source, output)
    config = load_analysis_config(config_path)
    study = _load_source(source)

    provenance = {
        "source_manifest": str((source / "comparison.yaml").resolve()),
        "model_skew_note": _MODEL_SKEW_NOTE,
        "matched_scenarios": [list(key) for key in study.scenario_keys],
        "noise_seed": study.noise_seed,
        "runtime_seed": study.runtime_seed,
        "runs": [
            {
                "arm": run.arm,
                "checkpoint_label": run.checkpoint_label,
                "training_seed": run.training_seed,
                "reward_profile": run.training.reward_profile,
                "policy_hash": run.policy_hash,
                "evaluation_dir": str(run.evaluation_dir),
            }
            for run in study.runs
        ],
        "baseline": {
            "arm": study.baseline_arm,
            "evaluation_dir": str(study.baseline_dir),
        },
    }

    episodes: list[dict[str, Any]] = []
    for run in study.runs:
        if run.checkpoint_label in config.checkpoint_labels:
            episodes.extend(
                _audit_evaluation_directory(
                    run.arm,
                    run.checkpoint_label,
                    run.training_seed,
                    run.evaluation_dir,
                    config.parity,
                )
            )
    if config.parity_include_baseline and study.baseline_dir is not None:
        episodes.extend(
            _audit_evaluation_directory(
                str(study.baseline_arm), "baseline", None, study.baseline_dir, config.parity
            )
        )
    parity = _parity_summary(episodes)
    counterfactuals = counterfactual_study(config.counterfactuals)

    if not parity["parity_passed"]:
        result = {
            "status": "failed",
            "study_name": config.study_name,
            "source": str(source),
            "config": config.model_dump(mode="json"),
            "provenance": provenance,
            "parity": parity,
            "counterfactuals": counterfactuals,
            "reason": "trace->formula->summary parity failed; physical interpretation is paused"
            " until the data/definition error is located",
        }
        _publish(result, output, figures=figures)
        return result

    attribution = {}
    for label in config.checkpoint_labels:
        attribution[label] = {
            "primary": _paired_contrast(
                episodes, config.contrast_primary[0], config.contrast_primary[1], label, config
            ),
            "context": _paired_contrast(
                episodes, config.contrast_context[0], config.contrast_context[1], label, config
            ),
        }
    result = {
        "status": "completed",
        "study_name": config.study_name,
        "source": str(source),
        "config": config.model_dump(mode="json"),
        "provenance": provenance,
        "parity": parity,
        "counterfactuals": counterfactuals,
        "attribution": attribution,
        "gate_p1_inputs": _gate_inputs(parity, attribution, config),
    }
    _publish(result, output, figures=figures)
    return {
        "status": result["status"],
        "output_dir": str(output),
        "parity_passed": parity["parity_passed"],
        "gate_p1_inputs": result["gate_p1_inputs"],
    }


def _publish(result: dict[str, Any], output: Path, *, figures: bool) -> None:
    from eco_planner.artifacts import write_json
    from eco_planner.reporting.artifacts import write_analysis

    from .report import write_report

    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "summary.json", result)
    files: list[str] = []
    if figures:
        from .report import plot

        files = plot(result, output)
    write_analysis(
        output,
        result,
        files,
        experiment="energy-proxy-speed-attribution",
        source=Path(result["source"]),
    )
    write_report(result, output, files)
