"""Run the fixed-seed energy benchmark matrix and aggregate episode artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = ROOT / "configs" / "benchmark" / "energy_matrix.yaml"


def _load_matrix(path: Path) -> dict[str, Any]:
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True, throw_on_missing=True)
    if not isinstance(raw, dict):
        raise TypeError("energy matrix config must resolve to a mapping")
    _validate_matrix(raw)
    return raw


def _validate_matrix(matrix: dict[str, Any]) -> None:
    if matrix.get("version") != 1:
        raise ValueError("energy matrix version must be 1")
    execution = matrix["execution"]
    if execution != {
        "simulator_hz": 10,
        "planner_hz": 2,
        "physics_world_step_size_s": 0.02,
        "decision_repeat": 5,
        "trajectory_execution_steps": 5,
    }:
        raise ValueError("energy matrix execution timing must encode 10 Hz / 2 Hz evaluation")
    metric = matrix["energy_metric"]
    if (
        metric["name"] != "metadrive_episode_energy"
        or metric["implementation"]
        != "recompute_metadrive_base_vehicle_formula_on_executed_trace"
        or metric["unit"] != "mL"
        or metric["sampling_interval_s"] != 0.1
    ):
        raise ValueError("energy matrix must use the 0.1 s MetaDrive fuel proxy contract")
    guidance = matrix["guidance_profiles"]
    ids = [item["id"] for item in guidance]
    if ids != ["none", "longitudinal_negative", "longitudinal_zero", "longitudinal_positive"]:
        raise ValueError("energy matrix guidance profiles are incomplete or out of order")
    jobs = matrix["jobs"]
    if not jobs:
        raise ValueError("energy matrix must contain jobs")
    scenario_names: set[str] = set()
    feature_names: set[str] = set()
    for job in jobs:
        if not job["scenarios"]:
            raise ValueError(f"energy matrix job {job['id']!r} has no scenarios")
        for scenario in job["scenarios"]:
            required = {
                "name",
                "map",
                "seed",
                "map_features",
                "traffic_condition",
                "termination_type",
                "energy_metric",
            }
            if set(scenario) != required:
                raise ValueError(f"scenario {scenario.get('name')!r} has incomplete metadata")
            if scenario["name"] in scenario_names:
                raise ValueError(f"duplicate energy matrix scenario {scenario['name']!r}")
            scenario_names.add(scenario["name"])
            feature_names.update(scenario["map_features"])
            if scenario["energy_metric"] != metric["name"]:
                raise ValueError("scenario energy metric disagrees with matrix metric")
    required_features = {
        "straight",
        "curve",
        "intersection",
        "merge",
        "lane_change",
        "speed_limit_30_kmh",
        "speed_limit_50_kmh",
        "low_density_traffic",
    }
    if not required_features <= feature_names:
        missing = sorted(required_features - feature_names)
        raise ValueError(f"energy matrix is missing features: {missing}")


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _scenario_metadata(job: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {scenario["name"]: scenario for scenario in job["scenarios"]}


def _validate_resolved_config(
    resolved: dict[str, Any], matrix: dict[str, Any], job: dict[str, Any]
) -> None:
    expected_scenarios = [
        {"name": item["name"], "map": item["map"], "seed": item["seed"]}
        for item in job["scenarios"]
    ]
    if resolved["scenarios"] != expected_scenarios:
        raise RuntimeError(f"resolved scenarios disagree with matrix job {job['id']!r}")
    if resolved["runtime"]["seed"] != matrix["planner_noise_seed"]:
        raise RuntimeError("resolved planner noise seed disagrees with energy matrix")
    env = resolved["env"]
    timing = matrix["execution"]
    if env["trajectory_execution_steps"] != timing["trajectory_execution_steps"]:
        raise RuntimeError("resolved execution prefix disagrees with energy matrix")
    if env["decision_repeat"] != timing["decision_repeat"]:
        raise RuntimeError("resolved decision_repeat disagrees with energy matrix")
    if env["physics_world_step_size"] != timing["physics_world_step_size_s"]:
        raise RuntimeError("resolved physics step disagrees with energy matrix")
    if env["programmatic_lane_speed_limit_kmh"] != job["speed_limit_kmh"]:
        raise RuntimeError("resolved speed limit disagrees with energy matrix")
    traffic = matrix["traffic_profiles"][job["traffic_profile"]]
    if env["traffic_density"] != traffic["density"]:
        raise RuntimeError("resolved traffic density disagrees with energy matrix")
    vehicle = matrix["vehicle_config"]
    if env["random_agent_model"] != vehicle["random_agent_model"]:
        raise RuntimeError("resolved vehicle randomization disagrees with energy matrix")
    for name in (
        "wheel_friction",
        "max_engine_force",
        "max_brake_force",
        "max_steering",
        "max_speed_km_h",
    ):
        if env["vehicle_config"][name] != vehicle[name]:
            raise RuntimeError(f"resolved vehicle_config.{name} disagrees with energy matrix")


def _collect_run(
    matrix: dict[str, Any],
    job: dict[str, Any],
    guidance: dict[str, Any],
    run_dir: Path,
    returncode: int,
) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    resolved_path = run_dir / "resolved_config.yaml"
    if not summary_path.exists() or not resolved_path.exists():
        return {
            "job": job["id"],
            "guidance": guidance["id"],
            "returncode": returncode,
            "status": "launcher_failure",
            "output_dir": str(run_dir),
            "episodes": [],
        }
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    resolved_raw = OmegaConf.to_container(
        OmegaConf.load(resolved_path), resolve=True, throw_on_missing=True
    )
    if not isinstance(resolved_raw, dict):
        raise TypeError("resolved evaluation config must be a mapping")
    _validate_resolved_config(resolved_raw, matrix, job)
    metadata = _scenario_metadata(job)
    episodes = []
    for episode in summary["episodes"]:
        scenario_name = episode["scenario"]["name"]
        if scenario_name not in metadata:
            raise RuntimeError(f"summary contains undeclared scenario {scenario_name!r}")
        episodes.append(
            {
                "scenario_metadata": metadata[scenario_name],
                "evaluation": episode,
            }
        )
    return {
        "job": job["id"],
        "guidance": guidance["id"],
        "returncode": returncode,
        "status": summary["status"],
        "output_dir": str(run_dir),
        "episodes": episodes,
    }


def run_matrix(matrix_path: Path, output_root: Path) -> int:
    matrix = _load_matrix(matrix_path)
    output_root.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(OmegaConf.create(matrix), output_root / "matrix_manifest.yaml", resolve=True)
    records: list[dict[str, Any]] = []
    failed = False
    for job in matrix["jobs"]:
        for guidance in matrix["guidance_profiles"]:
            run_dir = output_root / job["id"] / guidance["id"]
            command = [
                sys.executable,
                str(ROOT / "scripts" / "evaluate.py"),
                f"--config-name={job['experiment']}",
                f"guidance={guidance['config']}",
                f"hydra.run.dir={run_dir.as_posix()}",
            ]
            completed = subprocess.run(command, cwd=ROOT, check=False)
            record = _collect_run(matrix, job, guidance, run_dir, completed.returncode)
            records.append(record)
            _write_json(output_root / "matrix_summary.json", {"runs": records})
            failed = failed or completed.returncode != 0 or record["status"] != "completed"
    return 1 if failed else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run_matrix(args.matrix.resolve(), args.output_root.resolve()))


if __name__ == "__main__":
    main()
