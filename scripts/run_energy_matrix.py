"""Run the fixed-seed energy benchmark matrix without retrying failed episodes."""

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
    if matrix["execution"] != {
        "simulator_hz": 10,
        "planner_hz": 2,
        "physics_world_step_size_s": 0.02,
        "decision_repeat": 5,
        "trajectory_execution_steps": 5,
    }:
        raise ValueError("energy matrix must encode 10 Hz execution and 2 Hz replanning")
    if matrix["sampler"] != {"name": "ddim5", "ddim_stochasticity": 0.0}:
        raise ValueError("energy matrix must use deterministic DDIM5 for paired guidance")
    metric = matrix["energy_metric"]
    if (
        metric["name"] != "metadrive_fuel_proxy"
        or metric["implementation"] != "recompute_metadrive_base_vehicle_formula_on_executed_trace"
        or metric["unit"] != "mL"
        or metric["sampling_interval_s"] != 0.1
    ):
        raise ValueError("energy matrix must use the executed-trace MetaDrive fuel proxy")
    profiles = matrix["guidance_profiles"]
    if [profile["id"] for profile in profiles] != [
        "baseline",
        "longitudinal_negative",
        "longitudinal_zero",
        "longitudinal_positive",
    ]:
        raise ValueError("energy matrix guidance profiles are incomplete or out of order")
    names = [scenario["name"] for job in matrix["jobs"] for scenario in job["scenarios"]]
    if len(names) != len(set(names)):
        raise ValueError("energy matrix scenario names must be unique")
    required = {
        "cruise",
        "curve",
        "intersection",
        "merge_lane_change",
        "speed_limit_50_30_50",
        "traffic_follow",
    }
    if set(names) != required:
        raise ValueError("energy matrix does not contain its required fixed scenarios")


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _validate_resolved_config(
    resolved: dict[str, Any], matrix: dict[str, Any], job: dict[str, Any]
) -> None:
    expected_scenarios = [
        {"name": scenario["name"], "map": scenario["map"], "seed": scenario["seed"]}
        for scenario in job["scenarios"]
    ]
    if resolved["scenarios"] != expected_scenarios:
        raise RuntimeError(f"resolved scenarios disagree with matrix job {job['id']!r}")
    if resolved["runtime"]["seed"] != matrix["planner_noise_seed"]:
        raise RuntimeError("resolved planner noise seed disagrees with energy matrix")
    environment = resolved["env"]
    timing = matrix["execution"]
    for name, expected in (
        ("trajectory_execution_steps", timing["trajectory_execution_steps"]),
        ("decision_repeat", timing["decision_repeat"]),
        ("physics_world_step_size", timing["physics_world_step_size_s"]),
    ):
        if environment[name] != expected:
            raise RuntimeError(f"resolved env.{name} disagrees with energy matrix timing")
    vehicle = matrix["vehicle_config"]
    if environment["random_agent_model"] != vehicle["random_agent_model"]:
        raise RuntimeError("resolved vehicle randomization disagrees with energy matrix")


def _collect_run(
    matrix: dict[str, Any],
    job: dict[str, Any],
    guidance: dict[str, Any],
    run_dir: Path,
    returncode: int,
) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    resolved_path = run_dir / "resolved_config.yaml"
    if not summary_path.is_file() or not resolved_path.is_file():
        return {
            "job": job["id"],
            "guidance": guidance["id"],
            "returncode": returncode,
            "status": "launcher_failure",
            "output_dir": str(run_dir),
            "episodes": [],
        }
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    resolved = OmegaConf.to_container(
        OmegaConf.load(resolved_path), resolve=True, throw_on_missing=True
    )
    if not isinstance(resolved, dict):
        raise TypeError("resolved evaluation config must be a mapping")
    _validate_resolved_config(resolved, matrix, job)
    declared = {scenario["name"]: scenario for scenario in job["scenarios"]}
    episodes = []
    for episode in summary["episodes"]:
        scenario_name = episode["scenario"]["name"]
        if scenario_name not in declared:
            raise RuntimeError(f"summary contains undeclared scenario {scenario_name!r}")
        episodes.append({"scenario_metadata": declared[scenario_name], "evaluation": episode})
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
                f"planner/guidance={guidance['config']}",
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
