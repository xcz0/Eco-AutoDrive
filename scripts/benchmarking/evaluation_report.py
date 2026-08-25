"""Validate and compare serial, job-level, and vector evaluation artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from math import isfinite
from pathlib import Path

from hydra.utils import to_absolute_path
from omegaconf import OmegaConf

from eco_planner.artifacts import collect_repository_metadata
from eco_planner.evaluation.artifacts import load_runtime_metadata

from .common import measurement, write_benchmark_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("serial", type=Path)
    parser.add_argument("job_level", type=Path)
    parser.add_argument("vector", type=Path)
    parser.add_argument("--serial-wall-s", type=float, required=True)
    parser.add_argument("--job-level-wall-s", type=float, required=True)
    parser.add_argument("--vector-wall-s", type=float, required=True)
    parser.add_argument("--output", type=Path, default=Path("evaluation_modes.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wall_times = (args.serial_wall_s, args.job_level_wall_s, args.vector_wall_s)
    if any(not isfinite(value) or value <= 0.0 for value in wall_times):
        raise SystemExit("all evaluation wall times must be positive")
    report = build_report(
        args.serial,
        args.job_level,
        args.vector,
        serial_wall_s=args.serial_wall_s,
        job_level_wall_s=args.job_level_wall_s,
        vector_wall_s=args.vector_wall_s,
    )
    write_benchmark_artifacts(
        args.output.parent,
        OmegaConf.create(
            {
                "serial": str(args.serial.resolve()),
                "job_level": str(args.job_level.resolve()),
                "vector": str(args.vector.resolve()),
                "serial_wall_s": args.serial_wall_s,
                "job_level_wall_s": args.job_level_wall_s,
                "vector_wall_s": args.vector_wall_s,
            }
        ),
        args.output.name,
        report,
    )
    print(json.dumps(report, indent=2))


def build_report(
    serial_root: Path,
    job_level_root: Path,
    vector_root: Path,
    *,
    serial_wall_s: float,
    job_level_wall_s: float,
    vector_wall_s: float,
) -> dict[str, object]:
    wall_times = (serial_wall_s, job_level_wall_s, vector_wall_s)
    if any(not isfinite(value) or value <= 0.0 for value in wall_times):
        raise ValueError("all evaluation wall times must be positive finite values")
    groups = {
        "serial": _jobs(serial_root),
        "job_level": _jobs(job_level_root),
        "vector": _jobs(vector_root),
    }
    _validate_execution_modes(groups)
    _validate_comparable_workloads(groups)
    walls = {
        "serial": serial_wall_s,
        "job_level": job_level_wall_s,
        "vector": vector_wall_s,
    }
    return {
        "provenance": {
            **collect_repository_metadata(Path(to_absolute_path("."))),
            "sources": {
                "serial": str(serial_root.resolve()),
                "job_level": str(job_level_root.resolve()),
                "vector": str(vector_root.resolve()),
            },
        },
        "evaluation_modes": {
            name: _mode_report(jobs, walls[name]) for name, jobs in groups.items()
        },
    }


def _mode_report(jobs: list[dict[str, object]], outer_wall_s: float) -> dict[str, object]:
    elapsed = [float(job["metadata"]["elapsed_seconds"]) for job in jobs]
    return {
        "job_count": len(jobs),
        "outer_wall_s": measurement([outer_wall_s]),
        "job_elapsed_s": measurement(elapsed),
        "summed_job_elapsed_s": measurement([sum(elapsed)]),
        "jobs": jobs,
    }


def _jobs(root: Path) -> list[dict[str, object]]:
    paths = sorted(root.rglob("runtime_metadata.json"))
    if not paths:
        raise ValueError(f"{root} contains no runtime_metadata.json")
    jobs = []
    workloads: set[str] = set()
    for path in paths:
        config_path = path.parent / "resolved_config.yaml"
        if not config_path.is_file():
            raise ValueError(f"{path.parent} contains no resolved_config.yaml")
        payload = OmegaConf.to_container(
            OmegaConf.load(config_path), resolve=True, throw_on_missing=True
        )
        if not isinstance(payload, dict):
            raise TypeError(f"{config_path} must resolve to a mapping")
        metadata = load_runtime_metadata(path).model_dump(mode="json")
        execution = _execution(payload)
        workload = _workload(payload)
        _validate_metadata_matches_config(metadata, execution, workload)
        signature = _stable_json(workload)
        if signature in workloads:
            raise ValueError(f"{root} contains duplicate evaluation workloads")
        workloads.add(signature)
        jobs.append(
            {
                "runtime_metadata_path": str(path.resolve()),
                "metadata": metadata,
                "execution": execution,
                "workload": workload,
            }
        )
    return jobs


def _validate_metadata_matches_config(
    metadata: dict[str, object],
    execution: dict[str, object],
    workload: dict[str, object],
) -> None:
    runtime = metadata["inference_runtime"]
    sampler = metadata["sampler"]
    guidance = metadata["guidance"]
    runtime_execution = metadata["execution"]
    if runtime["seed"] != workload["runtime_seed"]:
        raise ValueError("runtime metadata seed disagrees with resolved config")
    if sampler["name"] != workload["sampler"]["name"]:
        raise ValueError("runtime metadata sampler disagrees with resolved config")
    if guidance["name"] != workload["guidance"]["name"]:
        raise ValueError("runtime metadata guidance disagrees with resolved config")
    if runtime_execution["mode"] != execution["mode"]:
        raise ValueError("runtime metadata execution mode disagrees with resolved config")
    if runtime_execution["vector_env_slots"] != execution.get("vector_env_slots"):
        raise ValueError("runtime metadata vector slots disagree with resolved config")


def _execution(config: dict[str, object]) -> dict[str, object]:
    evaluation = config.get("evaluation")
    if not isinstance(evaluation, dict) or not isinstance(evaluation.get("execution"), dict):
        raise ValueError("resolved evaluation config has no execution mapping")
    return dict(evaluation["execution"])


def _workload(config: dict[str, object]) -> dict[str, object]:
    evaluation_value = config.get("evaluation")
    env_value = config.get("env")
    runtime_value = config.get("runtime")
    video_value = config.get("video")
    if not all(
        isinstance(value, dict)
        for value in (evaluation_value, env_value, runtime_value, video_value)
    ):
        raise ValueError("resolved evaluation config is missing a required mapping")
    evaluation = dict(evaluation_value)
    evaluation.pop("execution", None)
    evaluation.pop("matrix", None)
    env = dict(env_value)
    runtime = dict(runtime_value)
    video = dict(video_value)
    return {
        "evaluation": evaluation,
        "env": {
            name: env.get(name)
            for name in (
                "traffic_density",
                "traffic_mode",
                "horizon",
                "trajectory_execution_steps",
            )
        },
        "model": config.get("model"),
        "sampler": config.get("sampler"),
        "guidance": config.get("guidance"),
        "runtime_seed": runtime["seed"],
        "scenarios": config.get("scenarios"),
        "video_enabled": video["enabled"],
    }


def _validate_execution_modes(groups: dict[str, list[dict[str, object]]]) -> None:
    expected = {
        "serial": ("serial", False),
        "job_level": ("parallel", False),
        "vector": ("serial", True),
    }
    for group, jobs in groups.items():
        expected_mode, expects_vector = expected[group]
        for job in jobs:
            execution = job["execution"]
            if execution["mode"] != expected_mode:
                raise ValueError(f"{group} input does not use execution.mode={expected_mode!r}")
            slots = execution.get("vector_env_slots")
            if expects_vector != (type(slots) is int and slots > 0):
                raise ValueError(f"{group} input has an invalid vector_env_slots value")


def _validate_comparable_workloads(groups: dict[str, list[dict[str, object]]]) -> None:
    signatures = {
        group: Counter(_stable_json(job["workload"]) for job in jobs)
        for group, jobs in groups.items()
    }
    baseline = signatures["serial"]
    for group in ("job_level", "vector"):
        if signatures[group] != baseline:
            raise ValueError(f"{group} workloads do not match the serial scenario matrix")


def _stable_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


if __name__ == "__main__":
    main()
