"""Validate and compare serial, job-level, and vector evaluation artifacts."""

from __future__ import annotations

import json
from collections import Counter
from math import isfinite
from pathlib import Path
from typing import cast

from hydra.utils import to_absolute_path
from omegaconf import OmegaConf

from eco_planner.artifacts import collect_repository_metadata
from eco_planner.benchmarking.config import measurement, write_benchmark_artifacts
from eco_planner.configuration import load_resolved_yaml_mapping
from eco_planner.evaluation.artifacts import load_runtime_metadata


def write_report(
    serial: Path,
    job_level: Path,
    vector: Path,
    *,
    serial_wall_s: float,
    job_level_wall_s: float,
    vector_wall_s: float,
    output: Path,
) -> dict[str, object]:
    wall_times = (serial_wall_s, job_level_wall_s, vector_wall_s)
    if any(not isfinite(value) or value <= 0.0 for value in wall_times):
        raise ValueError("all evaluation wall times must be positive")
    report = build_report(
        serial,
        job_level,
        vector,
        serial_wall_s=serial_wall_s,
        job_level_wall_s=job_level_wall_s,
        vector_wall_s=vector_wall_s,
    )
    write_benchmark_artifacts(
        output.parent,
        OmegaConf.create(
            {
                "serial": str(serial.resolve()),
                "job_level": str(job_level.resolve()),
                "vector": str(vector.resolve()),
                "serial_wall_s": serial_wall_s,
                "job_level_wall_s": job_level_wall_s,
                "vector_wall_s": vector_wall_s,
            }
        ),
        output.name,
        report,
    )
    return report


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
    elapsed = [
        _number(_mapping(job.get("metadata"), "job metadata"), "elapsed_seconds") for job in jobs
    ]
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
    jobs: list[dict[str, object]] = []
    workloads: set[str] = set()
    for path in paths:
        config_path = path.parent / "resolved_config.yaml"
        if not config_path.is_file():
            raise ValueError(f"{path.parent} contains no resolved_config.yaml")
        payload = load_resolved_yaml_mapping(config_path)
        metadata = cast(dict[str, object], load_runtime_metadata(path).model_dump(mode="json"))
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
    runtime = _mapping(metadata.get("inference_runtime"), "runtime metadata")
    sampler = _mapping(metadata.get("sampler"), "sampler metadata")
    guidance = _mapping(metadata.get("guidance"), "guidance metadata")
    runtime_execution = _mapping(metadata.get("execution"), "execution metadata")
    workload_sampler = _mapping(workload.get("sampler"), "workload sampler")
    workload_guidance = _mapping(workload.get("guidance"), "workload guidance")
    if runtime["seed"] != workload["runtime_seed"]:
        raise ValueError("runtime metadata seed disagrees with resolved config")
    if sampler["name"] != workload_sampler["name"]:
        raise ValueError("runtime metadata sampler disagrees with resolved config")
    if guidance["name"] != workload_guidance["name"]:
        raise ValueError("runtime metadata guidance disagrees with resolved config")
    topology = execution["topology"]
    expected_mode = "parallel" if topology == "job_parallel" else "serial"
    if runtime_execution["mode"] != expected_mode:
        raise ValueError("runtime metadata execution mode disagrees with resolved topology")
    if runtime_execution["vector_env_slots"] != execution.get("resolved_vector_env_slots"):
        raise ValueError("runtime metadata vector slots disagree with resolved config")


def _execution(config: dict[str, object]) -> dict[str, object]:
    evaluation = _mapping(config.get("evaluation"), "resolved evaluation config")
    execution = dict(_mapping(evaluation.get("execution"), "resolved evaluation execution"))
    resources = config.get("resources")
    resource_mapping = None if resources is None else _mapping(resources, "resolved resources")
    execution["resolved_vector_env_slots"] = (
        None
        if execution.get("topology") != "vector" or resource_mapping is None
        else resource_mapping.get("evaluation_vector_env_slots")
    )
    return execution


def _workload(config: dict[str, object]) -> dict[str, object]:
    evaluation = dict(_mapping(config.get("evaluation"), "resolved evaluation config"))
    evaluation.pop("execution", None)
    evaluation.pop("matrix", None)
    env = _mapping(config.get("env"), "resolved environment config")
    runtime = _mapping(config.get("runtime"), "resolved runtime config")
    video = _mapping(config.get("video"), "resolved video config")
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
    expected = {"serial": "serial", "job_level": "job_parallel", "vector": "vector"}
    for group, jobs in groups.items():
        expected_topology = expected[group]
        for job in jobs:
            execution = _mapping(job.get("execution"), "job execution")
            if execution["topology"] != expected_topology:
                raise ValueError(
                    f"{group} input does not use execution.topology={expected_topology!r}"
                )
            slots = execution.get("resolved_vector_env_slots")
            expects_vector = expected_topology == "vector"
            if expects_vector != (type(slots) is int and slots > 0):
                raise ValueError(f"{group} input has an invalid vector_env_slots value")


def _validate_comparable_workloads(groups: dict[str, list[dict[str, object]]]) -> None:
    signatures = {
        group: Counter(_stable_json(_mapping(job.get("workload"), "job workload")) for job in jobs)
        for group, jobs in groups.items()
    }
    baseline = signatures["serial"]
    for group in ("job_level", "vector"):
        if signatures[group] != baseline:
            raise ValueError(f"{group} workloads do not match the serial scenario matrix")


def _stable_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _mapping(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a mapping")
    return cast(dict[str, object], value)


def _number(mapping: dict[str, object], field: str) -> float:
    value = mapping.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    return float(value)
