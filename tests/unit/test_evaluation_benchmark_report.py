from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarking.evaluation_report import _jobs, build_report
from pydantic import ValidationError


def _metadata(
    *, seed: int = 0, mode: str = "serial", vector_slots: int | None = None
) -> dict[str, object]:
    return {
        "git_head": "fixture",
        "git_status_short": [],
        "platform": "fixture",
        "python": "fixture",
        "torch": "fixture",
        "lightning": "fixture",
        "metadrive": "fixture",
        "pydantic": "fixture",
        "inference_runtime": {
            "requested_accelerator": "cpu",
            "resolved_accelerator": "cpu",
            "requested_precision": "32-true",
            "resolved_precision": "32-true",
            "device": "cpu",
            "seed": seed,
            "world_size": 1,
        },
        "sampler": {
            "name": "dpm10",
            "implementation": "diffusers",
            "num_steps": 10,
            "timesteps": None,
            "initial_noise_scale": 0.5,
            "ddim_stochasticity": 0.0,
            "parity_label": "baseline",
        },
        "guidance": {"name": "none"},
        "execution": {
            "mode": mode,
            "vector_env_slots": vector_slots,
            "launcher": "basic",
            "worker_count": 1,
            "torch_threads_per_worker": None,
            "deterministic": False,
            "resolved_accelerator": "cpu",
            "process_id": 1,
            "logical_cpu_count": 1,
        },
        "elapsed_seconds": 1.0,
        "cuda_memory": None,
    }


def _write_job(
    root: Path,
    *,
    mode: str,
    vector_slots: int | None,
    seed: int = 0,
    metadata: dict[str, object] | None = None,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "runtime_metadata.json").write_text(
        json.dumps(
            _metadata(seed=seed, mode=mode, vector_slots=vector_slots)
            if metadata is None
            else metadata
        ),
        encoding="utf-8",
    )
    slots = "null" if vector_slots is None else str(vector_slots)
    (root / "resolved_config.yaml").write_text(
        f"""evaluation:
  mode: no_traffic
  profile: benchmark
  history_warmup_steps: 0
  evaluated_horizon_steps: 20
  execution:
    mode: {mode}
    vector_env_slots: {slots}
    torch_threads_per_worker: null
    deterministic: false
env:
  traffic_density: 0.0
  traffic_mode: null
  horizon: 20
  trajectory_execution_steps: 5
model: {{args_path: args.json, checkpoint_path: model.pth}}
sampler: {{name: dpm10}}
guidance: {{name: none}}
runtime: {{seed: {seed}}}
scenarios:
  - {{name: straight, map: S, seed: 0}}
video: {{enabled: false}}
""",
        encoding="utf-8",
    )


def test_evaluation_report_rejects_missing_metadata(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="contains no runtime_metadata"):
        _jobs(tmp_path)


def test_evaluation_report_accepts_only_comparable_execution_modes(tmp_path: Path) -> None:
    serial = tmp_path / "serial"
    job_level = tmp_path / "job"
    vector = tmp_path / "vector"
    _write_job(serial, mode="serial", vector_slots=None)
    _write_job(job_level, mode="parallel", vector_slots=None)
    _write_job(vector, mode="serial", vector_slots=1)

    report = build_report(
        serial,
        job_level,
        vector,
        serial_wall_s=3.0,
        job_level_wall_s=2.0,
        vector_wall_s=1.0,
    )

    modes = report["evaluation_modes"]
    assert modes["serial"]["outer_wall_s"]["median"] == 3.0  # type: ignore[index]

    _write_job(vector, mode="serial", vector_slots=1, seed=1)
    with pytest.raises(ValueError, match="do not match"):
        build_report(
            serial,
            job_level,
            vector,
            serial_wall_s=3.0,
            job_level_wall_s=2.0,
            vector_wall_s=1.0,
        )


def test_evaluation_report_validates_metadata_schema(tmp_path: Path) -> None:
    _write_job(
        tmp_path,
        mode="serial",
        vector_slots=None,
        metadata={"elapsed_seconds": 1.0},
    )

    with pytest.raises(ValidationError):
        _jobs(tmp_path)


def test_evaluation_report_rejects_duplicate_workloads(tmp_path: Path) -> None:
    _write_job(tmp_path / "first", mode="serial", vector_slots=None)
    _write_job(tmp_path / "second", mode="serial", vector_slots=None)

    with pytest.raises(ValueError, match="duplicate"):
        _jobs(tmp_path)
