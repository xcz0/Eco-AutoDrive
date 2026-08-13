from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from eco_planner.evaluation.config import parse_evaluation_config
from eco_planner.evaluation.execution import configure_job_execution


def _config(*, accelerator: str = "cpu", video: bool = False, threads: int = 2):
    config = OmegaConf.create(
        {
            "name": "execution-test",
            "map_query_radius_m": 100.0,
            "runtime": {
                "accelerator": accelerator,
                "devices": 1,
                "precision": "32-true",
                "seed": 0,
            },
            "video": {
                "enabled": video,
                "fps": 2,
                "screen_width": 32,
                "screen_height": 32,
                "film_width": 32,
                "film_height": 32,
                "scaling": 1.0,
            },
            "evaluation": {
                "mode": "traffic",
                "profile": "matrix",
                "history_warmup_steps": 20,
                "evaluated_horizon_steps": 5,
                "execution": {
                    "mode": "parallel",
                    "launcher": "joblib",
                    "worker_count": 2,
                    "torch_threads_per_worker": threads,
                    "deterministic": True,
                },
            },
            "env": {
                "horizon": 25,
                "traffic_mode": "trigger",
                "traffic_density": 0.05,
                "random_traffic": False,
                "accident_prob": 0.0,
            },
            "model": {"args_path": "args.json", "checkpoint_path": "model.pth"},
            "sampler": {"name": "dpm10", "implementation": "diffusers"},
            "guidance": {"name": "none"},
            "scenarios": [{"name": "straight", "map": "S", "seed": 0}],
        }
    )
    return parse_evaluation_config(config)


def test_cpu_parallel_execution_applies_explicit_thread_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    applied: list[int] = []
    monkeypatch.setattr("eco_planner.evaluation.execution.os.cpu_count", lambda: 8)
    monkeypatch.setattr(torch, "set_num_threads", applied.append)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    report = configure_job_execution(_config(threads=4))

    assert report.mode == "parallel"
    assert report.worker_count == 2
    assert report.torch_threads_per_worker == 4
    assert applied == [4]


def test_cpu_parallel_execution_rejects_oversubscribed_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("eco_planner.evaluation.execution.os.cpu_count", lambda: 4)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(ValueError, match="thread budget"):
        configure_job_execution(_config(threads=4))


def test_cuda_parallel_execution_requires_one_visible_gpu_and_determinism(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch, "set_num_threads", lambda value: None)
    deterministic: list[bool] = []
    monkeypatch.setattr(torch, "use_deterministic_algorithms", deterministic.append)

    report = configure_job_execution(_config(accelerator="cuda"))

    assert report.resolved_accelerator == "cuda"
    assert deterministic == [True]

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    with pytest.raises(ValueError, match="exactly one visible CUDA GPU"):
        configure_job_execution(_config(accelerator="cuda"))


def test_parallel_execution_rejects_video() -> None:
    with pytest.raises(ValueError, match="video"):
        configure_job_execution(_config(video=True))
