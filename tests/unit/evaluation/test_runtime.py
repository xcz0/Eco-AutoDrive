from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

from eco_planner.evaluation import runtime
from eco_planner.models.checkpoint import CheckpointLoadReport


def _config(**overrides: object) -> DictConfig:
    values: dict[str, object] = {
        "accelerator": "auto",
        "devices": 1,
        "precision": "auto",
        "seed": 7,
    }
    values.update(overrides)
    return OmegaConf.create(values)


def test_auto_runtime_resolves_cpu_fp32(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    settings = runtime.resolve_runtime_settings(_config())

    assert settings.resolved_accelerator == "cpu"
    assert settings.resolved_precision == "32-true"


@pytest.mark.parametrize(
    ("bf16_supported", "expected_precision"),
    [(True, "bf16-mixed"), (False, "16-mixed")],
)
def test_auto_runtime_resolves_cuda_precision_by_capability(
    monkeypatch: pytest.MonkeyPatch,
    bf16_supported: bool,
    expected_precision: str,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: bf16_supported)

    settings = runtime.resolve_runtime_settings(_config())

    assert settings.resolved_accelerator == "cuda"
    assert settings.resolved_precision == expected_precision


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        ({"devices": 2}, ValueError, "devices"),
        ({"seed": -1}, ValueError, "seed"),
        ({"accelerator": "mps"}, ValueError, "accelerator"),
        ({"precision": "64-true"}, ValueError, "precision"),
        ({"accelerator": "cpu", "precision": "16-mixed"}, ValueError, "CPU"),
    ],
)
def test_runtime_rejects_invalid_configuration(
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(error, match=message):
        runtime.resolve_runtime_settings(_config(**overrides))


def test_runtime_rejects_unavailable_explicit_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="unavailable"):
        runtime.resolve_runtime_settings(_config(accelerator="cuda"))


def test_runtime_rejects_unsupported_explicit_bf16(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)

    with pytest.raises(RuntimeError, match="does not support BF16"):
        runtime.resolve_runtime_settings(_config(accelerator="cuda", precision="bf16-mixed"))


class _TinyPlanner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(predicted_neighbor_num=1, future_len=2)
        self.register_parameter("anchor", nn.Parameter(torch.ones(()), requires_grad=False))
        self.eval()

    def forward(self, observation: dict[str, torch.Tensor], noise: torch.Tensor) -> torch.Tensor:
        assert observation["value"].device == self.anchor.device
        return noise * self.anchor


def test_cpu_fabric_runtime_assembles_model_and_replays_noise(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    planner = _TinyPlanner()
    checkpoint_report = CheckpointLoadReport(276, 6_042_628)
    monkeypatch.setattr(
        runtime,
        "load_official_diffusion_planner",
        lambda args_path, checkpoint_path: (planner, checkpoint_report),
    )
    config = _config(accelerator="cpu", precision="32-true", seed=11)

    fabric_runtime = runtime.create_fabric_inference_runtime(
        config,
        tmp_path,
        tmp_path,
    )
    first_generator = fabric_runtime.new_noise_generator()
    observation, first_noise, first_prediction = fabric_runtime.infer(
        {"value": torch.ones(1)}, first_generator
    )
    second_generator = fabric_runtime.new_noise_generator()
    _, second_noise, second_prediction = fabric_runtime.infer(
        {"value": torch.ones(1)}, second_generator
    )

    assert fabric_runtime.device == torch.device("cpu")
    assert fabric_runtime.report.resolved_precision == "32-true"
    assert fabric_runtime.report.world_size == 1
    assert observation["value"].device == fabric_runtime.device
    assert first_prediction.dtype == torch.float32
    assert torch.equal(first_noise, second_noise)
    assert torch.equal(first_prediction, second_prediction)
