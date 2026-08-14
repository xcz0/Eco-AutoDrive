from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from eco_planner.evaluation.config import RuntimeConfig
from eco_planner.evaluation.runtime import engine as runtime
from eco_planner.models import (
    CheckpointLoadReport,
    Dpm10SamplerConfig,
    NoGuidanceConfig,
    OfficialDiffusionPlannerConfig,
)
from eco_planner.models.planner import PlannerInferenceResult


def _config(**overrides: object) -> RuntimeConfig:
    values: dict[str, object] = {
        "accelerator": "auto",
        "devices": 1,
        "precision": "auto",
        "seed": 7,
    }
    values.update(overrides)
    return RuntimeConfig.model_validate(values)


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
    def __init__(self, config: OfficialDiffusionPlannerConfig) -> None:
        super().__init__()
        self.config = config
        self.register_parameter("anchor", nn.Parameter(torch.ones(()), requires_grad=False))
        self.eval()

    def forward(
        self,
        observation: dict[str, torch.Tensor],
        noise: torch.Tensor,
        generator: torch.Generator,
    ) -> PlannerInferenceResult:
        assert observation["ego_current_state"].device == self.anchor.device
        assert generator.device == self.anchor.device
        return PlannerInferenceResult(prediction=noise * self.anchor)


def test_cpu_fabric_runtime_assembles_model_and_replays_noise(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    official_model_config: OfficialDiffusionPlannerConfig,
    baseline_observation: dict[str, torch.Tensor],
) -> None:
    planner = _TinyPlanner(official_model_config)
    checkpoint_report = CheckpointLoadReport(276, 6_042_628)
    monkeypatch.setattr(
        runtime,
        "load_official_diffusion_planner",
        lambda args_path, checkpoint_path, sampler_config, guidance_config: (
            planner,
            checkpoint_report,
        ),
    )
    config = _config(accelerator="cpu", precision="32-true", seed=11)

    fabric_runtime = runtime.create_fabric_inference_runtime(
        config,
        Dpm10SamplerConfig(),
        NoGuidanceConfig(),
        tmp_path,
        tmp_path,
    )
    first_generator = fabric_runtime.new_noise_generator()
    first_result = fabric_runtime.infer(baseline_observation, first_generator)
    second_generator = fabric_runtime.new_noise_generator()
    second_result = fabric_runtime.infer(baseline_observation, second_generator)

    assert fabric_runtime.device == torch.device("cpu")
    assert fabric_runtime.report.resolved_precision == "32-true"
    assert fabric_runtime.report.world_size == 1
    assert first_result.prediction.dtype == np.float32
    assert np.array_equal(first_result.initial_noise, second_result.initial_noise)
    assert np.array_equal(first_result.prediction, second_result.prediction)
    assert np.shares_memory(first_result.ego_trajectory, first_result.prediction)
