from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from eco_planner.evaluation.config import RuntimeConfig
from eco_planner.evaluation.runtime import contracts
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
        generator: torch.Generator | tuple[torch.Generator, ...],
    ) -> PlannerInferenceResult:
        assert observation["ego_current_state"].device == self.anchor.device
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
    first_audit = first_result.audit_result()
    second_audit = second_result.audit_result()
    assert first_audit.prediction.dtype == np.float32
    assert np.array_equal(first_audit.initial_noise, second_audit.initial_noise)
    assert np.array_equal(first_audit.prediction, second_audit.prediction)
    assert np.shares_memory(first_result.ego_trajectory, first_audit.prediction)


@pytest.mark.parametrize("batch", [1, 2, 4, 8])
def test_runtime_batch_inference_matches_independent_serial_slots(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    official_model_config: OfficialDiffusionPlannerConfig,
    baseline_observation: dict[str, torch.Tensor],
    batch: int,
) -> None:
    planner = _TinyPlanner(official_model_config)
    monkeypatch.setattr(
        runtime,
        "load_official_diffusion_planner",
        lambda args_path, checkpoint_path, sampler_config, guidance_config: (
            planner,
            CheckpointLoadReport(276, 6_042_628),
        ),
    )
    fabric_runtime = runtime.create_fabric_inference_runtime(
        _config(accelerator="cpu", precision="32-true"),
        Dpm10SamplerConfig(),
        NoGuidanceConfig(),
        tmp_path,
        tmp_path,
    )
    batched_observation = {
        name: value.repeat((batch,) + (1,) * (value.ndim - 1))
        for name, value in baseline_observation.items()
    }
    generators = tuple(torch.Generator().manual_seed(100 + index) for index in range(batch))
    noise = torch.cat(
        [
            torch.randn(
                (1, 11, 80, 4),
                generator=torch.Generator().manual_seed(100 + index),
            )
            for index in range(batch)
        ]
    )

    batched = fabric_runtime.infer_batch(batched_observation, noise, generators)

    assert batched.ego_trajectories.shape == (batch, 80, 4)
    np.testing.assert_array_equal(batched.ego_trajectories, noise[:, 0].numpy())
    for index in range(batch):
        serial = fabric_runtime.infer(
            baseline_observation,
            torch.Generator().manual_seed(100 + index),
        )
        np.testing.assert_array_equal(batched.ego_trajectories[index], serial.ego_trajectory)


def test_cuda_execution_copy_does_not_wait_for_deferred_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the ego trajectory blocks the closed-loop path before full audit is requested."""

    synchronizations: list[str] = []

    class _Stream:
        def __init__(self, name: str) -> None:
            self.name = name
            self.waited_for: _Stream | None = None

        def wait_stream(self, stream: _Stream) -> None:
            self.waited_for = stream

        def synchronize(self) -> None:
            synchronizations.append(self.name)

    current = _Stream("current")
    transfer = _Stream("audit")
    real_empty = torch.empty

    def _empty(*args: object, **kwargs: object) -> torch.Tensor:
        kwargs.pop("pin_memory", None)
        return real_empty(*args, **kwargs)

    @contextmanager
    def _stream_context(stream: _Stream):
        yield stream

    monkeypatch.setattr(torch, "empty", _empty)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: current)
    monkeypatch.setattr(torch.cuda, "Stream", lambda *, device: transfer)
    monkeypatch.setattr(torch.cuda, "stream", _stream_context)
    device = torch.device("cuda")

    deferred = contracts.defer_host_tensors({"audit": (torch.ones(4), torch.float32)}, device)
    execution = contracts.copy_execution_trajectory(torch.ones((1, 1, 2, 4)), device)

    assert transfer.waited_for is current
    assert synchronizations == ["current"]
    np.testing.assert_array_equal(execution.ego_trajectory, np.ones((1, 2, 4), dtype=np.float32))

    audit = deferred.resolve()

    assert synchronizations == ["current", "audit"]
    torch.testing.assert_close(audit["audit"], torch.ones(4))
