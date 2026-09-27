"""Throughput's production planning interface and preserved measurement window."""

from types import SimpleNamespace

import pytest
import torch

import benchmarks.throughput as throughput
from benchmarks.config import ScalingBenchmarkConfig
from benchmarks.throughput import benchmark_planner_batch_scaling
from eco_planner.evaluation.inference import DiffusionEvaluationAgent
from eco_planner.runtime.host_transfer import HostTransfer
from tests.characterization.test_diffusion_inference import build_diffusion_case


@pytest.mark.parametrize("case", ["base_ddim", "fixed"])
def test_throughput_consumes_planning_runtime_and_profiled_host_adapter(case):
    runtime, observation, _, _ = build_diffusion_case(case)
    results = benchmark_planner_batch_scaling(
        runtime,
        observation[0],
        benchmark=ScalingBenchmarkConfig(
            kind="throughput",
            batch_sizes=(1, 2),
            worker_counts=(1,),
            warmup_cycles=1,
            measured_cycles=1,
            repeats=1,
        ),
    )
    assert [result["batch_size"] for result in results] == [1, 2]
    for result in results:
        assert result["peak_gpu_memory_bytes"] is None
        for name in ("host_to_device_s", "execution_s", "execution_to_host_s"):
            assert result[name]["median"] > 0


@pytest.mark.parametrize("case", ["base_stochastic", "fixed"])
@pytest.mark.parametrize("profile", [False, True])
def test_direct_planning_benchmark_preserves_evaluation_payload_and_rng(case, profile):
    runtime, observation, generators, _ = build_diffusion_case(case)
    states = [generator.get_state() for generator in generators]
    expected = DiffusionEvaluationAgent(runtime).infer_batch(
        observation, runtime.sample_noise(generators), generators, profile=profile
    )
    expected_states = [generator.get_state() for generator in generators]
    for generator, state in zip(generators, states, strict=True):
        generator.set_state(state)
    actual = throughput._infer_batch(
        runtime,
        HostTransfer(runtime.device),
        observation,
        runtime.sample_noise(generators),
        generators,
        profile=profile,
    )
    torch.testing.assert_close(dict(actual.audit_result()), dict(expected.audit_result()))
    torch.testing.assert_close(actual.ego_trajectories, expected.ego_trajectories)
    assert (actual.timing is not None) == profile
    for generator, state in zip(generators, expected_states, strict=True):
        assert torch.equal(generator.get_state(), state)


def test_throughput_wall_includes_noise_and_execution_copy_but_excludes_audit(monkeypatch):
    runtime, observation, _, _ = build_diffusion_case("base_ddim")
    clock = [0.0]
    profiles = []
    audits = []
    monkeypatch.setattr(throughput, "perf_counter", lambda: clock[0])

    def sample_noise(generators):
        clock[0] += 1.0
        return None

    def infer(*args, profile=False):
        profiles.append(profile)
        clock[0] += 2.0  # Model call through synchronous execution copy.

        def audit_result():
            audits.append(profile)
            clock[0] += 5.0

        return SimpleNamespace(
            audit_result=audit_result,
            timing=SimpleNamespace(host_to_device_s=0.2, execution_s=1.5, execution_to_host_s=0.3),
        )

    monkeypatch.setattr(runtime, "sample_noise", sample_noise)
    monkeypatch.setattr(throughput, "_infer_batch", infer)
    result = benchmark_planner_batch_scaling(
        runtime,
        observation[0],
        benchmark=ScalingBenchmarkConfig(
            kind="throughput",
            batch_sizes=(1,),
            worker_counts=(1,),
            warmup_cycles=1,
            measured_cycles=2,
            repeats=2,
        ),
    )[0]
    assert profiles == audits == [False, True, True, True, True]
    assert result["batch_wall_s"]["samples"] == [3.0, 3.0]
    assert result["samples_per_s"]["samples"] == [1.0 / 3.0, 1.0 / 3.0]
    assert result["execution_to_host_s"]["samples"] == [0.3, 0.3]
