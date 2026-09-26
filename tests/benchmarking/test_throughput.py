"""Throughput's production planning interface and evaluation timing adapter."""

from eco_planner.benchmarking.config import ScalingBenchmarkConfig
from eco_planner.benchmarking.throughput import benchmark_planner_batch_scaling
from tests.characterization.test_diffusion_inference import build_diffusion_case


def test_throughput_consumes_planning_runtime_and_profiled_host_adapter():
    runtime, observation, _, _ = build_diffusion_case("base_ddim")
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
