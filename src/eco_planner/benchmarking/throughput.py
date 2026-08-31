"""Planner and process-isolated MetaDrive throughput benchmarks."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from eco_planner.envs import (
    PlannerObservationSpec,
    VectorEnvScenario,
    VectorEnvStep,
    VectorMetaDriveEnv,
    collate_observations,
)
from eco_planner.envs.array_types import SingleObservation
from eco_planner.evaluation.config import EvaluationJobConfig, parse_evaluation_config
from eco_planner.evaluation.runtime import (
    FabricInferenceRuntime,
    create_fabric_inference_runtime,
)
from eco_planner.models import OfficialDiffusionPlannerConfig
from eco_planner.runtime_resources import require_resource_profile

from .config import (
    ScalingBenchmarkConfig,
    benchmark_provenance,
    host_resource_provenance,
    measurement,
    split_benchmark_config,
    write_benchmark_artifacts,
)


def run(config: DictConfig) -> None:
    evaluation_config, benchmark = split_benchmark_config(config, ScalingBenchmarkConfig)
    parsed = parse_evaluation_config(evaluation_config)
    require_resource_profile(parsed.resources)
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    env_config = {**parsed.env, "map": parsed.scenarios[0].map}
    runtime = create_fabric_inference_runtime(
        parsed.runtime,
        parsed.sampler,
        parsed.guidance,
        Path(parsed.model.args_path).resolve(),
        Path(parsed.model.checkpoint_path).resolve(),
    )
    observation = _representative_observation(runtime, parsed, env_config)
    report: dict[str, object] = {
        "provenance": _provenance(parsed, runtime, benchmark),
        "planner": benchmark_planner_batch_scaling(
            runtime,
            observation,
            benchmark=benchmark,
        ),
        "environment": benchmark_vector_environment_scaling(
            env_config,
            mode=parsed.evaluation.mode,
            model_config=runtime.planner_config,
            map_query_radius_m=parsed.map_query_radius_m,
            history_warmup_steps=parsed.evaluation.history_warmup_steps,
            benchmark=benchmark,
        ),
    }
    write_benchmark_artifacts(output_dir, config, "throughput.json", report)
    print(json.dumps(report, indent=2))


def benchmark_planner_batch_scaling(
    runtime: FabricInferenceRuntime,
    observation: SingleObservation,
    *,
    benchmark: ScalingBenchmarkConfig,
) -> list[dict[str, object]]:
    """Measure CPU batch input through the synchronous execution-trajectory copy."""

    results: list[dict[str, object]] = []
    for batch_size in benchmark.batch_sizes:
        batched_observation = collate_observations([observation] * batch_size)
        generators = tuple(
            torch.Generator(device=runtime.device).manual_seed(runtime.report.seed + index)
            for index in range(batch_size)
        )
        for _ in range(benchmark.warmup_cycles):
            runtime.infer_batch(
                batched_observation,
                _standard_normal_noise(runtime, generators),
                generators,
            ).audit_result()

        batch_wall_samples: list[float] = []
        samples_per_second: list[float] = []
        h2d_samples: list[float] = []
        execution_samples: list[float] = []
        d2h_samples: list[float] = []
        memory_samples: list[float] = []
        for _ in range(benchmark.repeats):
            if runtime.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(runtime.device)
            elapsed = 0.0
            h2d = 0.0
            execution = 0.0
            d2h = 0.0
            for _ in range(benchmark.measured_cycles):
                started = perf_counter()
                decision = runtime.infer_batch(
                    batched_observation,
                    _standard_normal_noise(runtime, generators),
                    generators,
                    profile=True,
                )
                elapsed += perf_counter() - started
                timing = decision.timing
                if timing is None:
                    raise RuntimeError("profiled planner decision did not return timing")
                h2d += timing.host_to_device_s
                execution += timing.execution_s
                d2h += timing.execution_to_host_s
                decision.audit_result()
            batch_wall_samples.append(elapsed / benchmark.measured_cycles)
            samples_per_second.append(batch_size * benchmark.measured_cycles / elapsed)
            h2d_samples.append(h2d / benchmark.measured_cycles)
            execution_samples.append(execution / benchmark.measured_cycles)
            d2h_samples.append(d2h / benchmark.measured_cycles)
            if runtime.device.type == "cuda":
                memory_samples.append(float(torch.cuda.max_memory_allocated(runtime.device)))
        results.append(
            {
                "batch_size": batch_size,
                "batch_wall_s": measurement(batch_wall_samples),
                "samples_per_s": measurement(samples_per_second),
                "host_to_device_s": measurement(h2d_samples),
                "execution_s": measurement(execution_samples),
                "execution_to_host_s": measurement(d2h_samples),
                "peak_gpu_memory_bytes": measurement(memory_samples) if memory_samples else None,
            }
        )
    return results


def benchmark_vector_environment_scaling(
    env_config: Mapping[str, object],
    *,
    mode: str,
    model_config: OfficialDiffusionPlannerConfig,
    map_query_radius_m: float,
    history_warmup_steps: int,
    benchmark: ScalingBenchmarkConfig,
) -> list[dict[str, object]]:
    """Measure vector step wall time and per-step worker/transport service time."""

    if mode not in {"traffic", "no_traffic"}:
        raise ValueError("mode must be 'traffic' or 'no_traffic'")
    trajectory = _stationary_trajectory(model_config.future_len)
    results: list[dict[str, object]] = []
    for worker_count in benchmark.worker_counts:
        steps_per_second: list[float] = []
        batch_step_wall_samples: list[float] = []
        worker_environment_samples: list[float] = []
        worker_observation_samples: list[float] = []
        observation_service_rate_samples: list[float] = []
        transport_sync_samples: list[float] = []
        imbalance_samples: list[float] = []
        configs = tuple(dict(env_config) for _ in range(worker_count))
        scenarios = tuple(
            VectorEnvScenario(name=f"benchmark-slot-{slot}", map=str(env_config["map"]), seed=slot)
            for slot in range(worker_count)
        )
        for _ in range(benchmark.repeats):
            with VectorMetaDriveEnv(
                configs,
                mode=mode,  # type: ignore[arg-type]
                observation_spec=PlannerObservationSpec.from_planner_config(model_config),
                map_query_radius_m=map_query_radius_m,
                history_warmup_steps=history_warmup_steps,
                scenarios=scenarios,
            ) as environments:
                environments.reset(scenarios)
                for _ in range(benchmark.warmup_cycles):
                    _require_active_steps(environments.step((trajectory,) * worker_count))
                elapsed = 0.0
                worker_environment = 0.0
                worker_observation = 0.0
                transport_sync = 0.0
                imbalance = 0.0
                for _ in range(benchmark.measured_cycles):
                    started = perf_counter()
                    steps = environments.step((trajectory,) * worker_count)
                    batch_wall = perf_counter() - started
                    elapsed += batch_wall
                    _require_active_steps(steps)
                    busy = [step.timing.environment_s + step.timing.observation_s for step in steps]
                    worker_environment += sum(step.timing.environment_s for step in steps)
                    worker_observation += sum(step.timing.observation_s for step in steps)
                    transport_sync += max(0.0, batch_wall - max(busy))
                    imbalance += sum(max(busy) - value for value in busy)
                total_steps = worker_count * benchmark.measured_cycles
                steps_per_second.append(total_steps / elapsed)
                batch_step_wall_samples.append(elapsed / benchmark.measured_cycles)
                worker_environment_samples.append(worker_environment / total_steps)
                worker_observation_samples.append(worker_observation / total_steps)
                observation_service_rate_samples.append(total_steps / worker_observation)
                transport_sync_samples.append(transport_sync / total_steps)
                imbalance_samples.append(imbalance / total_steps)
        results.append(
            {
                "worker_count": worker_count,
                "env_steps_per_s": measurement(steps_per_second),
                "batch_step_wall_s": measurement(batch_step_wall_samples),
                "worker_environment_s_per_step": measurement(worker_environment_samples),
                "worker_observation_s_per_step": measurement(worker_observation_samples),
                "worker_observation_builds_per_service_s": measurement(
                    observation_service_rate_samples
                ),
                "transport_sync_s_per_step": measurement(transport_sync_samples),
                "worker_imbalance_s_per_step": measurement(imbalance_samples),
            }
        )
    return results


def _standard_normal_noise(
    runtime: FabricInferenceRuntime, generators: Sequence[torch.Generator]
) -> torch.Tensor:
    config = runtime.planner_config
    shape = (1, 1 + config.predicted_neighbor_num, config.future_len, 4)
    return torch.cat(
        [
            torch.randn(shape, dtype=torch.float32, device=runtime.device, generator=generator)
            for generator in generators
        ]
    )


def _stationary_trajectory(future_len: int) -> np.ndarray:
    trajectory = np.zeros((future_len, 4), dtype=np.float32)
    trajectory[:, 2] = 1.0
    return trajectory


def _require_active_steps(steps: Sequence[VectorEnvStep]) -> None:
    for step in steps:
        if step.terminated or step.truncated:
            raise RuntimeError("benchmark environment ended before measurement completed")


def _representative_observation(
    runtime: FabricInferenceRuntime,
    config: EvaluationJobConfig,
    env_config: dict[str, object],
) -> SingleObservation:
    with VectorMetaDriveEnv(
        (env_config,),
        mode=config.evaluation.mode,
        observation_spec=PlannerObservationSpec.from_planner_config(runtime.planner_config),
        map_query_radius_m=config.map_query_radius_m,
        history_warmup_steps=config.evaluation.history_warmup_steps,
        scenarios=(VectorEnvScenario("benchmark-observation", str(env_config["map"]), 0),),
    ) as environments:
        reset = environments.reset(
            (VectorEnvScenario("benchmark-observation", str(env_config["map"]), 0),)
        )[0]
    return reset.observation


def _provenance(
    config: EvaluationJobConfig,
    runtime: FabricInferenceRuntime,
    benchmark: ScalingBenchmarkConfig,
) -> dict[str, object]:
    return {
        **benchmark_provenance(benchmark),
        **host_resource_provenance(),
        "runtime": asdict(runtime.report),
        "execution": config.evaluation.execution.model_dump(mode="json"),
        "sampler": asdict(runtime.sampler_report),
        "guidance": asdict(config.guidance),
        "traffic": {
            "mode": config.evaluation.mode,
            "traffic_mode": config.env.get("traffic_mode"),
            "traffic_density": config.env.get("traffic_density"),
        },
        "seeds": {
            "runtime": config.runtime.seed,
            "scenario": config.scenarios[0].seed,
            "planner_noise": [
                config.runtime.seed + index for index in range(max(benchmark.batch_sizes))
            ],
            "policy_action": None,
        },
        "scenarios": [scenario.model_dump(mode="json") for scenario in config.scenarios],
        "render_enabled": config.video.enabled,
    }
