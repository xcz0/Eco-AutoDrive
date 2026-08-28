"""Fixed-slot policy rollout and PPO update throughput benchmark."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from eco_planner.evaluation.config import ScenarioConfig
from eco_planner.rl.config import TrainingJobConfig, parse_training_config
from eco_planner.rl.optimization import PPOConfig, PPOUpdater
from eco_planner.rl.rollout import (
    FabricRolloutRuntime,
    RolloutEpisode,
    VectorRolloutCollector,
    VectorRolloutRoundTiming,
    collect_rollout_episode,
    create_fabric_rollout_runtime,
)

from .common import (
    RolloutBenchmarkConfig,
    benchmark_provenance,
    host_resource_provenance,
    measurement,
    split_benchmark_config,
    write_benchmark_artifacts,
)


def run(config: DictConfig) -> None:
    training_config, benchmark = split_benchmark_config(config, RolloutBenchmarkConfig)
    parsed = parse_training_config(training_config)
    report: dict[str, object] = {
        "provenance": {
            **benchmark_provenance(benchmark),
            **host_resource_provenance(),
            "runtime": parsed.runtime.model_dump(mode="json"),
            "sampler": asdict(parsed.sampler),
            "guidance": asdict(parsed.guidance),
            "policy": parsed.policy.model_dump(mode="json"),
            "base_ppo_config": parsed.ppo.model_dump(mode="json"),
            "traffic": {
                "mode": benchmark.mode,
                "density": parsed.env.get("traffic_density"),
            },
            "scenarios": [scenario.model_dump(mode="json") for scenario in parsed.scenarios],
            "seeds": {
                "scenario_base": benchmark.scenario_seed_base,
                "noise_base": benchmark.noise_seed_base,
                "policy_action_base": benchmark.policy_action_seed_base,
            },
            "render_enabled": False,
        },
        "rollout": [
            _measure_batch_size(parsed, collector_mode, batch_size, benchmark)
            for collector_mode in benchmark.collector_modes
            for batch_size in benchmark.batch_sizes
        ],
    }
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    write_benchmark_artifacts(output_dir, config, "rollout_throughput.json", report)
    print(json.dumps(report, indent=2))


def _measure_batch_size(
    config: TrainingJobConfig,
    collector_mode: str,
    batch_size: int,
    benchmark: RolloutBenchmarkConfig,
) -> dict[str, object]:
    sample_count = batch_size * benchmark.transitions_per_slot
    update_count = benchmark.warmup_updates + benchmark.measured_updates
    collection_samples: list[float] = []
    update_samples: list[float] = []
    timing_samples: list[tuple[VectorRolloutRoundTiming, ...]] = []
    cold_collection_samples: list[float] = []
    cold_update_samples: list[float] = []
    cold_timing_samples: list[tuple[VectorRolloutRoundTiming, ...]] = []
    worker_pool_startup_samples: list[float] = []

    for _ in range(benchmark.repeats):
        ppo_config = _effective_ppo_config(config.ppo, benchmark, sample_count, update_count)
        runtime = create_fabric_rollout_runtime(
            config.runtime,
            config.sampler,
            config.guidance,
            config.policy,
            Path(to_absolute_path(config.model.args_path)),
            Path(to_absolute_path(config.model.checkpoint_path)),
            policy_action_seed=benchmark.policy_action_seed_base,
        )
        updater = PPOUpdater(runtime.policy, ppo_config)
        specs = tuple(
            config.scenarios[index % len(config.scenarios)].model_copy(
                update={
                    "name": f"benchmark-slot-{index}",
                    "seed": benchmark.scenario_seed_base + index,
                }
            )
            for index in range(batch_size)
        )
        noise_seeds = tuple(benchmark.noise_seed_base + index for index in range(batch_size))
        policy_seeds = tuple(
            benchmark.policy_action_seed_base + index for index in range(batch_size)
        )
        diffusion_generators = tuple(runtime.new_noise_generator(seed) for seed in noise_seeds)
        policy_generators = tuple(runtime.new_policy_generator(seed) for seed in policy_seeds)
        vector_collector: VectorRolloutCollector | None = None
        for update_index in range(update_count):
            timings: list[VectorRolloutRoundTiming] = []
            started = perf_counter()
            if collector_mode == "vector":
                if vector_collector is None:
                    startup_started = perf_counter()
                    vector_collector = VectorRolloutCollector(
                        specs,
                        runtime,
                        config.env,
                        mode=benchmark.mode,
                        map_query_radius_m=config.map_query_radius_m,
                        history_warmup_steps=benchmark.history_warmup_steps,
                        torch_threads_per_worker=config.resources.torch_threads_per_worker,
                    )
                    worker_pool_startup_samples.append(perf_counter() - startup_started)
                slots = vector_collector.collect(
                    transitions_per_slot=benchmark.transitions_per_slot,
                    stopped_speed_threshold_mps=config.training.stopped_speed_threshold_mps,
                    diffusion_generators=diffusion_generators,
                    policy_generators=policy_generators,
                    noise_seeds=noise_seeds,
                    policy_action_seeds=policy_seeds,
                    timings=timings,
                )
            elif collector_mode == "serial":
                slots = _collect_serial_slots(
                    specs,
                    runtime,
                    config,
                    benchmark,
                    diffusion_generators,
                    policy_generators,
                    noise_seeds,
                    policy_seeds,
                )
            else:
                raise ValueError(f"unknown rollout collector mode {collector_mode!r}")
            collection_s = perf_counter() - started
            update_started = perf_counter()
            updater.update(tuple(episode for slot in slots for episode in slot))
            update_s = perf_counter() - update_started
            if update_index == 0:
                cold_collection_samples.append(collection_s)
                cold_update_samples.append(update_s)
                cold_timing_samples.append(tuple(timings) if collector_mode == "vector" else ())
            if update_index >= benchmark.warmup_updates:
                collection_samples.append(collection_s)
                update_samples.append(update_s)
                timing_samples.append(tuple(timings) if collector_mode == "vector" else ())
        if vector_collector is not None:
            vector_collector.close()

    result = _rollout_result(
        collector_mode,
        batch_size,
        sample_count,
        collection_samples,
        update_samples,
        timing_samples,
    )
    result["cold_start_update"] = _rollout_result(
        collector_mode,
        batch_size,
        sample_count,
        cold_collection_samples,
        cold_update_samples,
        cold_timing_samples,
    )
    result["worker_pool_startup_wall_s"] = (
        measurement(worker_pool_startup_samples) if worker_pool_startup_samples else None
    )
    return result


def _effective_ppo_config(
    base: PPOConfig,
    benchmark: RolloutBenchmarkConfig,
    sample_count: int,
    update_count: int,
) -> PPOConfig:
    optimizer_steps_per_update = benchmark.ppo_epochs * (
        sample_count // benchmark.ppo_minibatch_size
    )
    return base.model_copy(
        update={
            "epochs": benchmark.ppo_epochs,
            "batch_size": sample_count,
            "minibatch_size": benchmark.ppo_minibatch_size,
            "scheduler_total_optimizer_steps": update_count * optimizer_steps_per_update,
        }
    )


def _collect_serial_slots(
    specs: Sequence[ScenarioConfig],
    runtime: FabricRolloutRuntime,
    config: TrainingJobConfig,
    benchmark: RolloutBenchmarkConfig,
    diffusion_generators: Sequence[torch.Generator],
    policy_generators: Sequence[torch.Generator],
    noise_seeds: Sequence[int],
    policy_seeds: Sequence[int],
) -> tuple[tuple[RolloutEpisode, ...], ...]:
    slots: list[tuple[RolloutEpisode, ...]] = []
    for slot, spec in enumerate(specs):
        episodes: list[RolloutEpisode] = []
        collected = 0
        while collected < benchmark.transitions_per_slot:
            episode = collect_rollout_episode(
                spec,
                runtime,
                config.env,
                mode=benchmark.mode,
                map_query_radius_m=config.map_query_radius_m,
                history_warmup_steps=benchmark.history_warmup_steps,
                max_transitions=benchmark.transitions_per_slot - collected,
                stopped_speed_threshold_mps=config.training.stopped_speed_threshold_mps,
                diffusion_generator=diffusion_generators[slot],
                policy_generator=policy_generators[slot],
                noise_seed=noise_seeds[slot],
                policy_action_seed=policy_seeds[slot],
            )
            episodes.append(episode)
            collected += episode.transition_count
        slots.append(tuple(episodes))
    return tuple(slots)


def _rollout_result(
    collector_mode: str,
    batch_size: int,
    sample_count: int,
    collection_samples: Sequence[float],
    update_samples: Sequence[float],
    timing_samples: Sequence[Sequence[VectorRolloutRoundTiming]],
) -> dict[str, object]:
    if not (
        len(collection_samples) == len(update_samples) == len(timing_samples) and timing_samples
    ):
        raise ValueError("rollout benchmark samples must be non-empty and aligned")

    if collector_mode == "serial":
        return {
            "collector_mode": collector_mode,
            "batch_size": batch_size,
            **_common_rollout_measurements(sample_count, collection_samples, update_samples),
            "planner_decision_wall_s": None,
            "planner_bootstrap_wall_s": None,
            "planner_decision_phases": None,
            "planner_bootstrap_phases": None,
            "collate_wall_s": None,
            "audit_resolve_wall_s": None,
            "audit_transfer_accelerator_s": None,
            "environment_wall_s": None,
            "collection_unattributed_wall_s": None,
            "worker_busy_s_per_transition": None,
            "transport_sync_s_per_transition": None,
            "worker_imbalance_s_per_transition": None,
            "decision_batch_fill_ratio": None,
            "bootstrap_batch_fill_ratio": None,
            "policy_planner_batch_wall_s": None,
            "policy_planner_samples_per_s": None,
        }

    decision_wall: list[float] = []
    bootstrap_wall: list[float] = []
    collate_wall: list[float] = []
    audit_resolve_wall: list[float] = []
    audit_transfer_accelerator: list[float] = []
    environment_wall: list[float] = []
    unattributed: list[float] = []
    worker_busy_per_transition: list[float] = []
    transport_sync_per_transition: list[float] = []
    worker_imbalance_per_transition: list[float] = []
    decision_fill: list[float] = []
    bootstrap_fill: list[float] = []
    policy_planner_batch_wall: list[float] = []
    policy_planner_samples_per_second: list[float] = []
    for collection_s, timings in zip(collection_samples, timing_samples, strict=True):
        decisions = [item for item in timings if item.phase == "decision"]
        bootstraps = [item for item in timings if item.phase == "bootstrap"]
        decision_s = sum(item.planner_wall_s for item in decisions)
        bootstrap_s = sum(item.planner_wall_s for item in bootstraps)
        collate_s = sum(item.collate_wall_s for item in timings)
        audit_resolve_s = sum(item.audit_resolve_wall_s for item in decisions)
        audit_transfer_s = sum(item.audit_transfer_accelerator_s for item in decisions)
        env_s = sum(item.environment_wall_s for item in decisions)
        decision_wall.append(decision_s)
        policy_planner_batch_wall.append(decision_s / len(decisions))
        policy_planner_samples_per_second.append(
            sum(item.active_slots for item in decisions) / decision_s
        )
        if bootstraps:
            bootstrap_wall.append(bootstrap_s)
            bootstrap_fill.append(_fill_ratio(bootstraps))
        collate_wall.append(collate_s)
        audit_resolve_wall.append(audit_resolve_s)
        audit_transfer_accelerator.append(audit_transfer_s)
        environment_wall.append(env_s)
        unattributed.append(
            collection_s
            - decision_s
            - bootstrap_s
            - collate_s
            - audit_resolve_s
            - env_s
        )
        worker_busy_per_transition.append(
            sum(item.worker_busy_s for item in decisions) / sample_count
        )
        transport_sync_per_transition.append(
            sum(item.transport_sync_s for item in decisions) / sample_count
        )
        worker_imbalance_per_transition.append(
            sum(item.worker_imbalance_s for item in decisions) / sample_count
        )
        decision_fill.append(_fill_ratio(decisions))

    return {
        "collector_mode": collector_mode,
        "batch_size": batch_size,
        **_common_rollout_measurements(sample_count, collection_samples, update_samples),
        "planner_decision_wall_s": measurement(decision_wall),
        "planner_bootstrap_wall_s": measurement(bootstrap_wall) if bootstrap_wall else None,
        "planner_decision_phases": _planner_phase_measurements(timing_samples, "decision"),
        "planner_bootstrap_phases": (
            _planner_phase_measurements(timing_samples, "bootstrap")
            if bootstrap_wall
            else None
        ),
        "collate_wall_s": measurement(collate_wall),
        "audit_resolve_wall_s": measurement(audit_resolve_wall),
        "audit_transfer_accelerator_s": measurement(audit_transfer_accelerator),
        "environment_wall_s": measurement(environment_wall),
        "collection_unattributed_wall_s": measurement(unattributed),
        "worker_busy_s_per_transition": measurement(worker_busy_per_transition),
        "transport_sync_s_per_transition": measurement(transport_sync_per_transition),
        "worker_imbalance_s_per_transition": measurement(worker_imbalance_per_transition),
        "decision_batch_fill_ratio": measurement(decision_fill),
        "bootstrap_batch_fill_ratio": measurement(bootstrap_fill) if bootstrap_fill else None,
        "policy_planner_batch_wall_s": measurement(policy_planner_batch_wall),
        "policy_planner_samples_per_s": measurement(policy_planner_samples_per_second),
    }


def _common_rollout_measurements(
    sample_count: int,
    collection_samples: Sequence[float],
    update_samples: Sequence[float],
) -> dict[str, object]:
    return {
        "rollout_transitions_per_s": measurement(
            [sample_count / value for value in collection_samples]
        ),
        "collection_wall_s": measurement(collection_samples),
        "ppo_update_wall_s": measurement(update_samples),
        "end_to_end_update_wall_s": measurement(
            [
                collection + update
                for collection, update in zip(collection_samples, update_samples, strict=True)
            ]
        ),
    }


def _fill_ratio(timings: Sequence[VectorRolloutRoundTiming]) -> float:
    active = sum(item.active_slots for item in timings)
    capacity = sum(item.capacity for item in timings)
    if capacity <= 0:
        raise ValueError("rollout timing capacity must be positive")
    return active / capacity


_PLANNER_PHASE_FIELDS = (
    "host_to_device",
    "diffusion_noise",
    "prepare_policy_guidance",
    "policy_forward",
    "action_sampling",
    "complete_policy_guidance",
    "guidance_action_check",
    "execution_to_host",
)


def _planner_phase_measurements(
    timing_samples: Sequence[Sequence[VectorRolloutRoundTiming]],
    phase: str,
) -> dict[str, object]:
    matching = [
        [item.planner_timing for item in timings if item.phase == phase]
        for timings in timing_samples
    ]
    if any(not timings for timings in matching):
        raise ValueError(f"each rollout update must contain a {phase} planner timing")
    result: dict[str, object] = {}
    for field in _PLANNER_PHASE_FIELDS:
        values = [[getattr(timing, field) for timing in timings] for timings in matching]
        present = [[value for value in update if value is not None] for update in values]
        if all(not update for update in present):
            result[field] = None
            continue
        if any(len(update) != len(values[index]) for index, update in enumerate(present)):
            raise ValueError(f"planner phase field {field!r} is only partially populated")
        result[field] = {
            "host_call_wall_s": measurement(
                [sum(value.host_call_wall_s for value in update) for update in present]
            ),
            "accelerator_s": measurement(
                [sum(value.accelerator_s for value in update) for update in present]
            ),
        }
    result["profile_sync_wait_wall_s"] = measurement(
        [
            sum(timing.profile_sync_wait_wall_s for timing in timings)
            for timings in matching
        ]
    )
    return result
