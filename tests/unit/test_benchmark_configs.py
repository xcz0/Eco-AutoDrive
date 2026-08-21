from __future__ import annotations

from pathlib import Path

import pytest
from benchmarking.common import (
    RolloutBenchmarkConfig,
    ScalingBenchmarkConfig,
    parse_environment_job,
    split_benchmark_config,
)
from hydra import compose, initialize_config_dir
from pydantic import ValidationError

from eco_planner.evaluation.config import parse_evaluation_config
from eco_planner.rl.config import parse_training_config


def _compose(name: str):
    config_dir = Path(__file__).parents[2] / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        return compose(config_name=f"experiment/{name}")


def test_environment_benchmark_config_composes_and_parses() -> None:
    parsed = parse_environment_job(_compose("benchmark_environment"))

    assert parsed.benchmark.repeats == 5
    assert parsed.benchmark.map.startswith("SCS")
    assert parsed.env["horizon"] == 700


def test_throughput_benchmark_config_preserves_strict_evaluation_job() -> None:
    job, benchmark = split_benchmark_config(
        _compose("benchmark_throughput"), ScalingBenchmarkConfig
    )

    assert benchmark.batch_sizes == (1, 2, 4, 8, 16)
    parsed = parse_evaluation_config(job)
    assert parsed.evaluation.profile == "throughput"
    required_steps = (benchmark.warmup_cycles + benchmark.measured_cycles) * int(
        parsed.env["trajectory_execution_steps"]
    )
    assert parsed.env["horizon"] >= required_steps
    assert int(parsed.env["start_seed"]) == 0
    assert int(parsed.env["num_scenarios"]) >= max(benchmark.worker_counts)


def test_traffic_throughput_default_horizon_covers_all_cycles() -> None:
    job, benchmark = split_benchmark_config(
        _compose("benchmark_throughput_traffic"), ScalingBenchmarkConfig
    )
    parsed = parse_evaluation_config(job)
    required_steps = parsed.evaluation.history_warmup_steps + (
        benchmark.warmup_cycles + benchmark.measured_cycles
    ) * int(parsed.env["trajectory_execution_steps"])

    assert parsed.env["horizon"] >= required_steps


def test_rollout_benchmark_config_preserves_strict_training_job() -> None:
    job, benchmark = split_benchmark_config(_compose("benchmark_rollout"), RolloutBenchmarkConfig)

    assert benchmark.repeats == 3
    assert benchmark.collector_modes == ("serial", "vector")
    parsed = parse_training_config(job)
    assert parsed.training.replay_id == 0
    assert int(parsed.env["start_seed"]) == 0
    assert int(parsed.env["num_scenarios"]) >= max(benchmark.batch_sizes)


def test_rollout_benchmark_rejects_single_transition_ppo_batch() -> None:
    with pytest.raises(ValidationError, match="at least two transitions"):
        RolloutBenchmarkConfig.model_validate(
            {
                "batch_sizes": (1,),
                "collector_modes": ("serial", "vector"),
                "mode": "no_traffic",
                "history_warmup_steps": 0,
                "ppo_epochs": 1,
                "scenario_seed_base": 0,
                "noise_seed_base": 0,
                "policy_action_seed_base": 10_000,
                "warmup_updates": 1,
                "measured_updates": 1,
                "transitions_per_slot": 1,
                "repeats": 1,
            }
        )
