from __future__ import annotations

import pytest
import torch

from eco_planner.rl.optimization import PPOConfig
from eco_planner.rl.rollout.collector import VectorRolloutRoundTiming
from eco_planner.rl.rollout.runtime import (
    RolloutPlannerPhaseTiming,
    RolloutPlannerTiming,
    _finish_profile,
    _profile_call,
)
from scripts.benchmarking.common import RolloutBenchmarkConfig
from scripts.benchmarking.rollout import _effective_ppo_config, _rollout_result


def _base_ppo_config() -> PPOConfig:
    return PPOConfig(
        name="test",
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coefficient=0.5,
        entropy_coefficient=0.01,
        learning_rate=0.00025,
        adam_epsilon=1e-5,
        weight_decay=0.0,
        max_gradient_norm=0.5,
        epochs=4,
        batch_size=32,
        minibatch_size=16,
        minibatch_seed=0,
        scheduler_total_optimizer_steps=32,
        scheduler_minimum_learning_rate=0.0,
    )


def _benchmark_config() -> RolloutBenchmarkConfig:
    return RolloutBenchmarkConfig(
        kind="rollout",
        batch_sizes=(2,),
        collector_modes=("vector",),
        mode="no_traffic",
        history_warmup_steps=0,
        ppo_epochs=4,
        ppo_minibatch_size=16,
        scenario_seed_base=0,
        noise_seed_base=0,
        policy_action_seed_base=10_000,
        warmup_updates=1,
        measured_updates=3,
        transitions_per_slot=16,
        repeats=1,
    )


def _phase(host_s: float, accelerator_s: float) -> RolloutPlannerPhaseTiming:
    return RolloutPlannerPhaseTiming(host_call_wall_s=host_s, accelerator_s=accelerator_s)


def _planner_timing(phase: str) -> RolloutPlannerTiming:
    decision = phase == "decision"
    return RolloutPlannerTiming(
        phase=phase,  # type: ignore[arg-type]
        host_to_device=_phase(0.01, 0.008),
        diffusion_noise=_phase(0.02, 0.015),
        prepare_policy_guidance=_phase(0.3, 0.25),
        policy_forward=_phase(0.1, 0.08),
        action_sampling=_phase(0.03, 0.02) if decision else None,
        complete_policy_guidance=_phase(0.4, 0.35) if decision else None,
        guidance_action_check=_phase(0.01, 0.005) if decision else None,
        execution_to_host=_phase(0.02, 0.01) if decision else None,
        profile_sync_wait_wall_s=0.01,
    )


def test_effective_rollout_ppo_config_covers_every_optimizer_step() -> None:
    config = _effective_ppo_config(_base_ppo_config(), _benchmark_config(), 32, 4)

    assert config.batch_size == 32
    assert config.minibatch_size == 16
    assert config.epochs == 4
    assert config.optimizer_steps_per_update == 8
    assert config.scheduler_total_optimizer_steps == 32


@pytest.mark.smoke
def test_rollout_result_keeps_profiled_boundaries_separate() -> None:
    decision = VectorRolloutRoundTiming(
        phase="decision",
        active_slots=2,
        capacity=2,
        collate_wall_s=0.1,
        planner_wall_s=1.0,
        planner_timing=_planner_timing("decision"),
        environment_wall_s=0.2,
        audit_resolve_wall_s=0.3,
        audit_transfer_accelerator_s=0.25,
        worker_busy_s=0.2,
        transport_sync_s=0.1,
        worker_imbalance_s=0.0,
    )
    bootstrap = VectorRolloutRoundTiming(
        phase="bootstrap",
        active_slots=2,
        capacity=2,
        collate_wall_s=0.05,
        planner_wall_s=0.5,
        planner_timing=_planner_timing("bootstrap"),
        environment_wall_s=0.0,
        audit_resolve_wall_s=0.0,
        audit_transfer_accelerator_s=0.0,
        worker_busy_s=0.0,
        transport_sync_s=0.0,
        worker_imbalance_s=0.0,
    )

    result = _rollout_result("vector", 2, 32, [2.3], [0.4], [(decision, bootstrap)])

    assert result["planner_decision_wall_s"]["samples"] == [1.0]  # type: ignore[index]
    assert result["planner_bootstrap_wall_s"]["samples"] == [0.5]  # type: ignore[index]
    assert result["collate_wall_s"]["samples"] == pytest.approx([0.15])  # type: ignore[index]
    assert result["audit_resolve_wall_s"]["samples"] == [0.3]  # type: ignore[index]
    assert result["audit_transfer_accelerator_s"]["samples"] == [0.25]  # type: ignore[index]
    assert result["collection_unattributed_wall_s"]["samples"] == pytest.approx(  # type: ignore[index]
        [0.15]
    )
    assert "collection_overhead_s" not in result
    decision_phases = result["planner_decision_phases"]
    bootstrap_phases = result["planner_bootstrap_phases"]
    assert decision_phases["complete_policy_guidance"]["accelerator_s"]["samples"] == [  # type: ignore[index]
        0.35
    ]
    assert bootstrap_phases["complete_policy_guidance"] is None  # type: ignore[index]


def test_disabled_profile_does_not_create_cuda_events_or_synchronize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda")
    monkeypatch.setattr(
        torch.cuda,
        "Event",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Event created")),
    )
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("stream synchronized")),
    )

    result, timing = _profile_call(device, False, lambda: 7)

    assert result == 7
    assert timing is None
    assert _finish_profile(device, False) == 0.0


@pytest.mark.gpu
def test_cuda_profile_attributes_async_work_before_return() -> None:
    device = torch.device("cuda")

    _, pending = _profile_call(device, True, lambda: torch.cuda._sleep(20_000_000))
    sync_wait_s = _finish_profile(device, True)

    assert pending is not None
    timing = pending.resolve()
    assert timing.accelerator_s > 0.0
    assert sync_wait_s > 0.0
    assert timing.host_call_wall_s < timing.accelerator_s
