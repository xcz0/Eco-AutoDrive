from __future__ import annotations

from benchmarking.rollout import _rollout_result

from eco_planner.rl.collector import VectorRolloutRoundTiming


def _timing(
    phase: str,
    *,
    active_slots: int,
    capacity: int,
    planner: float,
    environment: float = 0.0,
    worker_busy: float = 0.0,
    worker_wait: float = 0.0,
    worker_imbalance: float = 0.0,
) -> VectorRolloutRoundTiming:
    return VectorRolloutRoundTiming(
        phase=phase,  # type: ignore[arg-type]
        active_slots=active_slots,
        capacity=capacity,
        planner_wall_s=planner,
        environment_wall_s=environment,
        worker_busy_s=worker_busy,
        worker_wait_s=worker_wait,
        worker_imbalance_s=worker_imbalance,
    )


def test_rollout_result_splits_decision_and_partial_bootstrap_metrics() -> None:
    timings = (
        _timing(
            "decision",
            active_slots=4,
            capacity=4,
            planner=1.0,
            environment=2.0,
            worker_busy=4.0,
            worker_wait=3.0,
            worker_imbalance=2.0,
        ),
        _timing(
            "decision",
            active_slots=4,
            capacity=4,
            planner=1.0,
            environment=2.0,
            worker_busy=4.0,
            worker_wait=3.0,
            worker_imbalance=2.0,
        ),
        _timing("bootstrap", active_slots=2, capacity=4, planner=0.5),
    )

    result = _rollout_result("vector", 4, 4, [10.0], [2.0], [timings])

    assert result["planner_decision_wall_s"]["median"] == 2.0  # type: ignore[index]
    assert result["planner_bootstrap_wall_s"]["median"] == 0.5  # type: ignore[index]
    assert result["environment_wall_s"]["median"] == 4.0  # type: ignore[index]
    assert result["collection_overhead_s"]["median"] == 3.5  # type: ignore[index]
    assert result["worker_busy_s_per_transition"]["median"] == 2.0  # type: ignore[index]
    assert result["worker_wait_s_per_transition"]["median"] == 1.5  # type: ignore[index]
    assert result["worker_imbalance_s_per_transition"]["median"] == 1.0  # type: ignore[index]
    assert result["decision_batch_fill_ratio"]["median"] == 1.0  # type: ignore[index]
    assert result["bootstrap_batch_fill_ratio"]["median"] == 0.5  # type: ignore[index]
    assert result["policy_planner_batch_wall_s"]["median"] == 1.0  # type: ignore[index]
    assert result["policy_planner_samples_per_s"]["median"] == 4.0  # type: ignore[index]


def test_rollout_result_uses_null_for_absent_bootstrap() -> None:
    timings = (_timing("decision", active_slots=1, capacity=1, planner=1.0),)

    result = _rollout_result("vector", 1, 1, [2.0], [1.0], [timings])

    assert result["planner_bootstrap_wall_s"] is None
    assert result["bootstrap_batch_fill_ratio"] is None


def test_serial_rollout_result_reports_only_comparable_wall_metrics() -> None:
    result = _rollout_result("serial", 1, 2, [4.0], [1.0], [()])

    assert result["rollout_transitions_per_s"]["median"] == 0.5  # type: ignore[index]
    assert result["planner_decision_wall_s"] is None
    assert result["decision_batch_fill_ratio"] is None
