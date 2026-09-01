from __future__ import annotations

from eco_planner.experiments.ppo_stability.validation import (
    PolicyEvaluationSummary,
    compare_policy_evaluations,
)


def _summary(
    label: str,
    *,
    collisions: int = 0,
    out_of_road: int = 0,
    episode_length: float = 100.0,
    route_progress: float = 10.0,
) -> PolicyEvaluationSummary:
    return PolicyEvaluationSummary.model_validate(
        {
            "checkpoint_label": label,
            "checkpoint_path": f"policy-{label}.pt",
            "policy_hash": "a" * 64,
            "evaluation_seed": 760025,
            "scenarios": ("held-out:S:16",),
            "noise_seeds": (1,),
            "transition_count": 100,
            "episode_count": 1,
            "mean_episode_length": episode_length,
            "collision_count": collisions,
            "out_of_road_count": out_of_road,
            "route_completion_delta": route_progress,
            "distance_m": 100.0,
            "mean_speed_mps": 5.0,
            "stopped_fraction": 0.0,
        }
    )


def test_policy_evaluation_gate_uses_only_safety_and_progress_metrics() -> None:
    initial = _summary("initial")
    final = _summary("final", episode_length=90.0, route_progress=9.0)

    comparison = compare_policy_evaluations(initial, final)

    assert comparison.passed
    assert comparison.episode_length_retention == 0.9
    assert comparison.route_progress_retention == 0.9
    assert "reward" not in PolicyEvaluationSummary.model_fields


def test_policy_evaluation_rejects_new_safety_failures() -> None:
    comparison = compare_policy_evaluations(
        _summary("initial"),
        _summary("final", out_of_road=1),
    )

    assert not comparison.passed
    assert not comparison.out_of_road_count_not_increased
