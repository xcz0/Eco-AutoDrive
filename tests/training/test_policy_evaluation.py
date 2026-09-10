from __future__ import annotations

from eco_planner.experiments.training.stability.comparison import (
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


def test_stability_evaluation_loads_requested_checkpoint_before_rollout(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import torch

    from eco_planner.configuration import ScenarioConfig
    from eco_planner.experiments.training.stability import validation
    from eco_planner.rl.artifacts import policy_state_hash
    from eco_planner.rl.optimization import save_exploration_policy_checkpoint
    from eco_planner.rl.policy import ExplorationPolicy
    from tests.training.test_ppo import _policy_config

    policy = ExplorationPolicy(_policy_config())
    checkpoint = tmp_path / "policy.pt"
    save_exploration_policy_checkpoint(checkpoint, policy)
    expected_hash = policy_state_hash(policy)
    with torch.no_grad():
        next(policy.parameters()).add_(1)
    assert policy_state_hash(policy) != expected_hash
    runtime = SimpleNamespace(policy=policy)
    config = SimpleNamespace(
        runtime=None,
        sampler=None,
        guidance=None,
        policy=None,
        model=SimpleNamespace(args_path="args.json", checkpoint_path="planner.pt"),
        training=SimpleNamespace(planner_compile_mode=None),
    )
    monkeypatch.setattr(validation, "create_fabric_rollout_runtime", lambda *a, **kw: runtime)
    monkeypatch.setattr(validation, "_evaluation_job_config", lambda *a: object())
    calls = []

    def evaluate(job, output, agent):
        assert policy_state_hash(agent.runtime.policy) == expected_hash
        assert agent.policy_checkpoint.policy_hash == expected_hash
        calls.append(output)
        return object()

    monkeypatch.setattr(validation, "run_evaluation_agent", evaluate)
    expected = _summary("final")
    monkeypatch.setattr(validation, "_summary_from_job", lambda *a, **kw: expected)
    result = validation.evaluate_policy_checkpoint(
        config,
        checkpoint,
        label="final",
        scenarios=(ScenarioConfig(name="s0", map="S", seed=0),),
        transitions_per_scenario=1,
        evaluation_seed=3,
        output_dir=tmp_path / "evaluation",
    )
    assert result is expected
    assert calls == [tmp_path / "evaluation"]
