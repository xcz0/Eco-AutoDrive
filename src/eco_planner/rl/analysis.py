"""Acceptance validation for pre-registered PPO training runs."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from eco_planner.rl.artifacts import TrainingRunSummary


def summarize_training_runs(root: Path) -> dict[str, object]:
    """Validate the four pre-registered closed-loop training runs under ``root``."""

    runs: dict[tuple[int, int], tuple[Path, TrainingRunSummary]] = {}
    for summary_path in sorted(root.glob("seed-*-replay-*/summary.json")):
        summary = TrainingRunSummary.model_validate_json(summary_path.read_text(encoding="utf-8"))
        key = (summary.training_seed, summary.replay_id)
        if key in runs:
            raise ValueError(f"duplicate training run {key}")
        runs[key] = (summary_path.parent, summary)
    expected = {(seed, replay) for seed in (0, 1) for replay in (0, 1)}
    if set(runs) != expected:
        raise ValueError(f"training runs mismatch: missing={sorted(expected - set(runs))}")

    replay_checks = []
    for seed in (0, 1):
        first_path, first = runs[(seed, 0)]
        second_path, second = runs[(seed, 1)]
        first_payload = first.model_dump(mode="json")
        second_payload = second.model_dump(mode="json")
        first_payload["replay_id"] = 0
        second_payload["replay_id"] = 0
        if first_payload != second_payload:
            raise ValueError(f"training seed {seed} summary is not exactly replayable")
        _compare_traces(first_path, second_path)
        replay_checks.append({"training_seed": seed, "exact": True})

    seed_zero = runs[(0, 0)][1]
    seed_one = runs[(1, 0)][1]
    if set(seed_zero.noise_seeds + seed_zero.policy_action_seeds) & set(
        seed_one.noise_seeds + seed_one.policy_action_seeds
    ):
        raise ValueError("independent training seeds reused a rollout random stream")
    run_checks = []
    for key, (_, summary) in sorted(runs.items()):
        _validate_run_acceptance(summary)
        run_checks.append(
            {
                "training_seed": key[0],
                "replay_id": key[1],
                "reward_sequence": [item.total_reward for item in summary.updates],
                "final_policy_hash": summary.final_policy_hash,
            }
        )
    return {
        "status": "passed",
        "total_runs": 4,
        "total_transitions": sum(item[1].total_transitions for item in runs.values()),
        "replay_checks": replay_checks,
        "runs": run_checks,
    }


def _validate_run_acceptance(summary: TrainingRunSummary) -> None:
    if summary.initial_policy_hash == summary.final_policy_hash:
        raise ValueError("PPO did not change the Exploration Policy")
    if summary.frozen_planner_hash_before != summary.frozen_planner_hash_after:
        raise ValueError("PPO changed the frozen planner")
    rewards = [item.total_reward for item in summary.updates]
    if max(rewards) - min(rewards) <= 1e-6:
        raise ValueError("training reward sequence did not change")
    if any(item.maximum_pre_clip_gradient_norm <= 0.0 for item in summary.updates):
        raise ValueError("training update has a zero gradient norm")
    if any(
        item.collision_count
        or item.out_of_road_count
        or item.stopped_fraction >= 0.05
        or item.route_completion_delta <= 0.0
        for item in summary.updates
    ):
        raise ValueError("training run failed the safety/progress anti-degeneracy gate")
    before = np.asarray(summary.probe_before.alpha + summary.probe_before.beta)
    after = np.asarray(summary.probe_after.alpha + summary.probe_after.beta)
    if float(np.max(np.abs(after - before))) <= 1e-6:
        raise ValueError("fixed-probe Beta parameters did not change")
    means = np.asarray(summary.probe_after.guidance_mean)
    if not np.any(np.std(means, axis=0) > 1e-6):
        raise ValueError("trained guidance distribution does not vary across observations")


def _compare_traces(first: Path, second: Path) -> None:
    first_files = sorted(path.relative_to(first) for path in (first / "updates").rglob("*.npz"))
    second_files = sorted(path.relative_to(second) for path in (second / "updates").rglob("*.npz"))
    if first_files != second_files:
        raise ValueError("same-seed replay produced different episode files")
    for relative in first_files:
        with (
            np.load(first / relative, allow_pickle=False) as left,
            np.load(second / relative, allow_pickle=False) as right,
        ):
            if set(left.files) != set(right.files):
                raise ValueError(f"replay trace fields differ for {relative}")
            for name in left.files:
                if not np.array_equal(left[name], right[name]):
                    raise ValueError(f"replay trace {relative}:{name} is not exact")
