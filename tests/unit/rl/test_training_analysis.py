from __future__ import annotations

import numpy as np

from eco_planner.rl.analysis import summarize_training_runs
from eco_planner.rl.artifacts import PolicyProbeSummary, TrainingRunSummary, TrainingUpdateSummary


def test_summarize_training_runs_validates_replays_and_acceptance(tmp_path) -> None:
    for seed in (0, 1):
        for replay in (0, 1):
            _write_run(tmp_path, seed, replay)

    report = summarize_training_runs(tmp_path)

    assert report["status"] == "passed"
    assert report["total_runs"] == 4
    assert report["total_transitions"] == 8
    assert report["replay_checks"] == [
        {"training_seed": 0, "exact": True},
        {"training_seed": 1, "exact": True},
    ]


def _write_run(root, seed: int, replay: int) -> None:
    run_path = root / f"seed-{seed}-replay-{replay}"
    updates_path = run_path / "updates"
    updates_path.mkdir(parents=True)
    np.savez(updates_path / "episode.npz", seed=np.asarray([seed], dtype=np.int64))
    before = PolicyProbeSummary(
        alpha=((1.0, 1.0), (1.0, 1.0)),
        beta=((1.0, 1.0), (1.0, 1.0)),
        guidance_mean=((0.0, 0.0), (0.0, 0.0)),
        boundary_mass=((0.0, 0.0), (0.0, 0.0)),
    )
    after = PolicyProbeSummary(
        alpha=((1.1, 1.0), (1.2, 1.0)),
        beta=((1.0, 1.0), (1.0, 1.0)),
        guidance_mean=((0.1, 0.0), (0.2, 0.0)),
        boundary_mass=((0.0, 0.0), (0.0, 0.0)),
    )
    summary = TrainingRunSummary(
        training_seed=seed,
        replay_id=replay,
        noise_seeds=(seed * 10 + 1,),
        policy_action_seeds=(seed * 10 + 2,),
        total_transitions=2,
        initial_policy_hash="a" * 64,
        final_policy_hash="b" * 64,
        frozen_planner_hash_before="c" * 64,
        frozen_planner_hash_after="c" * 64,
        probe_before=before,
        probe_after=after,
        updates=(_update(0, 1.0), _update(1, 2.0)),
    )
    (run_path / "summary.json").write_text(summary.model_dump_json(), encoding="utf-8")


def _update(index: int, reward: float) -> TrainingUpdateSummary:
    return TrainingUpdateSummary(
        update_index=index,
        sample_count=1,
        episode_count=1,
        total_reward=reward,
        dense_reward=reward,
        terminal_override=0.0,
        route_completion_delta=1.0,
        distance_m=1.0,
        mean_speed_mps=1.0,
        stopped_fraction=0.0,
        collision_count=0,
        out_of_road_count=0,
        maximum_position_error_m=0.0,
        maximum_heading_error_rad=0.0,
        beta_alpha_mean=(1.0, 1.0),
        beta_beta_mean=(1.0, 1.0),
        action_mean=(0.0, 0.0),
        action_variance=(0.0, 0.0),
        mean_policy_loss=0.0,
        mean_value_loss=0.0,
        mean_entropy_loss=0.0,
        mean_total_loss=0.0,
        mean_approximate_kl=0.0,
        mean_clip_fraction=0.0,
        mean_entropy=0.0,
        mean_explained_variance=0.0,
        maximum_pre_clip_gradient_norm=1.0,
        final_learning_rate=0.0,
    )
