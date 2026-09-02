"""Characterize stable evaluation report payloads."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import eco_planner.evaluation.report as evaluation_report
from eco_planner.evaluation import (
    CompletedEpisodeSummary,
    EnergySummary,
    EpisodeMetrics,
    ErrorValues,
    ExecutionErrorSummary,
    MapInputAudit,
    NoGuidanceSummary,
    SamplerSummary,
    ScenarioSummary,
    SpeedSummary,
    TerminationSummary,
    TrafficObservationSummary,
    WarmupSummary,
)
from eco_planner.experiments import energy_sweep as energy_study
from eco_planner.experiments import ppo_reproducibility as training_analysis
from eco_planner.experiments.ppo_stability.validation import (
    PolicyEvaluationSummary,
    compare_policy_evaluations,
)
from eco_planner.experiments.reward_ab import artifacts as reward_ab_artifacts
from eco_planner.experiments.reward_ab import report as reward_ab_analysis
from eco_planner.experiments.reward_ab.config import PPORewardABConfig
from eco_planner.rl.artifacts import PolicyProbeSummary, TrainingRunSummary, TrainingUpdateSummary


def _episode(*, seed: int, distance_m: float, energy_ml: float) -> CompletedEpisodeSummary:
    return CompletedEpisodeSummary(
        scenario=ScenarioSummary(name="traffic", map_sequence="S", seed=seed),
        evaluation_mode="traffic",
        traffic_density=0.2,
        route_length_m=2_500.0,
        noise_seed=seed,
        sampler=SamplerSummary(
            name="ddim5",
            implementation="diffusers",
            num_steps=5,
            timesteps=None,
            initial_noise_scale=1.0,
            ddim_stochasticity=0.0,
            parity_label="fixture",
        ),
        guidance=NoGuidanceSummary(name="none"),
        plan_cycles=2,
        simulator_steps=10,
        environment_steps_including_warmup=10,
        metrics=EpisodeMetrics(
            simulated_seconds=1.0,
            distance_m=distance_m,
            energy=EnergySummary(
                metric="metadrive_fuel_proxy",
                total_ml=energy_ml,
                distance_m=distance_m,
                ml_per_km=energy_ml * 1_000.0 / distance_m,
            ),
            speed_mps=SpeedSummary(minimum=5.0, mean=6.0 + seed, maximum=7.0),
            stopped_fraction=0.0,
            route_completion=0.4 + 0.1 * seed,
            total_reward=10.0 + seed,
            arrive_dest=seed == 1,
            collision=False,
            out_of_road=False,
        ),
        crash_vehicle=False,
        crash_object=False,
        crash_building=False,
        crash_human=False,
        crash_sidewalk=False,
        terminated=True,
        truncated=False,
        terminal_reason="arrive_dest",
        termination=TerminationSummary(type="arrive_dest", detail="fixture"),
        map_input_audit=MapInputAudit(
            speed_limit_sentinel_replaced_count=0,
            speed_limit_existing_preserved_count=1,
            configured_programmatic_lane_speed_limit_kmh=60.0,
            lane_speed_limit_kmh_counts={"60": 1},
            valid_lane_count_min=1,
            valid_lane_count_max=1,
            speed_limit_valid_count_min=1,
            speed_limit_valid_count_max=1,
            speed_limit_mps_min=60.0 / 3.6,
            speed_limit_mps_max=60.0 / 3.6,
            speed_limit_mps_unique_values=(60.0 / 3.6,),
        ),
        history_warmup=WarmupSummary(
            simulator_steps=0,
            simulated_seconds=0.0,
            ego_displacement_m_maximum=0.0,
            participant_count_minimum=0,
            participant_count_maximum=0,
        ),
        traffic_observation=TrafficObservationSummary(
            planning_frames=2,
            frames_with_participants=1,
            frames_with_participants_fraction=0.5,
            participant_count_minimum=0,
            participant_count_maximum=1,
        ),
        trajectory_execution_error=ExecutionErrorSummary(
            position_m=ErrorValues(maximum=0.0, mean=0.0, final=0.0),
            heading_rad=ErrorValues(maximum=0.0, mean=0.0, final=0.0),
        ),
    )


def test_evaluation_matrix_summary_schema_and_statistics_are_stable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    validated = evaluation_report.ValidatedMatrix(
        matrix_root=tmp_path,
        partial=False,
        observed_jobs={(0, 0.2), (1, 0.2)},
        expected_jobs={(0, 0.2), (1, 0.2)},
        scenario_count=1,
        episodes=(
            _episode(seed=0, distance_m=100.0, energy_ml=10.0),
            _episode(seed=1, distance_m=200.0, energy_ml=30.0),
        ),
    )
    monkeypatch.setattr(
        evaluation_report,
        "validate_matrix_artifacts",
        lambda *_args, **_kw: validated,
    )

    report = evaluation_report.build_matrix_report(tmp_path)

    assert set(report) == {
        "matrix_root",
        "matrix_complete",
        "matrix_successful",
        "observed_job_grid",
        "expected_job_grid",
        "expected_episode_count",
        "validated_episode_count",
        "status_counts",
        "termination_type_counts",
        "bootstrap_seed",
        "bootstrap_samples",
        "interface_limits",
        "episodes",
        "statistics",
    }
    assert report["matrix_complete"] is True
    assert report["status_counts"] == {"completed": 2, "failed": 0}
    assert report["episodes"] == [
        {
            "scenario": "traffic",
            "seed": 0,
            "traffic_density": 0.2,
            "terminal_reason": "arrive_dest",
            "status": "completed",
            "termination": {"type": "arrive_dest", "detail": "fixture"},
            "simulated_seconds": 1.0,
            "distance_m": 100.0,
            "energy_total_ml": 10.0,
            "energy_ml_per_km": 100.0,
            "route_completion": 0.4,
            "mean_speed_mps": 6.0,
            "total_reward": 10.0,
        },
        {
            "scenario": "traffic",
            "seed": 1,
            "traffic_density": 0.2,
            "terminal_reason": "arrive_dest",
            "status": "completed",
            "termination": {"type": "arrive_dest", "detail": "fixture"},
            "simulated_seconds": 1.0,
            "distance_m": 200.0,
            "energy_total_ml": 30.0,
            "energy_ml_per_km": 150.0,
            "route_completion": 0.5,
            "mean_speed_mps": 7.0,
            "total_reward": 11.0,
        },
    ]
    statistics = report["statistics"]
    assert statistics["traffic/density_0.20"]["metrics"]["distance_m"]["mean"] == 150.0
    assert statistics["traffic/density_0.20"]["metrics"]["energy_total_ml"]["median"] == 20.0
    assert statistics["traffic/density_0.20"]["arrive_rate"] == 0.5


def test_energy_study_run_record_schema_preserves_episode_and_traffic_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "traffic" / "baseline"
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "episodes": [
                    {"scenario": {"name": "s0", "seed": 0}, "distance_m": 123.0},
                    {"scenario": {"name": "s1", "seed": 0}, "distance_m": 456.0},
                ],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "resolved_config.yaml").write_text(
        "evaluation:\n  mode: traffic\nenv:\n  traffic_density: 0.125\n", encoding="utf-8"
    )
    job = energy_study.EvaluationJobSpec(id="traffic", config_name="fixture")
    guidance = energy_study.GuidanceProfileSpec(
        id="baseline", config="none", longitudinal_scale=None
    )

    def episode_payload(name: str, distance: float) -> dict[str, object]:
        return {"scenario": {"name": name, "seed": 0}, "distance_m": distance}

    episodes = tuple(
        SimpleNamespace(
            evaluation_mode="traffic",
            traffic_density=0.125,
            scenario=SimpleNamespace(
                model_dump=lambda name=name, **_kwargs: {"name": name, "seed": 0}
            ),
            model_dump=lambda name=name, distance=distance, **_kwargs: episode_payload(
                name, distance
            ),
        )
        for name, distance in (("s0", 123.0), ("s1", 456.0))
    )
    monkeypatch.setattr(
        energy_study,
        "load_job_summary",
        lambda _path: SimpleNamespace(status="completed", episodes=episodes),
    )

    record = energy_study._collect_run(job, guidance, run_dir, returncode=0)

    assert record == {
        "job": "traffic",
        "guidance": "baseline",
        "returncode": 0,
        "status": "completed",
        "output_dir": str(run_dir),
        "episodes": [
            {
                "scenario_metadata": {
                    "name": "s0",
                    "seed": 0,
                    "traffic_condition": "low_density_trigger_0.125",
                },
                "evaluation": {"scenario": {"name": "s0", "seed": 0}, "distance_m": 123.0},
            },
            {
                "scenario_metadata": {
                    "name": "s1",
                    "seed": 0,
                    "traffic_condition": "low_density_trigger_0.125",
                },
                "evaluation": {"scenario": {"name": "s1", "seed": 0}, "distance_m": 456.0},
            },
        ],
    }


def _update(
    index: int,
    reward: float,
    *,
    action: float = 0.0,
    fuel: float = 2.0,
    distance: float = 20.0,
    progress: float = 2.0,
    speed: float = 10.0,
    collision: int = 0,
    out_of_road: int = 0,
) -> TrainingUpdateSummary:
    return TrainingUpdateSummary.model_construct(
        update_index=index,
        sample_count=2,
        mean_episode_length=2.0,
        total_reward=reward,
        route_completion_delta=progress,
        mean_speed_mps=speed,
        stopped_fraction=0.0,
        collision_count=collision,
        out_of_road_count=out_of_road,
        action_mean=(0.0, action),
        action_std=(0.0, 0.0),
        executed_fuel_proxy_total_ml=fuel,
        executed_fuel_proxy_distance_m=distance,
        maximum_pre_clip_gradient_norm=1.0,
        mean_policy_loss=-0.1,
        mean_value_loss=0.2,
        mean_entropy=0.3,
        mean_approximate_kl=0.01,
    )


def _training_summary(
    seed: int,
    replay: int,
    *,
    post_update: TrainingUpdateSummary | None = None,
) -> TrainingRunSummary:
    probe_before = PolicyProbeSummary.model_construct(
        alpha=((1.0, 1.0), (1.0, 1.0)),
        beta=((1.0, 1.0), (1.0, 1.0)),
        guidance_mean=((0.0, 0.0), (0.0, 0.0)),
        boundary_mass=((0.0, 0.0), (0.0, 0.0)),
    )
    probe_after = PolicyProbeSummary.model_construct(
        alpha=((1.1, 1.0), (1.0, 1.1)),
        beta=((1.0, 1.1), (1.1, 1.0)),
        guidance_mean=((0.1, 0.0), (0.0, 0.1)),
        boundary_mass=((0.0, 0.0), (0.0, 0.0)),
    )
    return TrainingRunSummary.model_construct(
        status="completed",
        training_seed=seed,
        replay_id=replay,
        noise_seeds=(seed * 10 + 1,),
        policy_action_seeds=(seed * 10 + 2,),
        total_transitions=4,
        initial_policy_hash="a" * 64,
        final_policy_hash="b" * 64,
        frozen_planner_hash_before="c" * 64,
        frozen_planner_hash_after="c" * 64,
        probe_before=probe_before,
        probe_after=probe_after,
        updates=(_update(0, 1.0), post_update or _update(1, 2.0)),
        reward_profile="metadrive_builtin_v1",
    )


def _ab_config() -> PPORewardABConfig:
    return PPORewardABConfig.model_validate(
        {
            "version": 2,
            "base_training_config": "fixture",
            "profiles": [
                {"id": "builtin", "reward_config": "metadrive_builtin_v1"},
                {"id": "energy", "reward_config": "plannerrft_energy_v1"},
            ],
            "matched_training": {
                "update_count": 2,
                "transitions_per_environment": 2,
                "scheduler_total_optimizer_steps": 2,
                "training_seeds": [1],
                "replay_ids": [2],
            },
            "review_thresholds": {
                "longitudinal_action_mean_deadband": 0.01,
                "energy_intensity_deadband_fraction": 0.01,
                "maximum_progress_drop_fraction": 0.2,
                "maximum_mean_speed_drop_fraction": 0.2,
                "maximum_collision_count_increase": 0,
                "maximum_out_of_road_count_increase": 0,
            },
        }
    )


def _rollout(
    action: float,
    fuel: float,
    progress: float,
    speed: float,
    collision: bool,
    out: bool,
) -> dict[str, np.ndarray]:
    return {
        "guidance_action": np.asarray([[0.0, action], [0.0, action]]),
        "route_completion_delta": np.asarray([progress, progress]),
        "speed_mps": np.asarray([speed, speed]),
        "stopped": np.asarray([False, False]),
        "crash_vehicle": np.asarray([collision, False]),
        "crash_object": np.asarray([False, False]),
        "crash_building": np.asarray([False, False]),
        "crash_human": np.asarray([False, False]),
        "crash_sidewalk": np.asarray([False, False]),
        "out_of_road": np.asarray([out, False]),
        "step_distance_m": np.asarray([10.0, 10.0]),
        "native_step_energy_ml": np.asarray([1.0, 1.0]),
        "executed_fuel_proxy_step_energy_ml": np.asarray([fuel, fuel]),
    }


def test_ppo_reward_ab_report_schema_and_effect_window_metrics_are_stable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = _ab_config()
    common_initial = _rollout(0.0, 1.0, 1.0, 5.0, False, False)
    builtin = reward_ab_artifacts.RewardABRunArtifacts(
        tmp_path / "builtin",
        _training_summary(
            1,
            2,
            post_update=_update(1, 2.0, action=0.1, fuel=4.0, progress=4.0, speed=10.0),
        ),
        common_initial,
    )
    energy = reward_ab_artifacts.RewardABRunArtifacts(
        tmp_path / "energy",
        _training_summary(
            1,
            2,
            post_update=_update(
                1,
                2.0,
                action=0.3,
                fuel=3.0,
                progress=2.0,
                speed=9.0,
                collision=1,
                out_of_road=1,
            ),
        ).model_copy(update={"reward_profile": "plannerrft_energy_v1"}),
        common_initial,
    )
    monkeypatch.setattr(reward_ab_analysis, "load_ab_config", lambda _path: config)
    monkeypatch.setattr(
        reward_ab_analysis,
        "load_run",
        lambda path, _profile: builtin if path.name == "builtin" else energy,
    )

    report = reward_ab_analysis.summarize_ab(tmp_path)

    assert report["mechanical_status"] == "passed"
    assert report["review_status"] == "pending_human_review"
    assert set(report) == {
        "mechanical_status",
        "review_status",
        "pairs",
        "cross_seed_summary",
        "mechanical_checks",
        "review_questions",
    }
    pair = report["pairs"][0]
    assert pair["effect_window"] == "updates 1..N-1; update 0 is the matched pre-update collection"
    assert pair["changes"] == {
        "longitudinal_action_mean_delta": pytest.approx(0.2),
        "energy_intensity_change_fraction": pytest.approx(-0.25),
        "progress_change_fraction": pytest.approx(-0.5),
        "mean_speed_change_fraction": pytest.approx(-0.1),
        "collision_count_delta": 1,
        "out_of_road_count_delta": 1,
    }
    assert pair["review_flags"] == {
        "longitudinal_action_changed": True,
        "energy_intensity_changed": True,
        "progress_regressed": True,
        "mean_speed_regressed": False,
        "collision_regressed": True,
        "out_of_road_regressed": True,
        "energy_drop_confounded_by_progress_drop": True,
    }
    assert report["cross_seed_summary"]["aggregation_unit"] == "matched training seed/replay pair"


def _policy_summary(
    label: str,
    *,
    episodes: float,
    progress: float,
    collisions: int = 0,
) -> PolicyEvaluationSummary:
    return PolicyEvaluationSummary.model_validate(
        {
            "checkpoint_label": label,
            "checkpoint_path": f"{label}.pt",
            "policy_hash": "a" * 64,
            "evaluation_seed": 760025,
            "scenarios": ("held-out:S:16",),
            "noise_seeds": (1,),
            "transition_count": 100,
            "episode_count": 1,
            "mean_episode_length": episodes,
            "collision_count": collisions,
            "out_of_road_count": 0,
            "route_completion_delta": progress,
            "distance_m": 100.0,
            "mean_speed_mps": 5.0,
            "stopped_fraction": 0.0,
        }
    )


def test_ppo_stability_comparison_payload_freezes_acceptance_fields() -> None:
    comparison = compare_policy_evaluations(
        _policy_summary("initial", episodes=100.0, progress=10.0),
        _policy_summary("final", episodes=95.0, progress=9.5),
    )

    payload = comparison.model_dump(mode="json")

    assert set(payload) == {
        "initial",
        "final",
        "episode_length_retention",
        "route_progress_retention",
        "collision_count_not_increased",
        "out_of_road_count_not_increased",
        "passed",
    }
    assert payload["episode_length_retention"] == 0.95
    assert payload["route_progress_retention"] == 0.95
    assert payload["collision_count_not_increased"] is True
    assert payload["out_of_road_count_not_increased"] is True
    assert payload["passed"] is True


def test_training_reproducibility_acceptance_report_schema_is_stable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    summaries = {
        (seed, replay): _training_summary(seed, replay) for seed in (0, 1) for replay in (0, 1)
    }
    for seed, replay in summaries:
        run = tmp_path / f"seed-{seed}-replay-{replay}"
        (run / "updates" / "update-000").mkdir(parents=True)
        (run / "summary.json").write_text(
            json.dumps({"seed": seed, "replay": replay}), encoding="utf-8"
        )
        np.savez(run / "updates" / "update-000" / "episode.npz", values=np.asarray([seed, 1]))

    def load_summary(content: str, *_args: object, **_kwargs: object) -> TrainingRunSummary:
        record = json.loads(content)
        return summaries[(record["seed"], record["replay"])]

    monkeypatch.setattr(
        training_analysis.TrainingRunSummary,
        "model_validate_json",
        staticmethod(load_summary),
    )

    report = training_analysis.summarize_training_runs(tmp_path)

    assert report == {
        "status": "passed",
        "total_runs": 4,
        "total_transitions": 16,
        "replay_checks": [
            {"training_seed": 0, "exact": True},
            {"training_seed": 1, "exact": True},
        ],
        "runs": [
            {
                "training_seed": 0,
                "replay_id": 0,
                "reward_sequence": [1.0, 2.0],
                "final_policy_hash": "b" * 64,
            },
            {
                "training_seed": 0,
                "replay_id": 1,
                "reward_sequence": [1.0, 2.0],
                "final_policy_hash": "b" * 64,
            },
            {
                "training_seed": 1,
                "replay_id": 0,
                "reward_sequence": [1.0, 2.0],
                "final_policy_hash": "b" * 64,
            },
            {
                "training_seed": 1,
                "replay_id": 1,
                "reward_sequence": [1.0, 2.0],
                "final_policy_hash": "b" * 64,
            },
        ],
    }
