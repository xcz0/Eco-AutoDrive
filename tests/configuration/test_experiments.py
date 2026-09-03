from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from omegaconf import DictConfig

from eco_planner.evaluation import EvaluationJobConfig, parse_evaluation_config
from eco_planner.experiments.energy_sweep import load_energy_study
from eco_planner.experiments.ppo_stability.config import TrialParameters, load_stability_config
from eco_planner.experiments.ppo_stability.monitor import StabilityMonitor
from eco_planner.experiments.ppo_stability.report import rank_validation_configs
from eco_planner.experiments.ppo_stability.search import compose_trial_training_config, create_study
from eco_planner.experiments.reward_sanity import evaluate_sanity, load_sanity_config
from eco_planner.rl.artifacts import TrainingUpdateSummary

ComposeConfig = Callable[[str, list[str] | None], DictConfig]


def test_experiment_manifests_are_strict_and_reference_composable_jobs(
    monkeypatch: pytest.MonkeyPatch,
    compose_config: ComposeConfig,
    config_root: Path,
) -> None:
    monkeypatch.setenv("MACHINE_NAME", "rtx3050_laptop")
    energy = load_energy_study(config_root / "experiments" / "energy" / "matrix.yaml")

    for job in energy.jobs:
        for guidance in energy.guidance_profiles:
            config = compose_config(
                job.config_name,
                [f"components/guidance={guidance.config}"],
            )
            parsed = parse_evaluation_config(config)
            assert isinstance(parsed, EvaluationJobConfig)
            assert parsed.runtime.seed == 0
            assert parsed.sampler.name == "ddim5"
            assert parsed.env["random_agent_model"] is False
def test_reward_sanity_config_covers_anti_hacking_and_gate_cases(config_root: Path) -> None:
    config = load_sanity_config(config_root / "experiments" / "reward" / "sanity.yaml")

    assert {item.name for item in config.cases} == {
        "cruise",
        "stationary",
        "extremely_low_speed",
        "slower_progress",
        "low_route_progress",
        "overspeed",
        "following_non_closing",
        "approaching_collision",
        "uncomfortable",
        "collision",
        "out_of_road",
        "wrong_direction",
    }
    assert len(config.comparisons) == 7


def test_reward_sanity_report_requires_every_declared_check_to_pass(config_root: Path) -> None:
    config = load_sanity_config(config_root / "experiments" / "reward" / "sanity.yaml")

    report = evaluate_sanity(config)

    assert report["status"] == "passed"
    assert report["case_count"] == 12
    assert all(item["passed"] for item in report["checks"])


def test_ppo_stability_manifest_composes_balanced_independent_scenarios(
    monkeypatch: pytest.MonkeyPatch,
    config_root: Path,
) -> None:
    monkeypatch.setenv("MACHINE_NAME", "rtx3050_laptop")
    study = load_stability_config(config_root / "experiments" / "ppo" / "stability.yaml")
    parameters = TrialParameters(
        learning_rate=2.5e-5,
        epochs=2,
        batch_size=128,
        minibatch_size=64,
        target_kl=0.01,
    )

    _, training = compose_trial_training_config(
        study,
        parameters,
        training_seed=0,
        update_count=30,
    )

    assert len(training.scenarios) == 8
    assert {(item.map, item.seed) for item in training.scenarios} == {
        (map_name, seed) for seed in range(4) for map_name in ("S", "SC")
    }
    assert training.ppo.target_kl == 0.01
    assert not training.ppo.gradient_diagnostics
    assert training.ppo.scheduler_total_optimizer_steps == 120


def test_stability_monitor_reports_registered_domain_prune_reasons(config_root: Path) -> None:
    study = load_stability_config(config_root / "experiments" / "ppo" / "stability.yaml")
    monitor = StabilityMonitor(study.pruning)
    update = lambda index, **values: TrainingUpdateSummary.model_construct(  # noqa: E731
        update_index=index,
        sample_count=100,
        mean_episode_length=values.get("episode_length", 10.0),
        out_of_road_count=values.get("out_of_road", 0),
        kl_early_stopped=values.get("early_stop", False),
        mean_clip_fraction=values.get("clip_fraction", 0.0),
        mean_approximate_kl=0.0,
    )

    assert monitor.add(update(0)) is None
    assert monitor.add(update(1, episode_length=2.9)) == "episode_length_below_minimum_retention"

    monitor = StabilityMonitor(study.pruning)
    reasons = [monitor.add(update(index, out_of_road=10)) for index in range(3)]
    assert reasons[-1] == "sustained_out_of_road"

    monitor = StabilityMonitor(study.pruning)
    reasons = [monitor.add(update(index, early_stop=True)) for index in range(3)]
    assert reasons[-1] == "sustained_target_kl_early_stop"

    monitor = StabilityMonitor(study.pruning)
    reasons = [monitor.add(update(index, clip_fraction=0.5)) for index in range(3)]
    assert all(reason is None for reason in reasons)
    assert study.pruning.clip_fraction is None

    enabled = StabilityMonitor(study.pruning.model_copy(update={"clip_fraction": 0.5}))
    reasons = [enabled.add(update(index, clip_fraction=0.5)) for index in range(3)]
    assert reasons[-1] == "sustained_clip_fraction"


def test_stability_experiment_sqlite_continuation_and_validation_ranking(
    config_root: Path,
    tmp_path: Path,
) -> None:
    config = load_stability_config(config_root / "experiments" / "ppo" / "stability.yaml")

    first = create_study(config, tmp_path)
    second = create_study(config, tmp_path)

    assert first.study_name == second.study_name
    records = [
        {
            "config_id": config_id,
            "state": "complete",
            "minimum_episode_length_retention": retention,
            "evaluation": {"passed": True, "route_progress_retention": route},
        }
        for config_id, retention, route in (
            (3, 0.95, 1.0),
            (3, 0.90, 1.0),
            (7, 0.92, 1.1),
            (7, 0.92, 1.1),
        )
    ]

    assert rank_validation_configs(records, required_seed_count=2) == [7, 3]

    records_with_failed_run = [
        {
            "config_id": 3,
            "state": "failed",
            "reason": "ValueError: guidance action must be strictly inside (-1, 1)",
            "minimum_episode_length_retention": 1.0,
            "evaluation": None,
        },
        {
            "config_id": 3,
            "state": "complete",
            "minimum_episode_length_retention": 0.95,
            "evaluation": {"passed": True, "route_progress_retention": 1.0},
        },
    ]

    assert rank_validation_configs(records_with_failed_run, required_seed_count=2) == []
