"""Synthetic paired scenarios test statistics and reporting, not research outcomes."""

import json
from dataclasses import replace

import numpy as np
import pytest
from scipy.stats import bootstrap, pearsonr, spearmanr

from eco_planner.analysis.evaluation import (
    ScalarComparison,
    ScalarComparisonRun,
    paired,
    scalar_reward,
)
from eco_planner.analysis.reporting.experiments import experiment_figures
from eco_planner.analysis.reporting.markdown import write_report
from eco_planner.analysis.statistics import (
    ScenarioBootstrapConfig,
    advantage_comparison,
    scenario_effect,
)
from eco_planner.evaluation.artifacts.models import FailedEpisodeSummary
from tests.analysis.test_reports import job
from tests.characterization.test_experiment_outputs import _episode, _training_summary

BOOTSTRAP = ScenarioBootstrapConfig(confidence_level=0.95, n_resamples=10000, bootstrap_seed=0)


def scenario_job(energies, *, failed=(), unsafe=()):
    episodes = []
    for seed, energy in enumerate(energies):
        episode = _episode(seed=seed, distance_m=100.0, energy_ml=float(energy))
        if seed in failed:
            fields = {
                k: v
                for k, v in episode.model_dump(mode="json").items()
                if k in FailedEpisodeSummary.model_fields
            }
            fields.update(
                status="failed",
                trace_status="empty",
                termination={"type": "runtime_error", "detail": "fixture failure"},
                failure={
                    "phase": "inference",
                    "exception_type": "ValueError",
                    "message": "fixture failure",
                    "traceback": "fixture",
                },
            )
            episode = FailedEpisodeSummary.model_validate_json(json.dumps(fields))
        elif seed in unsafe:
            episode = episode.model_copy(
                update={
                    "metrics": episode.metrics.model_copy(
                        update={"collision": True, "out_of_road": True, "wrong_direction": True}
                    )
                }
            )
        episodes.append(episode)
    return job().model_copy(
        update={"episodes": tuple(episodes), "status": "failed" if failed else "completed"}
    )


def run(arm, seed, evaluation, label="final"):
    return ScalarComparisonRun(arm, label, _training_summary(seed, 0), evaluation)


@pytest.fixture
def comparison():
    baseline = scenario_job([10, 20, 30, 40])
    runs = (
        run("a1", 0, scenario_job([9, 19, 29, 39], failed=(0,), unsafe=(3,))),
        run("a2", 0, scenario_job([8, 17, 26, 35], failed=(1,), unsafe=(3,))),
        run("a1", 1, baseline),
        run("a2", 1, scenario_job([11, 22, 33, 44])),
        run("a1", 2, baseline),
        run("a2", 2, scenario_job([9, 18, 27, 36])),
        run("a1", 0, baseline, "initial"),
        run("a2", 0, scenario_job([20, 30, 40, 50]), "initial"),
    )
    return ScalarComparison(baseline, runs, BOOTSTRAP, (0, 1, 2))


def test_scenario_bootstrap_matches_scipy_and_undefined_cases():
    delta = np.array([-4.0, -1.0, 0.0, 2.0, 9.0])
    expected = bootstrap(
        (delta,),
        np.mean,
        method="percentile",
        confidence_level=0.95,
        n_resamples=10000,
        rng=np.random.default_rng(0),
    ).confidence_interval
    result = scenario_effect(delta, BOOTSTRAP)
    assert result["estimate"] == np.mean(delta)
    assert result["ci95"] == list(expected)
    assert result == scenario_effect(delta, BOOTSTRAP)
    assert result["ci_crosses_zero"] is True
    assert scenario_effect(np.array([]), BOOTSTRAP)["estimate"] is None
    single = scenario_effect(np.array([7.0]), BOOTSTRAP)
    assert single["estimate"] == 7.0 and single["ci95"] is None
    assert scenario_effect(np.array([7.0, 7.0]), BOOTSTRAP)["ci95"] == [7.0, 7.0]
    assert scenario_effect(np.array([0.0, 0.0]), BOOTSTRAP)["ci_crosses_zero"] is True
    with pytest.raises(ValueError, match="finite"):
        scenario_effect(np.array([np.nan, 1.0]), BOOTSTRAP)


def test_three_contrasts_safety_denominators_and_initial_separation(comparison):
    data = scalar_reward(comparison)
    primary = data["contrasts"]["a2-a1"]["final"]
    assert primary["direction_counts"] == {
        "lower": 2,
        "higher": 1,
        "zero": 0,
        "unavailable": 0,
        "total": 3,
    }
    assert [e["estimate"] for e in primary["effects"]] == [-3.5, 2.5, -2.5]
    assert [e["training_seed"] for e in primary["effects"]] == [0, 1, 2]
    assert primary["effects"][0]["sample_count"] == 2
    assert data["contrasts"]["a1-a0"]["final"]["effects"][0]["sample_count"] == 3
    assert data["contrasts"]["a2-a0"]["final"]["effects"][0]["sample_count"] == 3
    assert "direction_counts" not in data["contrasts"]["a2-a1"]["initial"]
    assert data["baseline"]["completion"]["episode_count"] == 4
    a2 = data["runs"][1]["outcomes"]
    assert a2["completion"]["completed_rate"] == 0.75
    assert a2["safety"]["denominator"] == 3
    assert a2["safety"]["unavailable_count"] == 1
    assert a2["safety"]["collision"] == {"count": 1, "rate": 1 / 3}
    assert a2["safety"]["out_of_road"]["count"] == 1
    assert a2["failures"][0]["failure"]["message"] == "fixture failure"
    assert primary["effects"][0]["comparison"]["available_rate"] == 0.5


def test_pair_order_stable_and_missing_or_duplicate_keys_fail(comparison):
    data = scalar_reward(comparison)
    reversed_runs = tuple(
        replace(
            r,
            evaluation=r.evaluation.model_copy(
                update={"episodes": tuple(reversed(r.evaluation.episodes))}
            ),
        )
        for r in reversed(comparison.runs)
    )
    reordered = replace(comparison, runs=reversed_runs)
    assert scalar_reward(reordered)["contrasts"] == data["contrasts"]
    for episodes in (
        comparison.baseline.episodes[:-1],
        (*comparison.baseline.episodes, comparison.baseline.episodes[0]),
    ):
        with pytest.raises(ValueError, match="pairs"):
            paired(
                comparison.baseline, comparison.baseline.model_copy(update={"episodes": episodes})
            )


@pytest.mark.parametrize("missing", [False, True])
def test_three_layer_report_and_seed_figures(comparison, tmp_path, missing):
    if missing:
        comparison = replace(
            comparison,
            runs=tuple(
                r for r in comparison.runs if not (r.arm == "a2" and r.training.training_seed == 2)
            ),
        )
    data = scalar_reward(comparison)
    files = experiment_figures("scalar-reward", data, tmp_path)
    write_report("scalar-reward", tmp_path, tmp_path, data, files)
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert (
        report.index("## 1. Completion")
        < report.index("## 2. Safety")
        < report.index("## 3. Paired")
    )
    assert report.index("## 3. Paired") < report.index("![")
    for contrast in ("a2-a1", "a1-a0", "a2-a0"):
        svg = (tmp_path / f"figures/{contrast}-seed-effects.svg").read_text(encoding="utf-8")
        for seed in (0, 1, 2):
            assert f"seed {seed}" in svg
        assert "scenario bootstrap 95% CI" in svg
    if missing:
        final = data["contrasts"]["a2-a1"]["final"]
        assert final["direction_counts"]["unavailable"] == 1
        assert final["partial"] is True
        assert "unavailable" in (tmp_path / "figures/a2-a1-seed-effects.svg").read_text(
            encoding="utf-8"
        )
    no_figures = tmp_path / "without-figures"
    no_figures.mkdir()
    write_report("scalar-reward", tmp_path, no_figures, data, [])
    assert "Scenario 95% CI" in (no_figures / "report.md").read_text()
    assert not (no_figures / "figures").exists()


def test_scipy_correlations_preserve_ties_and_undefined():
    x, y = np.array([1.0, 2.0, 2.0, 8.0]), np.array([4.0, 2.0, 3.0, 0.0])
    stats = advantage_comparison(x, y)
    assert stats["pearson"] == pearsonr(x, y).statistic
    assert stats["spearman"] == spearmanr(x, y).statistic
    for a, b in ((np.ones(4), y), (np.array([1.0]), np.array([2.0]))):
        result = advantage_comparison(a, b)
        assert result["pearson"] is None and result["spearman"] is None
