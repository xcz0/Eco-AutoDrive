"""Task H evaluation-diagnostics manifest, composition, and summary diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from eco_planner.experiments.reward.scalar.composition import compose_policy_evaluation_config
from eco_planner.experiments.reward.scalar.config import load_scalar_reward_protocol
from eco_planner.experiments.training.evaluation_diagnostics.config import (
    EvaluationDiagnosticsStudyConfig,
    load_evaluation_diagnostics_study,
)
from eco_planner.experiments.training.evaluation_diagnostics.diagnostics import (
    COMPARISON_METRICS,
    beta_probe_statistics,
    build_diagnostics_summary,
    deterministic_vs_stochastic_shift,
    paired_beta_deltas,
    stochastic_mean,
    stochastic_metric_summary,
)
from eco_planner.experiments.training.evaluation_diagnostics.runner import (
    _require_source_layout,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = REPOSITORY_ROOT / "configs" / "experiments" / "reward" / "scalar.yaml"
STUDY_PATH = (
    REPOSITORY_ROOT / "configs" / "experiments" / "training" / "evaluation-diagnostics.yaml"
)
STOCHASTIC_OVERRIDES = [
    "evaluation.policy_checkpoint.action_mode=sample",
    "evaluation.policy_checkpoint.policy_action_seed=810001",
]


@pytest.fixture(autouse=True)
def without_machine_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MACHINE_NAME", raising=False)


def _study(**overrides: object) -> EvaluationDiagnosticsStudyConfig:
    values: dict[str, object] = {
        "version": 1,
        "study_name": "unit_evaluation_diagnostics",
        "protocol": "experiments/reward/scalar.yaml",
        "training_seeds": [0, 1],
        "policy_action_seeds": [810001, 810002, 810003],
    }
    values.update(overrides)
    return EvaluationDiagnosticsStudyConfig.model_validate(values)


def test_study_manifest_freezes_the_task_h_inputs() -> None:
    study = load_evaluation_diagnostics_study(STUDY_PATH)

    assert study.study_name == "issue94_task_h_evaluation_diagnostics"
    assert study.training_seeds == [0, 1]
    assert study.policy_action_seeds == [810001, 810002, 810003]
    assert study.protocol_path() == PROTOCOL_PATH


def test_study_manifest_rejects_invalid_seed_lists() -> None:
    with pytest.raises(ValueError, match="unique"):
        _study(training_seeds=[0, 0])
    with pytest.raises(ValueError, match="unique"):
        _study(policy_action_seeds=[1, 1])
    with pytest.raises(ValueError, match="non-negative"):
        _study(policy_action_seeds=[-1])


def test_stochastic_evaluation_composition_pins_the_matched_protocol() -> None:
    protocol = load_scalar_reward_protocol(PROTOCOL_PATH)

    _, parsed = compose_policy_evaluation_config(
        protocol,
        "a1",
        "final",
        Path("runs/r0-seed-0/policy-final.pt"),
        overrides=STOCHASTIC_OVERRIDES,
    )

    assert parsed.policy_checkpoint is not None
    assert parsed.policy_checkpoint.action_mode == "sample"
    assert parsed.policy_checkpoint.policy_action_seed == 810001
    assert parsed.runtime.seed == protocol.evaluation.seed
    assert parsed.sampler.ddim_stochasticity == 0.0


def test_beta_probe_statistics_aggregates_affine_beta_moments() -> None:
    probe = {
        "alpha": [[2.0, 2.0], [2.0, 2.0]],
        "beta": [[2.0, 2.0], [2.0, 2.0]],
    }

    statistics = beta_probe_statistics(probe)

    assert statistics["context_count"] == 2
    assert statistics["beta_mean"]["mean"] == pytest.approx([0.0, 0.0])
    assert statistics["concentration"]["mean"] == pytest.approx([4.0, 4.0])
    assert statistics["variance"]["mean"] == pytest.approx([0.2, 0.2])


def test_paired_beta_deltas_reports_per_dimension_mean_and_rms() -> None:
    reference = {"alpha": [[2.0, 2.0]], "beta": [[2.0, 2.0]]}
    stress = {"alpha": [[1.0, 2.0]], "beta": [[2.0, 2.0]]}

    deltas = paired_beta_deltas(reference, stress)

    assert deltas["concentration"]["mean_delta_per_dimension"] == pytest.approx([-1.0, 0.0])
    assert deltas["beta_mean"]["mean_delta_per_dimension"][1] == pytest.approx(0.0)
    assert deltas["beta_mean"]["mean_delta_per_dimension"][0] < 0.0
    assert deltas["variance"]["rms"] > 0.0


def test_stochastic_metric_summary_and_mean_track_action_seed_spread() -> None:
    values = {
        810001: {"mean_speed_mps": 10.0, "energy_ml_per_km": 47.0},
        810002: {"mean_speed_mps": 10.4, "energy_ml_per_km": 47.2},
    }

    summary = stochastic_metric_summary(values)
    mean = stochastic_mean(values)

    assert summary["mean_speed_mps"]["min"] == pytest.approx(10.0)
    assert summary["mean_speed_mps"]["max"] == pytest.approx(10.4)
    assert summary["mean_speed_mps"]["by_action_seed"]["810002"] == pytest.approx(10.4)
    assert mean["mean_speed_mps"] == pytest.approx(10.2)
    assert mean["energy_ml_per_km"] == pytest.approx(47.1)


def test_stochastic_helpers_reject_empty_or_mismatched_values() -> None:
    with pytest.raises(ValueError, match="at least one"):
        stochastic_metric_summary({})
    with pytest.raises(ValueError, match="at least one"):
        stochastic_mean({})
    with pytest.raises(ValueError, match="metric set"):
        stochastic_mean({1: {"mean_speed_mps": 1.0}, 2: {"energy_ml_per_km": 2.0}})


def test_deterministic_vs_stochastic_shift_reports_relative_effects() -> None:
    shift = deterministic_vs_stochastic_shift(
        {"mean_speed_mps": 10.0, "energy_ml_per_km": 47.0, "route_completion": 0.95},
        {"mean_speed_mps": 10.1, "energy_ml_per_km": 47.0, "route_completion": 0.95},
    )

    assert shift["mean_speed_mps"]["delta"] == pytest.approx(0.1)
    assert shift["mean_speed_mps"]["relative"] == pytest.approx(0.01)
    assert shift["energy_ml_per_km"]["delta"] == pytest.approx(0.0)


def test_build_diagnostics_summary_compares_all_three_layers() -> None:
    records, beta_records = _unit_records()

    summary = build_diagnostics_summary(records, beta_records, [810001, 810002])

    assert summary["status"] == "completed"
    assert summary["training_seeds"] == [0]
    assert summary["policy_action_seeds"] == [810001, 810002]
    assert set(summary["policies"]["0"]) == {"initial", "r0", "rstress"}
    stochastic = summary["policies"]["0"]["rstress"]["stochastic"]["mean_speed_mps"]
    assert stochastic["mean"] == pytest.approx(10.2)
    assert summary["paired_rstress_minus_r0"]["deterministic"]["0"]["mean_speed_mps"][
        "delta"
    ] == pytest.approx(0.2)
    assert summary["paired_rstress_minus_r0"]["stochastic_mean"]["0"]["mean_speed_mps"][
        "delta"
    ] == pytest.approx(0.3)
    assert (
        summary["paired_rstress_minus_r0"]["direction_matches_deterministic"]["0"]["mean_speed_mps"]
        is True
    )
    assert summary["beta_distribution"]["0"]["r0"]["final"]["concentration"]["mean"] == (
        pytest.approx([4.0, 4.0])
    )
    assert set(summary["stochastic_vs_deterministic"]["0"]["r0"]) == set(COMPARISON_METRICS)


def test_source_layout_validation_requires_the_frozen_artifacts(tmp_path: Path) -> None:
    study = _study(training_seeds=[0])
    with pytest.raises(RuntimeError, match="missing"):
        _require_source_layout(tmp_path, study)

    for arm in ("r0", "rstress"):
        run_dir = tmp_path / f"{arm}-seed-0"
        run_dir.mkdir()
        (run_dir / "summary.json").write_text(
            json.dumps({"probe_before": {}, "probe_after": {}}), encoding="utf-8"
        )
        (run_dir / "policy-initial.pt").touch()
        (run_dir / "policy-final.pt").touch()
    for policy in ("initial", "r0", "rstress"):
        heldout = tmp_path / "heldout" / "seed-0" / policy
        heldout.mkdir(parents=True)
        (heldout / "summary.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")

    _require_source_layout(tmp_path, study)


def _heldout_values(speed: float, energy: float) -> dict[str, Any]:
    return {
        "mean_speed_mps": speed,
        "distance_m": 100.0,
        "route_completion": 0.95,
        "energy_total_ml": energy * 0.1,
        "energy_ml_per_km": energy,
        "arrive_dest_fraction": 0.8125,
        "collision_count": 0,
        "out_of_road_count": 0,
        "stopped_fraction": 0.0,
        "simulated_seconds": 30.0,
    }


def _unit_records() -> tuple[dict[int, dict[str, dict[str, Any]]], dict[int, dict[str, Any]]]:
    probe = {"alpha": [[2.0, 2.0]], "beta": [[2.0, 2.0]]}
    records = {
        0: {
            "initial": {
                "checkpoint": "runs/r0-seed-0/policy-initial.pt",
                "deterministic": _heldout_values(10.0, 47.0),
                "stochastic": {
                    810001: _heldout_values(10.0, 47.0),
                    810002: _heldout_values(10.0, 47.0),
                },
            },
            "r0": {
                "checkpoint": "runs/r0-seed-0/policy-final.pt",
                "deterministic": _heldout_values(9.9, 46.9),
                "stochastic": {
                    810001: _heldout_values(9.9, 46.9),
                    810002: _heldout_values(9.9, 46.9),
                },
            },
            "rstress": {
                "checkpoint": "runs/rstress-seed-0/policy-final.pt",
                "deterministic": _heldout_values(10.1, 47.1),
                "stochastic": {
                    810001: _heldout_values(10.1, 47.1),
                    810002: _heldout_values(10.3, 47.2),
                },
            },
        }
    }
    beta_records = {
        0: {arm: {"probe_before": probe, "probe_after": probe} for arm in ("r0", "rstress")}
    }
    return records, beta_records
