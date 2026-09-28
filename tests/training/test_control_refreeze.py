"""Phase C canonical control re-freeze: schema, selection, orchestration, CLI."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import median
from types import SimpleNamespace

import pytest

from eco_planner.experiments.training.control_refreeze import runner
from eco_planner.experiments.training.control_refreeze.diagnostics import (
    ControlRefreezeConfig,
    load_control_refreeze,
)

_STUDY = """\
version: 1
study_name: synthetic_control_refreeze
protocol: experiments/comparison/calibrated.yaml
arm: r0
training_seed: 0
update_count: 4
base_overrides: []
mc_draws: 8
mc_seed: 1
selection: first_passing_in_declared_order
historical_comparator: E-051 Phase B
reference:
  label: canonical_k5
  overrides: []
  description: canonical k=5 reference
candidates:
- label: mechanism_fix
  overrides:
  - ppo.value_coefficient=0.1
  description: minimal critic-scale correction
- label: lr_fallback
  overrides:
  - ppo.learning_rate=0.0003
  description: learning-rate-only fallback
gate:
  target_kl: 0.006
  kl_floor: 1.0e-06
  within_target_kl_fraction: 0.9
  runaway_tail_updates: 10
  runaway_tail_factor: 10.0
  ratio_change_floor: 0.0001
  guidance_rms_shift_floor: 0.01
  beta_parameter_floor: 0.1
  boundary_mass_ceiling: 0.2
  episode_length_retention_floor: 0.5
  collision_budget: 2
  heldout_relative_change_floor: 0.001
  heldout_metrics:
  - mean_speed_mps
  - energy_ml_per_km
"""


def _study(tmp_path: Path) -> ControlRefreezeConfig:
    path = tmp_path / "control-refreeze.yaml"
    path.write_text(_STUDY, encoding="utf-8")
    return load_control_refreeze(path)


def _synthetic_metrics(count: int, *, kl_scale: float, move_scale: float) -> dict:
    kl = [1.0e-05 * kl_scale + index * 1.0e-08 for index in range(count)]
    analytic = [value * 1.2 for value in kl]
    zero = [0.0] * count
    ones = [1.0] * count
    return {
        "initial_policy_hash": "shared-initial-hash",
        "final_policy_hash": f"final-{kl_scale}",
        "post_update_kl": kl,
        "post_update_kl_median": median(kl),
        "post_update_kl_analytic": analytic,
        "post_update_kl_analytic_median": median(analytic),
        "post_update_kl_single_draw_k1": kl,
        "post_update_kl_single_draw_k3_median": median(kl) * 0.5,
        "policy_ratio_change": 1.0e-03 * move_scale,
        "policy_ratio_mean_median": 1.0 + 1.0e-07 * move_scale,
        "policy_ratio_mean": [1.0 + 1.0e-07 * move_scale] * count,
        "policy_ratio_std": [1.0e-05 * move_scale] * count,
        "probe_guidance_rms_shift": 0.02 * move_scale,
        "min_beta_alpha": 1.9,
        "min_beta_beta": 1.9,
        "probe_boundary_mass_max_after": 0.02,
        "probe_boundary_mass_max_before": 0.01,
        "behavior": {
            "collision_count": 0,
            "out_of_road_count": 0,
            "episode_length_first_median": 8.0,
            "episode_length_last_median": 8.0,
        },
        "pre_clip_gradient_norm": [30.0 * move_scale] * count,
        "post_clip_gradient_norm": [0.5] * count,
        "parameter_delta_vs_initial_final": {
            "actor_head": 0.02 * move_scale,
            "shared_trunk": 1.8,
            "value_head": 0.05,
        },
        "raw_advantage_mean": [0.1 * move_scale] * count,
        "raw_advantage_std": [2.0 * move_scale] * count,
        "normalized_advantage_mean": [0.0] * count,
        "normalized_advantage_std": ones,
        "center_advantage_mean": zero,
        "center_advantage_std": [2.0 * move_scale] * count,
        "value_target_mean": [5.0 * move_scale] * count,
        "value_target_std": [3.0 * move_scale] * count,
        "reward_total": [10.0 * move_scale] * count,
        "effective_global_clip_coefficient": [0.5 / (30.0 * move_scale)] * count,
        "beta_concentration_mean": [[4.0, 4.0]] * count,
        "gradient_diagnostics": {
            group: [1.0 * move_scale] * count
            for group in (
                "actor_head_policy",
                "shared_trunk_policy",
                "value_head_critic",
                "shared_trunk_critic",
                "actor_head_entropy",
                "shared_trunk_entropy",
            )
        },
    }


def _synthetic_gae(count: int, scale: float) -> list[dict]:
    return [
        {
            "advantage_mean": 0.1 * scale,
            "advantage_std": 2.0 * scale,
            "advantage_abs_mean": 1.5 * scale,
            "value_target_mean": 5.0 * scale,
            "value_target_std": 3.0 * scale,
            "bootstrap_mean": 0.05,
            "bootstrap_abs_mean": 0.2 * scale,
        }
        for _ in range(count)
    ]


def _heldout_value(speed: float, energy: float) -> dict:
    return {
        "mean_speed_mps": speed,
        "distance_m": 100.0,
        "route_completion": 0.9,
        "energy_total_ml": energy,
        "energy_ml_per_km": energy,
        "arrive_dest_fraction": 0.8,
        "collision_count": 0,
        "out_of_road_count": 0,
    }


def test_config_validation_enforces_control_space_and_unique_labels(tmp_path: Path) -> None:
    study = _study(tmp_path)
    assert [candidate.label for candidate in study.all_runs()] == [
        "canonical_k5",
        "mechanism_fix",
        "lr_fallback",
    ]
    base = study.model_dump()
    cases = [
        dict(base, reference={**base["reference"], "overrides": ["ppo.gamma=0.9"]}),
        dict(base, candidates=[base["candidates"][0], base["candidates"][0]]),
        dict(base, candidates=[{**base["candidates"][0], "label": "canonical_k5"}]),
        dict(base, candidates=[{**base["candidates"][0], "overrides": []}]),
        dict(base, candidates=[{**base["candidates"][0], "overrides": ["runtime.seed=1"]}]),
        dict(
            base,
            candidates=[{**base["candidates"][0], "overrides": ["ppo.gamma=0.9", "ppo.gamma=0.8"]}],
        ),
        dict(
            base,
            candidates=[{**base["candidates"][0], "overrides": ["components/reward=r1"]}],
        ),
        dict(
            base,
            candidates=[
                {**base["candidates"][0], "overrides": ["ppo.diagnostic_reward_divisor=5"]}
            ],
        ),
        dict(
            base,
            candidates=[
                {**base["candidates"][0], "overrides": ["training.diagnostic_execution_steps=1"]}
            ],
        ),
    ]
    for data in cases:
        with pytest.raises(ValueError):
            ControlRefreezeConfig.model_validate(data)


def test_select_picks_first_passing_candidate_in_declared_order(tmp_path: Path) -> None:
    study = _study(tmp_path)
    gate = {
        "canonical_k5": {"passed_all": False},
        "mechanism_fix": {"passed_all": True},
        "lr_fallback": {"passed_all": True},
    }
    selected = runner._select(study, gate)
    assert selected is not None and selected.label == "mechanism_fix"
    assert runner._select(study, {label: {"passed_all": False} for label in gate}) is None


def test_estimator_consistency_and_finiteness() -> None:
    metrics = _synthetic_metrics(4, kl_scale=1.0, move_scale=1.0)
    consistency = runner._estimator_consistency(metrics)
    assert consistency["same_order_of_magnitude"] is True
    assert consistency["max_over_min_ratio"] <= 10.0

    metrics["post_update_kl"][0] = float("nan")
    with pytest.raises(FloatingPointError, match="non-finite"):
        runner._require_finite("mechanism_fix", metrics)


def test_run_orchestration_selects_and_writes_control(tmp_path, monkeypatch) -> None:
    _study(tmp_path)
    scales = {
        "canonical_k5": (4.0e-02, 1.0),
        "mechanism_fix": (1.0, 1.2),
        "lr_fallback": (2.0, 1.4),
    }

    monkeypatch.setattr(runner, "_ensure_training_run", lambda *a, **k: None)
    monkeypatch.setattr(
        runner,
        "_load_training_config",
        lambda run_dir: SimpleNamespace(
            ppo=SimpleNamespace(model_dump=lambda mode: {"learning_rate": 1.5e-4})
        ),
    )
    monkeypatch.setattr(
        runner,
        "_arm_metrics",
        lambda study, run_dir, parsed: _synthetic_metrics(
            study.update_count,
            kl_scale=scales[run_dir.name][0],
            move_scale=scales[run_dir.name][1],
        ),
    )
    monkeypatch.setattr(
        runner,
        "_offline_gae_stats",
        lambda study, run_dir, parsed: _synthetic_gae(study.update_count, 1.0),
    )
    monkeypatch.setattr(runner, "_check_provenance", lambda *a: None)

    def fake_heldout(study, protocol, output_dir, specs):
        runs = {
            spec.label: {
                "label": spec.label,
                "final_values": _heldout_value(
                    10.0, 5.02 if spec.label == "mechanism_fix" else 5.0
                ),
            }
            for spec in specs
        }
        return {"initial_values": _heldout_value(10.0, 5.0), "runs": runs}

    monkeypatch.setattr(runner, "_heldout", fake_heldout)

    output = tmp_path / "out"
    result = runner.run(tmp_path / "control-refreeze.yaml", output, figures=False)
    assert result["status"] == "completed"
    assert result["selection_rule"] == "first_passing_in_declared_order"
    assert result["selected_label"] == "mechanism_fix"
    assert result["selected_config"]["overrides"] == ["ppo.value_coefficient=0.1"]

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["kind"] == "training-control-refreeze"
    assert summary["reference_label"] == "canonical_k5"
    assert [run["label"] for run in summary["runs"]] == [
        "canonical_k5",
        "mechanism_fix",
        "lr_fallback",
    ]
    assert [run["is_reference"] for run in summary["runs"]] == [True, False, False]
    assert summary["runs"][0]["gate"]["passed_all"] is False
    assert summary["runs"][1]["gate"]["passed_all"] is True
    assert summary["attribution"]["ratios_vs_reference"]["mechanism_fix"][
        "value_target_mean"
    ] == pytest.approx(1.2)
    assert (output / "report.md").is_file()
    assert (output / "analysis.json").is_file()
    assert not (output / "figures").exists()


def test_run_records_failed_candidate_and_continues(tmp_path, monkeypatch) -> None:
    _study(tmp_path)

    def fake_ensure(study, protocol, run_dir, spec):
        if spec.label == "lr_fallback":
            raise ValueError("guidance action must be strictly inside (-1, 1)")

    monkeypatch.setattr(runner, "_ensure_training_run", fake_ensure)
    monkeypatch.setattr(
        runner,
        "_load_training_config",
        lambda run_dir: SimpleNamespace(
            ppo=SimpleNamespace(model_dump=lambda mode: {"learning_rate": 1.5e-4})
        ),
    )
    monkeypatch.setattr(
        runner,
        "_arm_metrics",
        lambda study, run_dir, parsed: _synthetic_metrics(
            study.update_count,
            kl_scale=1.0 if run_dir.name == "mechanism_fix" else 4.0e-02,
            move_scale=1.2 if run_dir.name == "mechanism_fix" else 1.0,
        ),
    )
    monkeypatch.setattr(
        runner,
        "_offline_gae_stats",
        lambda study, run_dir, parsed: _synthetic_gae(study.update_count, 1.0),
    )
    monkeypatch.setattr(runner, "_check_provenance", lambda *a: None)

    def fake_heldout(study, protocol, output_dir, specs):
        runs = {
            spec.label: {
                "label": spec.label,
                "final_values": _heldout_value(
                    10.0, 5.02 if spec.label == "mechanism_fix" else 5.0
                ),
            }
            for spec in specs
        }
        return {"initial_values": _heldout_value(10.0, 5.0), "runs": runs}

    monkeypatch.setattr(runner, "_heldout", fake_heldout)

    output = tmp_path / "out"
    result = runner.run(tmp_path / "control-refreeze.yaml", output, figures=False)
    assert result["selected_label"] == "mechanism_fix"
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["failed_candidates"] == ["lr_fallback"]
    fallback = next(run for run in summary["runs"] if run["label"] == "lr_fallback")
    assert fallback["status"] == "failed"
    assert fallback["metrics"] is None
    assert "strictly inside" in fallback["failure"]["message"]


def test_control_refreeze_cli_routes(monkeypatch) -> None:
    from scripts import experiments as cli

    calls = []
    monkeypatch.setattr(runner, "run", lambda *a, **kw: calls.append((a, kw)))
    args = cli.build_parser().parse_args(
        [
            "training",
            "control-refreeze",
            "run",
            "--output-dir",
            "out",
            "--no-figures",
        ]
    )
    assert args.config.is_file()
    cli.dispatch(args)
    assert calls == [((args.config, args.output_dir), {"figures": False})]
    analyze_args = cli.build_parser().parse_args(
        [
            "training",
            "control-refreeze",
            "analyze",
            "--source-dir",
            "source",
            "--output-dir",
            "out",
        ]
    )
    assert analyze_args.action == "analyze"
