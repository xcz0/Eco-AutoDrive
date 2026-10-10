"""Matched cadence-attribution study: schema, orchestration glue, and CLI routing."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import median
from types import SimpleNamespace

import pytest

from eco_planner.contracts import CLOSED_LOOP_EXECUTION_STEPS
from eco_planner.experiments.training.cadence_attribution import runner
from eco_planner.experiments.training.cadence_attribution.diagnostics import (
    CadenceAttributionConfig,
    load_cadence_attribution,
)

_STUDY = """\
version: 1
study_name: synthetic_cadence_attribution
protocol: experiments/comparison/calibrated.yaml
arm: r0
training_seed: 0
update_count: 4
base_overrides: []
execution_steps:
- 1
- 5
historical_comparator: E-039 effective-update region
mc_draws: 8
mc_seed: 1
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


def _study(tmp_path: Path) -> CadenceAttributionConfig:
    path = tmp_path / "cadence-attribution.yaml"
    path.write_text(_STUDY, encoding="utf-8")
    return load_cadence_attribution(path)


def _synthetic_metrics(count: int, scale: float) -> dict:
    kl = [1.0e-05 * scale + index * 1.0e-08 for index in range(count)]
    analytic = [value * 1.2 for value in kl]
    zero = [0.0] * count
    ones = [1.0] * count
    return {
        "post_update_kl": kl,
        "post_update_kl_median": median(kl),
        "post_update_kl_analytic": analytic,
        "post_update_kl_analytic_median": median(analytic),
        "post_update_kl_single_draw_k3_median": median(kl) * 0.5,
        "policy_ratio_change": 1.0e-03 * scale,
        "policy_ratio_mean_median": 1.0 + 1.0e-07 * scale,
        "probe_guidance_rms_shift": 0.02 * scale,
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
        "pre_clip_gradient_norm": [30.0 * scale] * count,
        "post_clip_gradient_norm": [0.5] * count,
        "parameter_delta_vs_initial_final": {
            "actor_head": 0.02 * scale,
            "shared_trunk": 1.8,
            "value_head": 0.05,
        },
        "raw_advantage_mean": [0.1 * scale] * count,
        "raw_advantage_std": [2.0 * scale] * count,
        "normalized_advantage_mean": [0.0] * count,
        "normalized_advantage_std": ones,
        "center_advantage_mean": zero,
        "center_advantage_std": [2.0 * scale] * count,
        "value_target_mean": [5.0 * scale] * count,
        "value_target_std": [3.0 * scale] * count,
        "reward_total": [10.0 * scale] * count,
        "reward_component_means": {
            component: [1.0 * scale] * count
            for component in ("ttc", "progress", "comfort", "speed", "energy")
        },
        "effective_global_clip_coefficient": [0.5 / (30.0 * scale)] * count,
        "beta_concentration_mean": [[4.0, 4.0]] * count,
        "gradient_diagnostics": {
            group: [1.0 * scale] * count
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


def test_config_validation_and_diagnostic_selection(tmp_path: Path) -> None:
    study = _study(tmp_path)
    assert study.execution_steps == [1, 5]
    assert study.diagnostic_steps() == 1
    assert study.arm_label(5) == "k5"
    base = study.model_dump()
    cases = []
    unsorted = dict(base, execution_steps=[5, 1])
    cases.append(unsorted)
    missing_canonical = dict(base, execution_steps=[1, 2])
    cases.append(missing_canonical)
    only_canonical = dict(base, execution_steps=[5, 5])
    cases.append(only_canonical)
    out_of_range = dict(base, execution_steps=[0, 5])
    cases.append(out_of_range)
    for data in cases:
        with pytest.raises(ValueError):
            CadenceAttributionConfig.model_validate(data)


def test_summarize_and_compare_report_medians_and_ratio() -> None:
    summary = runner._summarize([1.0, 2.0, 3.0])
    assert summary == {"median": 2.0, "mean": 2.0, "min": 1.0, "max": 3.0}
    comparison = runner._compare([2.0, 2.0], [4.0, 4.0])
    assert comparison["median_ratio"] == pytest.approx(2.0)
    assert runner._compare([0.0, 0.0], [1.0, 1.0])["median_ratio"] is None


def test_run_orchestration_writes_attribution_and_publishes(tmp_path, monkeypatch) -> None:
    _study(tmp_path)

    monkeypatch.setattr(runner, "_ensure_training_run", lambda *a, **k: None)
    monkeypatch.setattr(
        runner,
        "_load_training_config",
        lambda run_dir: SimpleNamespace(ppo=SimpleNamespace(max_gradient_norm=0.5)),
    )
    monkeypatch.setattr(
        runner,
        "_arm_metrics",
        lambda study, run_dir, parsed: _synthetic_metrics(
            study.update_count, 1.0 if run_dir.name == "k1" else 5.0
        ),
    )
    monkeypatch.setattr(
        runner,
        "_offline_gae_stats",
        lambda study, run_dir, parsed: _synthetic_gae(
            study.update_count, 1.0 if run_dir.name == "k1" else 5.0
        ),
    )
    monkeypatch.setattr(runner, "_check_provenance", lambda *a: None)

    def fake_heldout(study, protocol, output_dir, arm_metrics):
        arms = {}
        for steps in study.execution_steps:
            change = 0.02 if steps == CLOSED_LOOP_EXECUTION_STEPS else 0.0
            arms[steps] = {
                "label": study.arm_label(steps),
                "final_values": _heldout_value(10.0 * (1.0 + change), 5.0 * (1.0 + change)),
            }
        return {"initial_values": _heldout_value(10.0, 5.0), "arms": arms}

    monkeypatch.setattr(runner, "_heldout", fake_heldout)

    output = tmp_path / "out"
    result = runner.run(tmp_path / "cadence-attribution.yaml", output, figures=False)
    assert result["status"] == "completed"
    assert result["execution_steps"] == [1, 5]

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["kind"] == "training-cadence-attribution"
    assert summary["diagnostic_execution_steps"] == 1
    assert [arm["label"] for arm in summary["arms"]] == ["k1", "k5"]
    k1 = next(arm for arm in summary["arms"] if arm["label"] == "k1")
    k5 = next(arm for arm in summary["arms"] if arm["label"] == "k5")
    assert k5["metrics"]["value_target_mean"][0] == pytest.approx(25.0)
    assert k1["metrics"]["value_target_mean"][0] == pytest.approx(5.0)

    attribution = summary["attribution"]
    assert attribution["execution_steps"] == {"diagnostic": 1, "canonical": 5}
    assert attribution["value_target"]["mean"]["median_ratio"] == pytest.approx(5.0)
    assert attribution["value_target"]["mean"]["diagnostic"]["median"] == pytest.approx(5.0)
    assert set(attribution["advantage_forms"]) == {"raw", "center", "normalized"}
    assert set(attribution["gate_a_evidence"]) == {
        "actor_state_advantage_geometry",
        "reward_value_scale",
        "physical_time_credit",
        "critic_trunk_clipping_coupling",
    }
    # The canonical arm passes held-out condition c7; the diagnostic arm does not.
    assert k5["gate"]["conditions"]["c7_heldout_exceeds_noise"] is True
    assert k1["gate"]["conditions"]["c7_heldout_exceeds_noise"] is False
    assert (output / "report.md").is_file()
    assert (output / "analysis.json").is_file()
    assert not (output / "figures").exists()


def test_cadence_attribution_cli_routes(monkeypatch) -> None:
    from scripts import experiments as cli

    calls = []
    monkeypatch.setattr(runner, "run", lambda *a, **kw: calls.append((a, kw)))
    args = cli.build_parser().parse_args(
        [
            "training",
            "cadence-attribution",
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
            "cadence-attribution",
            "analyze",
            "--source-dir",
            "source",
            "--output-dir",
            "out",
        ]
    )
    assert analyze_args.action == "analyze"
