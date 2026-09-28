"""Phase B causal-counterfactual study: schema, orchestration glue, and CLI routing."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import median
from types import SimpleNamespace

import pytest

from eco_planner.experiments.training.counterfactual_attribution import runner
from eco_planner.experiments.training.counterfactual_attribution.diagnostics import (
    CounterfactualAttributionConfig,
    load_counterfactual_attribution,
)

_STUDY = """\
version: 1
study_name: synthetic_counterfactual
protocol: experiments/comparison/calibrated.yaml
arm: r0
training_seed: 0
update_count: 4
base_overrides: []
mc_draws: 8
mc_seed: 1
historical_comparator: E-050 Phase A
reference:
  label: k5_baseline
  mechanism: canonical_k5_reference
  overrides: []
  description: canonical k=5 reference
arms:
- label: reward_scale_normalized
  mechanism: reward_value_scale
  overrides:
  - ppo.diagnostic_reward_divisor=5.0
  description: divide training reward by k
- label: value_loss_scaled
  mechanism: critic_coupling
  overrides:
  - ppo.value_coefficient=0.1
  description: scale the value loss
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


def _study(tmp_path: Path) -> CounterfactualAttributionConfig:
    path = tmp_path / "counterfactual-attribution.yaml"
    path.write_text(_STUDY, encoding="utf-8")
    return load_counterfactual_attribution(path)


def _synthetic_metrics(count: int, scale: float) -> dict:
    kl = [1.0e-05 * scale + index * 1.0e-08 for index in range(count)]
    analytic = [value * 1.2 for value in kl]
    zero = [0.0] * count
    ones = [1.0] * count
    return {
        "initial_policy_hash": "shared-initial-hash",
        "final_policy_hash": f"final-{scale}",
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


def test_config_validation_enforces_single_reference_and_unique_labels(tmp_path: Path) -> None:
    study = _study(tmp_path)
    assert [arm.label for arm in study.all_arms()] == [
        "k5_baseline",
        "reward_scale_normalized",
        "value_loss_scaled",
    ]
    base = study.model_dump()
    cases = [
        dict(base, reference={**base["reference"], "overrides": ["ppo.gamma=0.9"]}),
        dict(base, arms=[base["arms"][0], base["arms"][0]]),
        dict(base, arms=[{**base["arms"][0], "label": "k5_baseline"}]),
        dict(base, arms=[{**base["arms"][0], "overrides": []}]),
        dict(base, arms=[{**base["arms"][0], "overrides": ["runtime.seed=1"]}]),
        dict(
            base,
            arms=[{**base["arms"][0], "overrides": ["ppo.gamma=0.9", "ppo.gamma=0.8"]}],
        ),
    ]
    for data in cases:
        with pytest.raises(ValueError):
            CounterfactualAttributionConfig.model_validate(data)


def test_check_shared_initial_policy_rejects_divergent_start() -> None:
    runner._check_shared_initial_policy(
        {"a": {"initial_policy_hash": "x"}, "b": {"initial_policy_hash": "x"}}
    )
    with pytest.raises(ValueError, match="share one initial policy"):
        runner._check_shared_initial_policy(
            {"a": {"initial_policy_hash": "x"}, "b": {"initial_policy_hash": "y"}}
        )


def test_run_orchestration_writes_gate_b_evidence_and_publishes(tmp_path, monkeypatch) -> None:
    _study(tmp_path)
    scales = {"k5_baseline": 5.0, "reward_scale_normalized": 1.0, "value_loss_scaled": 2.0}

    monkeypatch.setattr(runner, "_ensure_training_run", lambda *a, **k: None)
    monkeypatch.setattr(
        runner,
        "_load_training_config",
        lambda run_dir: SimpleNamespace(
            ppo=SimpleNamespace(max_gradient_norm=0.5, diagnostic_reward_divisor=1.0)
        ),
    )
    monkeypatch.setattr(
        runner,
        "_arm_metrics",
        lambda study, run_dir, parsed: _synthetic_metrics(study.update_count, scales[run_dir.name]),
    )
    monkeypatch.setattr(
        runner,
        "_offline_gae_stats",
        lambda study, run_dir, parsed: _synthetic_gae(study.update_count, scales[run_dir.name]),
    )
    monkeypatch.setattr(runner, "_check_provenance", lambda *a: None)

    def fake_heldout(study, protocol, output_dir, specs):
        arms = {
            spec.label: {
                "label": spec.label,
                "final_values": _heldout_value(10.0, 5.0),
            }
            for spec in specs
        }
        return {"initial_values": _heldout_value(10.0, 5.0), "arms": arms}

    monkeypatch.setattr(runner, "_heldout", fake_heldout)

    output = tmp_path / "out"
    result = runner.run(tmp_path / "counterfactual-attribution.yaml", output, figures=False)
    assert result["status"] == "completed"
    assert result["reference_label"] == "k5_baseline"

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["kind"] == "training-counterfactual-attribution"
    assert summary["reference_label"] == "k5_baseline"
    assert [arm["label"] for arm in summary["arms"]] == [
        "k5_baseline",
        "reward_scale_normalized",
        "value_loss_scaled",
    ]
    assert [arm["diagnostic_only"] for arm in summary["arms"]] == [False, True, True]

    attribution = summary["attribution"]
    assert attribution["reference_label"] == "k5_baseline"
    assert set(attribution["gate_b_evidence"]) == {"reward_value_scale", "critic_coupling"}
    assert attribution["gate_b_evidence"]["reward_value_scale"]["arms"] == [
        "reward_scale_normalized"
    ]
    ratios = attribution["ratios_vs_reference"]
    assert ratios["reward_scale_normalized"]["value_target_mean"] == pytest.approx(0.2)
    assert ratios["value_loss_scaled"]["value_target_mean"] == pytest.approx(0.4)
    assert (output / "report.md").is_file()
    assert (output / "analysis.json").is_file()
    assert not (output / "figures").exists()


def test_counterfactual_attribution_cli_routes(monkeypatch) -> None:
    from scripts import experiments as cli

    calls = []
    monkeypatch.setattr(runner, "run", lambda *a, **kw: calls.append((a, kw)))
    args = cli.build_parser().parse_args(
        [
            "training",
            "counterfactual-attribution",
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
            "counterfactual-attribution",
            "analyze",
            "--source-dir",
            "source",
            "--output-dir",
            "out",
        ]
    )
    assert analyze_args.action == "analyze"
