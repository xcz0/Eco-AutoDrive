"""Task G objective positive-control manifest, composition, pairing, and Gate G."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from eco_planner.configuration import load_resolved_yaml_mapping
from eco_planner.experiments.reward.scalar.config import load_scalar_reward_protocol
from eco_planner.experiments.training.objective_positive_control.composition import (
    compose_frozen_ppo_overrides,
    compose_positive_control_training_config,
    heldout_dir,
    run_label,
)
from eco_planner.experiments.training.objective_positive_control.config import (
    GateGConfig,
    ObjectivePositiveControlStudyConfig,
    load_objective_positive_control_study,
)
from eco_planner.experiments.training.objective_positive_control.diagnostics import (
    evaluate_gate_g,
    flatten_probe_delta,
    heldout_values,
    load_probe_field,
    paired_heldout_deltas,
    probe_cosine,
    probe_rms,
    probe_sign_agreement,
)
from eco_planner.models import Ddim5SamplerConfig
from eco_planner.rl.config import TrainingJobConfig
from eco_planner.rl.reward import (
    PlannerRFTEnergyRewardConfig,
    PlannerRFTNoEnergyRewardConfig,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = REPOSITORY_ROOT / "configs" / "experiments" / "reward" / "scalar.yaml"
STUDY_PATH = (
    REPOSITORY_ROOT / "configs" / "experiments" / "training" / "objective-positive-control.yaml"
)
_STUDY_OVERRIDES: dict[str, object] = {
    "training_seeds": [0, 1],
    "update_count": 50,
    "ppo": {"learning_rate": 1.5e-04, "epochs": 1, "max_gradient_norm": 0.5},
    "base_overrides": ["components/resources=rtx_a4000"],
    "gate": {
        "guidance_rms_floor": 0.01,
        "probe_match_tolerance": 1.0e-06,
        "paired_relative_effect_floor": 0.001,
        "separation_metrics": ["mean_speed_mps", "energy_ml_per_km", "route_completion"],
        "collision_budget": 2,
        "arrive_dest_drop_ceiling": 0.125,
        "stopped_fraction_increase_ceiling": 0.1,
        "route_completion_drop_ceiling": 0.1,
    },
}


@pytest.fixture(autouse=True)
def without_machine_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MACHINE_NAME", raising=False)


def _study(**overrides: object) -> ObjectivePositiveControlStudyConfig:
    values: dict[str, object] = {
        "version": 1,
        "study_name": "unit_positive_control",
        "protocol": "experiments/reward/scalar.yaml",
        "arms": "r0_vs_rstress",
        "r0": {
            "label": "ppo_plannerrft_r0_calibrated",
            "reward_profile": "plannerrft_no_energy_calibrated_v1",
        },
        "rstress": {
            "label": "ppo_plannerrft_band_energy_lam64",
            "reward_profile": "plannerrft_energy_band_lam64_v1",
        },
        **_STUDY_OVERRIDES,
    }
    values.update(overrides)
    return ObjectivePositiveControlStudyConfig.model_validate(values)


def _gate(**overrides: object) -> GateGConfig:
    values: dict[str, object] = {
        "guidance_rms_floor": 0.01,
        "probe_match_tolerance": 1.0e-06,
        "paired_relative_effect_floor": 0.001,
        "separation_metrics": ["mean_speed_mps", "energy_ml_per_km", "route_completion"],
        "collision_budget": 2,
        "arrive_dest_drop_ceiling": 0.125,
        "stopped_fraction_increase_ceiling": 0.1,
        "route_completion_drop_ceiling": 0.1,
    }
    values.update(overrides)
    return GateGConfig.model_validate(values)


def test_study_manifest_freezes_the_task_e_and_task_f_inputs() -> None:
    study = load_objective_positive_control_study(STUDY_PATH)

    assert study.study_name == "issue94_task_g_objective_positive_control"
    assert study.training_seeds == [0, 1]
    assert study.update_count == 50
    assert study.ppo.learning_rate == pytest.approx(1.5e-04)
    assert study.ppo.epochs == 1
    assert study.ppo.max_gradient_norm == pytest.approx(0.5)
    assert study.r0.reward_profile == "plannerrft_no_energy_calibrated_v1"
    assert study.rstress.reward_profile == "plannerrft_energy_band_lam64_v1"
    assert "components/resources=rtx_a4000" in study.base_overrides
    assert "ppo.batch_size=128" in study.base_overrides


def test_study_manifest_rejects_invalid_seeds_and_metric_sets() -> None:
    with pytest.raises(ValueError, match="unique"):
        _study(training_seeds=[0, 0])
    with pytest.raises(ValueError, match="at least 2"):
        _study(training_seeds=[0])
    with pytest.raises(ValueError, match="mean_speed_mps or energy_ml_per_km"):
        _study(gate=_gate(separation_metrics=["route_completion"]))


def test_frozen_ppo_overrides_render_the_task_f_selection() -> None:
    overrides = compose_frozen_ppo_overrides(
        learning_rate=1.5e-04, epochs=1, max_gradient_norm=0.5, update_count=50
    )

    assert overrides == [
        "ppo.learning_rate=0.00015",
        "ppo.epochs=1",
        "ppo.max_gradient_norm=0.5",
        "training.update_count=50",
        "ppo.scheduler_total_optimizer_steps=50",
    ]


def test_run_label_and_heldout_dir_layout() -> None:
    assert run_label("rstress", 1) == "rstress-seed-1"
    root = Path("out")
    assert heldout_dir(root, 0, "initial") == Path("out") / "heldout" / "seed-0" / "initial"
    assert heldout_dir(root, 1, "rstress") == Path("out") / "heldout" / "seed-1" / "rstress"


def test_training_composition_pins_the_matched_pair_and_frozen_region() -> None:
    protocol = load_scalar_reward_protocol(PROTOCOL_PATH)
    study = load_objective_positive_control_study(STUDY_PATH)
    overrides = [
        *study.base_overrides,
        *compose_frozen_ppo_overrides(
            learning_rate=study.ppo.learning_rate,
            epochs=study.ppo.epochs,
            max_gradient_norm=study.ppo.max_gradient_norm,
            update_count=study.update_count,
        ),
    ]
    parsed_by_arm = {}
    for arm_profile in (study.r0.reward_profile, study.rstress.reward_profile):
        for seed in study.training_seeds:
            _, parsed = compose_positive_control_training_config(
                protocol, arm_profile, seed, overrides
            )
            assert isinstance(parsed, TrainingJobConfig)
            assert parsed.reward.name == arm_profile
            assert parsed.runtime.seed == seed
            assert parsed.training.replay_id == 0
            assert isinstance(parsed.sampler, Ddim5SamplerConfig)
            assert {(item.map, item.seed) for item in parsed.scenarios} == protocol.training_pairs()
            assert parsed.ppo.batch_size == 128
            assert parsed.ppo.minibatch_size == 128
            assert parsed.ppo.learning_rate == pytest.approx(1.5e-04)
            assert parsed.ppo.epochs == 1
            assert parsed.ppo.max_gradient_norm == pytest.approx(0.5)
            assert parsed.ppo.scheduler_total_optimizer_steps == 50
            assert parsed.training.update_count == 50
        parsed_by_arm[arm_profile] = parsed
    r0 = parsed_by_arm[study.r0.reward_profile].reward
    rstress = parsed_by_arm[study.rstress.reward_profile].reward
    assert isinstance(r0, PlannerRFTNoEnergyRewardConfig)
    assert isinstance(rstress, PlannerRFTEnergyRewardConfig)
    assert r0.weights.total == 16.0
    assert rstress.weights.energy == 64.0
    assert rstress.weights.total == 80.0
    assert r0.progress == rstress.progress
    assert r0.comfort == rstress.comfort
    assert r0.energy.mode == "reference_exponential"
    assert rstress.energy.mode == "calibrated_band"


def test_training_composition_rejects_protocol_violations() -> None:
    protocol = load_scalar_reward_protocol(PROTOCOL_PATH)

    with pytest.raises(ValueError, match="runtime.seed"):
        compose_positive_control_training_config(
            protocol, "plannerrft_no_energy_calibrated_v1", 0, ["runtime.seed=1"]
        )
    with pytest.raises(ValueError, match="reward profile"):
        compose_positive_control_training_config(
            protocol,
            "plannerrft_no_energy_calibrated_v1",
            0,
            ["components/reward=plannerrft_energy_v1"],
        )
    with pytest.raises(ValueError, match="num_scenarios"):
        compose_positive_control_training_config(
            protocol, "plannerrft_no_energy_calibrated_v1", 0, ["env.num_scenarios=4"]
        )
    with pytest.raises(ValueError, match="training seed"):
        compose_positive_control_training_config(protocol, "plannerrft_no_energy_calibrated_v1", 7)


def test_probe_math_helpers() -> None:
    r0 = [[0.0, 0.0], [1.0, -1.0]]
    rstress = [[0.02, 0.0], [1.0, -0.98]]
    delta = flatten_probe_delta(r0, rstress)
    assert delta == pytest.approx([0.02, 0.0, 0.0, 0.02])
    assert probe_rms(delta) == pytest.approx(0.014142135623731)
    assert probe_rms([0.0]) == 0.0
    same = [0.1, -0.2, 0.3]
    flipped = [-0.1, -0.2, 0.3]
    opposite = [-0.1, 0.2, -0.3]
    assert probe_cosine(same, same) == pytest.approx(1.0)
    assert probe_cosine(same, opposite) == pytest.approx(-1.0)
    assert probe_sign_agreement(same, flipped) == pytest.approx(2.0 / 3.0)
    assert probe_sign_agreement(same, opposite) == pytest.approx(0.0)
    assert probe_cosine([0.0, 0.0], [1.0, 1.0]) == 0.0
    with pytest.raises(ValueError, match="context set"):
        flatten_probe_delta(r0, [[0.0, 0.0]])


def test_load_probe_field_reads_the_training_summary(tmp_path: Path) -> None:
    summary = {
        "probe_before": {"guidance_mean": [[0.0, 0.0]]},
        "probe_after": {"guidance_mean": [[0.5, -0.25]]},
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    assert load_probe_field(tmp_path, "probe_after", "guidance_mean") == [[0.5, -0.25]]
    missing = {"probe_after": {"guidance_mean": [[0.5, -0.25]]}}
    (tmp_path / "summary.json").write_text(json.dumps(missing), encoding="utf-8")
    with pytest.raises(KeyError):
        load_probe_field(tmp_path, "probe_before", "guidance_mean")


def _episode(**metric_updates: object) -> dict[str, Any]:
    metrics: dict[str, object] = {
        "speed_mps": {"mean": 10.0},
        "distance_m": 1000.0,
        "route_completion": 0.5,
        "energy": {"total_ml": 47.0, "ml_per_km": 47.0},
        "arrive_dest": True,
        "collision": False,
        "out_of_road": False,
        "stopped_fraction": 0.0,
        "simulated_seconds": 20.0,
    }
    metrics.update(metric_updates)
    return {"metrics": metrics}


def test_heldout_values_aggregates_collapse_indicators() -> None:
    values = heldout_values(
        [
            _episode(),
            _episode(
                speed_mps={"mean": 8.0},
                stopped_fraction=0.2,
                simulated_seconds=30.0,
                energy={"total_ml": 50.0, "ml_per_km": 50.0},
            ),
        ]
    )

    assert values["mean_speed_mps"] == pytest.approx(9.0)
    assert values["energy_ml_per_km"] == pytest.approx(48.5)
    assert values["arrive_dest_fraction"] == pytest.approx(1.0)
    assert values["stopped_fraction"] == pytest.approx(0.1)
    assert values["simulated_seconds"] == pytest.approx(25.0)
    with pytest.raises(ValueError, match="at least one episode"):
        heldout_values([])


def test_paired_heldout_deltas_handle_none_and_zero_baselines() -> None:
    r0 = {"mean_speed_mps": 10.0, "route_completion": 0.0, "energy_ml_per_km": None}
    rstress = {"mean_speed_mps": 9.0, "route_completion": 0.1, "energy_ml_per_km": 46.0}
    paired = paired_heldout_deltas(
        r0, rstress, ["mean_speed_mps", "route_completion", "energy_ml_per_km"]
    )

    assert paired["mean_speed_mps"]["delta"] == pytest.approx(-1.0)
    assert paired["mean_speed_mps"]["relative"] == pytest.approx(0.1)
    assert paired["route_completion"]["delta"] is None
    assert paired["energy_ml_per_km"]["delta"] is None


def _heldout(**overrides: object) -> dict[str, Any]:
    values: dict[str, object] = {
        "mean_speed_mps": 10.0,
        "distance_m": 2500.0,
        "route_completion": 0.9,
        "energy_total_ml": 118.0,
        "energy_ml_per_km": 47.0,
        "arrive_dest_fraction": 0.8125,
        "collision_count": 0,
        "out_of_road_count": 0,
        "stopped_fraction": 0.0,
        "simulated_seconds": 25.0,
    }
    values.update(overrides)
    return values


def _run_record(heldout: dict[str, Any], probe_after: list[list[float]]) -> dict[str, Any]:
    return {
        "metrics": {},
        "heldout": heldout,
        "probe_after_guidance_mean": probe_after,
    }


def _records(
    r0_heldout_by_seed: dict[int, dict[str, Any]],
    rstress_heldout_by_seed: dict[int, dict[str, Any]],
    r0_probe: list[list[float]] | None = None,
    rstress_probe: list[list[float]] | None = None,
) -> dict[int, dict[str, dict[str, Any]]]:
    r0_probe = r0_probe if r0_probe is not None else [[0.0, 0.0], [0.0, 0.0]]
    rstress_probe = rstress_probe if rstress_probe is not None else [[0.1, -0.1], [0.1, -0.1]]
    records: dict[int, dict[str, dict[str, Any]]] = {}
    for seed in sorted(r0_heldout_by_seed):
        records[seed] = {
            "r0": _run_record(r0_heldout_by_seed[seed], r0_probe),
            "rstress": _run_record(rstress_heldout_by_seed[seed], rstress_probe),
        }
    return records


def _passing_records() -> dict[int, dict[str, dict[str, Any]]]:
    return _records(
        {0: _heldout(), 1: _heldout()},
        {
            0: _heldout(mean_speed_mps=9.5, energy_ml_per_km=46.0),
            1: _heldout(mean_speed_mps=9.4, energy_ml_per_km=45.9),
        },
    )


def test_gate_g_passes_the_positive_control() -> None:
    gate = evaluate_gate_g(_passing_records(), _gate())

    assert gate["conditions"] == {
        "c1_guidance_distribution_separation": True,
        "c2_closed_loop_separation": True,
        "c3_paired_effect_reproduced": True,
        "c4_direction_matches_objective": True,
        "c5_no_collapse_explanation": True,
    }
    assert gate["passed"] is True
    assert gate["failure_reasons"] == []
    separated = gate["separated_metrics"]
    assert separated["mean_speed_mps"]["separated"] is True
    assert separated["energy_ml_per_km"]["separated"] is True
    assert separated["route_completion"]["separated"] is False


def test_gate_g_requires_two_seeds() -> None:
    with pytest.raises(ValueError, match="at least two training seeds"):
        evaluate_gate_g(_records({0: _heldout()}, {0: _heldout(mean_speed_mps=9.5)}), _gate())


def test_gate_g_condition_1_fails_on_small_or_opposed_probe_deltas() -> None:
    small = _records(
        {0: _heldout(), 1: _heldout()},
        {0: _heldout(mean_speed_mps=9.5), 1: _heldout(mean_speed_mps=9.4)},
        r0_probe=[[0.0, 0.0], [0.0, 0.0]],
        rstress_probe=[[0.001, 0.0], [0.0, 0.001]],
    )
    assert (
        evaluate_gate_g(small, _gate())["conditions"]["c1_guidance_distribution_separation"]
        is False
    )

    records_opposed: dict[int, dict[str, dict[str, Any]]] = {
        seed: {
            "r0": _run_record(_heldout(), [[0.0, 0.0], [0.0, 0.0]]),
            "rstress": _run_record(
                _heldout(mean_speed_mps=9.5 if seed == 0 else 9.4),
                ([[-0.1, 0.1], [-0.1, 0.1]] if seed == 1 else [[0.1, -0.1], [0.1, -0.1]]),
            ),
        }
        for seed in (0, 1)
    }
    verdict = evaluate_gate_g(records_opposed, _gate())
    assert verdict["conditions"]["c1_guidance_distribution_separation"] is False
    assert verdict["probe_paired"]["adjacent_seed_cosines"] == [pytest.approx(-1.0)]


def test_gate_g_condition_2_fails_below_noise_or_on_opposed_direction() -> None:
    below_noise = _records(
        {0: _heldout(), 1: _heldout()},
        {
            0: _heldout(mean_speed_mps=9.5, energy_ml_per_km=46.0),
            1: _heldout(mean_speed_mps=9.995, energy_ml_per_km=46.995),
        },
    )
    verdict = evaluate_gate_g(below_noise, _gate())
    assert verdict["conditions"]["c2_closed_loop_separation"] is False
    assert verdict["conditions"]["c3_paired_effect_reproduced"] is False

    opposed = _records(
        {0: _heldout(), 1: _heldout()},
        {
            0: _heldout(mean_speed_mps=9.5, energy_ml_per_km=46.0),
            1: _heldout(mean_speed_mps=10.5, energy_ml_per_km=48.0),
        },
    )
    verdict = evaluate_gate_g(opposed, _gate())
    assert verdict["conditions"]["c2_closed_loop_separation"] is False


def test_gate_g_condition_4_fails_when_both_deltas_rise() -> None:
    records = _records(
        {0: _heldout(), 1: _heldout()},
        {
            0: _heldout(mean_speed_mps=10.5, energy_ml_per_km=48.0),
            1: _heldout(mean_speed_mps=10.4, energy_ml_per_km=47.9),
        },
    )
    assert (
        evaluate_gate_g(records, _gate())["conditions"]["c4_direction_matches_objective"] is False
    )


def test_gate_g_condition_5_fails_on_collision_parking_or_route_collapse() -> None:
    collision = _records(
        {0: _heldout(), 1: _heldout()},
        {
            0: _heldout(mean_speed_mps=9.5, collision_count=1, out_of_road_count=2),
            1: _heldout(mean_speed_mps=9.4, energy_ml_per_km=46.0),
        },
    )
    assert evaluate_gate_g(collision, _gate())["conditions"]["c5_no_collapse_explanation"] is False

    parking = _records(
        {0: _heldout(), 1: _heldout()},
        {
            0: _heldout(mean_speed_mps=9.5, energy_ml_per_km=46.0),
            1: _heldout(mean_speed_mps=9.4, energy_ml_per_km=45.9, stopped_fraction=0.2),
        },
    )
    assert evaluate_gate_g(parking, _gate())["conditions"]["c5_no_collapse_explanation"] is False

    route = _records(
        {0: _heldout(), 1: _heldout()},
        {
            0: _heldout(mean_speed_mps=9.5, energy_ml_per_km=46.0),
            1: _heldout(mean_speed_mps=9.4, energy_ml_per_km=45.9, route_completion=0.75),
        },
    )
    assert evaluate_gate_g(route, _gate())["conditions"]["c5_no_collapse_explanation"] is False


def test_resolved_manifest_roundtrip_is_strict() -> None:
    raw = load_resolved_yaml_mapping(STUDY_PATH)
    raw["training_seeds"] = [0]
    with pytest.raises(ValueError, match="at least 2"):
        ObjectivePositiveControlStudyConfig.model_validate(raw)
