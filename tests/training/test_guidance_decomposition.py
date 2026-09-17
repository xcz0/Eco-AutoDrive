from __future__ import annotations

import json
from copy import deepcopy

import numpy as np
import pytest
from omegaconf import OmegaConf

from eco_planner._repository import CONFIG_ROOT
from eco_planner.analysis.decomposition import (
    ARM_NAMES,
    analyze_decomposition_episodes,
    episode_metrics,
)
from eco_planner.analysis.horizon import PLANNER_RESPONSE_CHECKPOINTS_S
from eco_planner.experiments.guidance.decomposition.diagnostics import (
    DecompositionConfig,
    SeedArms,
    analyze_episodes,
    design,
)
from eco_planner.experiments.guidance.decomposition.runner import _annotate, save_decisions

CHECKPOINT_STEPS = [round(value * 10) for value in PLANNER_RESPONSE_CHECKPOINTS_S]


def study() -> DecompositionConfig:
    return DecompositionConfig.model_validate(
        OmegaConf.to_container(
            OmegaConf.load(CONFIG_ROOT / "experiments/guidance/decomposition.yaml")
        )
    )


def step(speed: float) -> dict:
    distance = speed * 0.1
    intensity = 32.5 * np.exp(0.036 * speed)
    return {
        "dt_s": 0.1,
        "speed_mps": speed,
        "distance_m": distance,
        "fuel_ml": intensity * distance / 1000,
        "energy_score": np.exp(-intensity / 50),
        "progress_m": distance,
        "route_completion": 0.5,
        "position_error_m": 0.0,
        "planner_first_speed_mps": speed,
        "planner_mean_speed_mps": speed,
        "reference_first_speed_mps": 10.0,
        "target_first_delta_mps": speed - 10,
        "collision": False,
        "out_of_road": False,
        "arrive_dest": False,
        "max_step": False,
        "terminated": False,
        "truncated": False,
    }


def episodes(speed_fn) -> list[dict]:
    config = study()
    arms = design(config).arms
    rows = []
    for spec in config.seed_arms:
        for seed in config.noise_seeds:
            for scenario in ("0", "1"):
                for arm in ARM_NAMES:
                    speed = speed_fn(spec.seed, arm, scenario)
                    periods = 4
                    lateral, longitudinal = arms[spec.seed][arm]
                    rows.append(
                        {
                            "id": f"{spec.seed}-{scenario}-{arm}-{seed}",
                            "training_seed": spec.seed,
                            "scenario": scenario,
                            "map": "S",
                            "map_seed": int(scenario),
                            "arm": arm,
                            "noise_seed": seed,
                            "g_lat": lateral,
                            "g_lon": longitudinal,
                            "cycles": periods,
                            "status": "episode_terminated",
                            "steps": [step(speed) for _ in range(periods)],
                            "planner_cycles": [
                                {
                                    "plan_cycle": 0,
                                    "checkpoint_steps": CHECKPOINT_STEPS,
                                    "forward_displacement_m": [
                                        speed * (index + 1) for index in range(7)
                                    ],
                                    "first_speed_mps": speed,
                                    "mean_speed_mps": speed,
                                }
                            ],
                        }
                    )
    return rows


def small() -> DecompositionConfig:
    return study().model_copy(update={"required_scenarios": 1})


def speeds(deltas: dict[str, float]):
    return lambda seed, arm, scenario: 10.0 + deltas[arm]


def test_config_validation():
    config = study()
    assert config.training_seeds == [0, 1]
    assert config.noise_seeds == [760025]
    assert design(config).arms[1]["joint"] == (0.061707102750171675, 0.06364626040828059)
    for mutate in (
        lambda data: data.update(noise_seeds=[]),
        lambda data: data.update(training_seeds=[0]),
        lambda data: data["seed_arms"][0]["arms"].reverse(),
        lambda data: data["seed_arms"][0]["arms"][1].update(lateral=0.5),
        lambda data: data["seed_arms"][0]["arms"][2].update(longitudinal=0.5),
        lambda data: data["seed_arms"][0]["arms"][3].update(longitudinal=0.9),
        lambda data: data["seed_arms"][0]["arms"][0].update(lateral=1.5),
    ):
        data = deepcopy(config.model_dump())
        mutate(data)
        with pytest.raises(ValueError):
            DecompositionConfig.model_validate(data)


def test_corrected_dimension_order():
    config = study()
    for spec in config.seed_arms:
        arms = {arm.name: arm for arm in spec.arms}
        assert arms["lon"].lateral == arms["r0"].lateral
        assert arms["lon"].longitudinal != arms["r0"].longitudinal
        assert arms["lat"].longitudinal == arms["r0"].longitudinal
        assert arms["lat"].lateral != arms["r0"].lateral


@pytest.mark.parametrize(
    "deltas,verdict",
    [
        ({"r0": 0.0, "lon": 1.0, "lat": 0.0, "joint": 1.0}, "longitudinal-dominated"),
        ({"r0": 0.0, "lon": 0.0, "lat": 1.0, "joint": 1.0}, "lateral-dominated"),
        ({"r0": 0.0, "lon": 0.2, "lat": 0.2, "joint": 1.0}, "nonlinear-interaction"),
        ({"r0": 0.0, "lon": -0.6, "lat": -0.6, "joint": -1.0}, "not-reproduced-state-dependent"),
    ],
)
def test_attribution_verdicts(deltas, verdict):
    rows = episodes(speeds(deltas))
    result = analyze_episodes(rows, small(), ["0", "1"])
    assert result["verdict"]["per_seed"]["0"]["speed_mps"] == verdict
    assert result["verdict"]["per_seed"]["1"]["speed_mps"] == verdict
    assert result["verdict"]["overall"]["speed_mps"] == verdict


def test_effect_and_interaction_math():
    rows = episodes(speeds({"r0": 0, "lon": 2, "lat": 3, "joint": 7}))
    result = analyze_decomposition_episodes(rows, design(small()), ["0", "1"])
    metrics = result["metrics"]["0"]["speed_mps"]
    assert metrics["effects"]["lon"]["median"] == pytest.approx(2.0)
    assert metrics["effects"]["lat"]["median"] == pytest.approx(3.0)
    assert metrics["effects"]["joint"]["median"] == pytest.approx(7.0)
    assert metrics["interaction"]["median"] == pytest.approx(2.0)
    assert metrics["effects"]["lon"]["positive_count"] == 2


def test_variable_length_and_safety_accounting():
    rows = episodes(lambda seed, arm, scenario: 10.0)
    rows[0]["steps"] = []
    result = analyze_decomposition_episodes(rows, design(small()), ["0", "1"])
    assert result["episodes"][0]["metrics"] is None
    assert result["unsafe_or_incomplete_episodes"] == []
    rows = episodes(lambda seed, arm, scenario: 10.0)
    rows[0]["steps"][2]["collision"] = True
    result = analyze_decomposition_episodes(rows, design(small()), ["0", "1"])
    assert result["unsafe_or_incomplete_episodes"] == [rows[0]["id"]]
    rows = episodes(lambda seed, arm, scenario: 10.0)
    rows[0]["steps"][0]["fuel_ml"] *= 2
    result = analyze_decomposition_episodes(rows, design(small()), ["0", "1"])
    assert result["proxy_errors"] == [{"episode": rows[0]["id"], "step": 0}]
    with pytest.raises(ValueError, match="incomplete"):
        analyze_decomposition_episodes(rows[:-1], design(small()), ["0", "1"])


def test_annotate_maps_native_order():
    config = study()
    spec = config.seed_arms[0]
    rows = [{"id": "a", "g_lat": arm.lateral, "g_lon": arm.longitudinal} for arm in spec.arms]
    _annotate(rows, SeedArms.model_validate(spec.model_dump()))
    assert [row["arm"] for row in rows] == list(ARM_NAMES)
    assert all(row["training_seed"] == spec.seed for row in rows)


def test_prefix_and_first_waypoint_metrics():
    metrics = episode_metrics({"steps": [step(4.0)] + [step(10.0)] * 29})
    assert metrics is not None
    assert metrics["prefix_speed_mps"] == pytest.approx(np.mean([4.0] + [10.0] * 19))
    assert metrics["first_waypoint_speed_mps"] == pytest.approx(4.0)
    assert metrics["first_waypoint_distance_m"] == pytest.approx(0.4)
    assert metrics["step_count"] == 30


@pytest.mark.parametrize("figures", [False, True])
def test_offline_recompute_matches_live_statistics(tmp_path, figures):
    from eco_planner.analysis.runner import analyze
    from eco_planner.artifacts import write_json
    from tests.analysis.test_reports import assert_report

    source = tmp_path / "source"
    source.mkdir()
    rows = episodes(speeds({"r0": 0, "lon": 1, "lat": 0, "joint": 1}))
    names = ["0", "1"]
    config = small()
    write_json(source / "intervention_config.json", config.model_dump())
    write_json(source / "episodes.json", {"episodes": rows})
    write_json(source / "scenarios.json", {"scenarios": [{"name": name} for name in names]})
    save_decisions(rows, config, names, source)
    original_files = {p.name: p.read_bytes() for p in source.iterdir()}
    output = tmp_path / "analysis"
    returned = analyze("guidance-decomposition", source, output, figures=figures)
    result = json.loads((output / "summary.json").read_text())
    expected = analyze_episodes(rows, config, names)
    assert result == {"status": "completed", **expected}
    assert returned == {
        "status": "completed",
        "output_dir": str(output.resolve()),
        "verdict": expected["verdict"],
    }
    assert_report(output, figures=figures)
    assert {p.name: p.read_bytes() for p in source.iterdir()} == original_files
    payload = json.loads((output / "analysis.json").read_text())
    if figures:
        assert set(payload["figures"]) == {
            f"figures/{name}.{extension}"
            for name in ("effect-by-metric", "planner-response", "speed-interaction-scenarios")
            for extension in ("svg", "png")
        }
    else:
        assert payload["figures"] == []

    recorded = json.loads((source / "decisions.json").read_text())
    recorded["verdict"]["overall"]["speed_mps"] = "recorded_decision"
    write_json(source / "decisions.json", recorded)
    analyze("guidance-decomposition", source, tmp_path / "again", figures=False)
    again = json.loads((tmp_path / "again" / "summary.json").read_text())
    assert again["verdict"]["overall"]["speed_mps"] == "recorded_decision"
    assert (
        again["metrics"]["0"]["speed_mps"]["effects"]["lon"]["median"]
        == expected["metrics"]["0"]["speed_mps"]["effects"]["lon"]["median"]
    )
    (source / "decisions.json").unlink()
    with pytest.raises(FileNotFoundError, match="decisions.json"):
        analyze("guidance-decomposition", source, tmp_path / "missing", figures=False)


def test_decomposition_cli_routes(monkeypatch):
    from eco_planner.experiments.guidance.decomposition import runner
    from scripts import experiments as cli

    calls = []
    monkeypatch.setattr(runner, "run", lambda *a, **kw: calls.append((a, kw)))
    args = cli.build_parser().parse_args(
        ["guidance", "decomposition", "run", "--output-dir", "out", "--no-figures"]
    )
    assert args.config.is_file()
    cli.dispatch(args)
    assert calls == [((args.config, args.output_dir), {"figures": False})]
