from __future__ import annotations

import json

import numpy as np
import pytest
from omegaconf import OmegaConf

from eco_planner._repository import CONFIG_ROOT
from eco_planner.analysis.deferral import (
    analyze_deferral_episodes,
    zero_crossing_step,
)
from eco_planner.contracts import PLANNER_HORIZON
from eco_planner.experiments.guidance.deferral.diagnostics import (
    DeferralConfig,
    analyze_episodes,
    design,
)
from eco_planner.experiments.guidance.deferral.runner import save_decisions


def study() -> DeferralConfig:
    return DeferralConfig.model_validate(
        OmegaConf.to_container(OmegaConf.load(CONFIG_ROOT / "experiments/guidance/deferral.yaml"))
    )


def step(speed: float) -> dict:
    distance = speed * 0.1
    intensity = 32.5 * np.exp(0.036 * speed)
    return {
        "dt_s": 0.1,
        "speed_mps": speed,
        "distance_m": distance,
        "fuel_ml": intensity * distance / 1000,
        "collision": False,
        "out_of_road": False,
        "terminated": False,
        "truncated": False,
    }


def episodes(shape) -> list[dict]:
    config = study()
    rows = []
    for scenario in ("0", "1"):
        for noise in config.noise_seeds:
            for action in config.longitudinal_actions:
                cycles = []
                for cycle in range(config.total_window_steps):
                    base = 10.0 * cycle
                    values = [
                        base + action * shape(scenario, cycle, index)
                        for index in range(PLANNER_HORIZON)
                    ]
                    cycles.append(
                        {
                            "plan_cycle": cycle,
                            "checkpoint_steps": [1, 2, 5, 10, 20, 40, 80],
                            "forward_displacement_m": [
                                values[index] for index in (0, 1, 4, 9, 19, 39, 79)
                            ],
                            "waypoint_forward_displacement_m": values,
                            "first_speed_mps": 0.0,
                            "mean_speed_mps": 0.0,
                        }
                    )
                rows.append(
                    {
                        "id": f"{scenario}-{action}-{noise}",
                        "scenario": scenario,
                        "map": "S",
                        "map_seed": int(scenario),
                        "noise_seed": noise,
                        "g_lat": 0.0,
                        "g_lon": action,
                        "execution_horizon": 1,
                        "cycles": config.total_window_steps,
                        "status": "window_complete",
                        "steps": [step(10.0) for _ in range(config.total_window_steps)],
                        "planner_cycles": cycles,
                    }
                )
    return rows


def small(required: int = 1) -> DeferralConfig:
    return study().model_copy(update={"required_scenarios": required})


def test_config_validation():
    config = study()
    assert config.longitudinal_actions == [-1.0, -0.5, 0.0, 0.5, 1.0]
    assert config.execution_steps == 1
    assert config.total_window_steps == 20
    for field, value in (
        ("noise_seeds", [0]),
        ("lateral_action", 0.5),
        ("longitudinal_actions", [-1.0, 0.0, 1.0]),
        ("execution_steps", 2),
        ("majority_fraction", 0.0),
        ("crossing_tolerance_steps", -1),
    ):
        data = config.model_dump()
        data[field] = value
        with pytest.raises(ValueError):
            DeferralConfig.model_validate(data)
    assert design(config).execution_steps == 1


def test_zero_crossing_step():
    assert zero_crossing_step(np.array([-1.0, -0.5, 0.2, 0.4])) == 2
    assert zero_crossing_step(np.array([1.0, -1.0])) == 0
    assert zero_crossing_step(np.array([-1.0, -1.0])) is None


def test_repeated_deferral_verdict():
    names = ["0", "1"]
    deferred = episodes(lambda scenario, cycle, index: -1.0 if index == 0 else 1.0)
    result = analyze_episodes(deferred, small(), names)
    assert result["gate_c"]["status"] == "repeated_deferral"
    assert result["scenarios"]["0"]["crossing_median_step"] == 1.0
    assert result["scenarios"]["0"]["deferred_count"] == 20

    absent = episodes(lambda scenario, cycle, index: 1.0)
    result = analyze_episodes(absent, small(), names)
    assert result["gate_c"]["status"] == "deferral_absent"


def test_mixed_verdict_requires_scenario_majority():
    names = ["0", "1"]

    def shape(scenario, cycle, index):
        if scenario == "0":
            return -1.0 if index == 0 else 1.0
        return 1.0

    result = analyze_episodes(episodes(shape), small(required=2), names)
    assert result["gate_c"]["status"] == "mixed_or_inconclusive"
    assert result["gate_c"]["repeated_deferral_scenarios"] == ["0"]


def deferred_shape(scenario, cycle, index):
    return -1.0 if index == 0 else 1.0


def test_safety_incomplete_and_proxy_matrix():
    names = ["0", "1"]
    rows = episodes(deferred_shape)
    result = analyze_episodes(rows, small(), names)
    assert result["gate_c"]["safety_clean"]

    rows[0]["steps"][3]["collision"] = True
    result = analyze_episodes(rows, small(), names)
    assert not result["gate_c"]["safety_clean"]
    assert result["gate_c"]["status"] == "safety_or_proxy_failure"
    assert result["gate_c"]["unsafe_or_incomplete_episodes"] == [rows[0]["id"]]

    rows = episodes(deferred_shape)
    rows[0]["steps"][0]["fuel_ml"] *= 2
    result = analyze_episodes(rows, small(), names)
    assert result["gate_c"]["status"] == "safety_or_proxy_failure"

    rows = episodes(deferred_shape)
    with pytest.raises(ValueError, match="incomplete"):
        analyze_episodes(rows[:-1], small(), names)


def test_waypoint_response_requires_full_horizon():
    rows = episodes(lambda scenario, cycle, index: -1.0 if index == 0 else 1.0)
    rows[0]["planner_cycles"][0]["waypoint_forward_displacement_m"] = [0.0, 1.0]
    with pytest.raises(ValueError, match="PLANNER_HORIZON"):
        analyze_episodes(rows, small(), ["0", "1"])


def test_early_termination_uses_matched_common_cycles():
    rows = episodes(lambda scenario, cycle, index: -1.0 if index == 0 else 1.0)
    victim = next(
        row
        for row in rows
        if row["scenario"] == "0" and row["noise_seed"] == 0 and row["g_lon"] == -1.0
    )
    victim["planner_cycles"] = victim["planner_cycles"][:5]
    result = analyze_deferral_episodes(
        rows, design(small()), ["0", "1"], majority_fraction=0.5, crossing_tolerance_steps=1
    )
    assert result["scenarios"]["0"]["cycle_count"] == 5


@pytest.mark.parametrize("figures", [False, True])
def test_offline_recompute_matches_live_statistics(tmp_path, figures):
    from eco_planner.analysis.runner import analyze
    from eco_planner.artifacts import write_json
    from tests.analysis.test_reports import assert_report

    source = tmp_path / "source"
    source.mkdir()
    rows = episodes(lambda scenario, cycle, index: -1.0 if index == 0 else 1.0)
    names = ["0", "1"]
    config = small()
    write_json(source / "intervention_config.json", config.model_dump())
    write_json(source / "episodes.json", {"episodes": rows})
    write_json(source / "scenarios.json", {"scenarios": [{"name": name} for name in names]})
    save_decisions(rows, config, names, source)
    original_files = {p.name: p.read_bytes() for p in source.iterdir()}
    output = tmp_path / "analysis"
    returned = analyze("guidance-deferral", source, output, figures=figures)
    result = json.loads((output / "summary.json").read_text())
    expected = analyze_episodes(rows, config, names)
    assert result == {"status": "completed", **expected}
    assert returned == {
        "status": "completed",
        "output_dir": str(output.resolve()),
        "gate_c": expected["gate_c"],
    }
    assert_report(output, figures=figures)
    assert {p.name: p.read_bytes() for p in source.iterdir()} == original_files
    payload = json.loads((output / "analysis.json").read_text())
    if figures:
        assert set(payload["figures"]) == {
            f"figures/{name}.{extension}"
            for name in (
                "effect-by-cycle-and-time",
                "first-vs-full-by-cycle",
                "crossing-step-by-cycle",
            )
            for extension in ("svg", "png")
        }
    else:
        assert payload["figures"] == []

    recorded = json.loads((source / "decisions.json").read_text())
    recorded["gate_c"]["status"] = "recorded_decision"
    write_json(source / "decisions.json", recorded)
    analyze("guidance-deferral", source, tmp_path / "again", figures=False)
    again = json.loads((tmp_path / "again" / "summary.json").read_text())
    assert again["gate_c"]["status"] == "recorded_decision"
    assert (
        again["scenarios"]["0"]["median_first_waypoint_effect_m"]
        == expected["scenarios"]["0"]["median_first_waypoint_effect_m"]
    )
    (source / "decisions.json").unlink()
    with pytest.raises(FileNotFoundError, match="decisions.json"):
        analyze("guidance-deferral", source, tmp_path / "missing", figures=False)


def test_deferral_cli_routes(monkeypatch):
    from eco_planner.experiments.guidance.deferral import runner
    from scripts import experiments as cli

    calls = []
    monkeypatch.setattr(runner, "run", lambda *a, **kw: calls.append((a, kw)))
    args = cli.build_parser().parse_args(
        ["guidance", "deferral", "run", "--output-dir", "out", "--no-figures"]
    )
    assert args.config.is_file()
    cli.dispatch(args)
    assert calls == [((args.config, args.output_dir), {"figures": False})]
