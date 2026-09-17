from __future__ import annotations

import json
from copy import deepcopy

import numpy as np
import pytest
from omegaconf import OmegaConf

from eco_planner._repository import CONFIG_ROOT
from eco_planner.analysis.horizon import (
    PLANNER_RESPONSE_CHECKPOINTS_S,
    analyze_horizon_episodes,
)
from eco_planner.experiments.guidance.horizon.diagnostics import (
    HorizonInterventionConfig,
    analyze_episodes,
)
from eco_planner.experiments.guidance.horizon.runner import save_decisions

CHECKPOINT_STEPS = [round(value * 10) for value in PLANNER_RESPONSE_CHECKPOINTS_S]


def study() -> HorizonInterventionConfig:
    return HorizonInterventionConfig.model_validate(
        OmegaConf.to_container(OmegaConf.load(CONFIG_ROOT / "experiments/guidance/horizon.yaml"))
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
        "position_error_m": 0.0,
        "planner_first_speed_mps": speed,
        "planner_mean_speed_mps": speed,
        "reference_first_speed_mps": 10.0,
        "target_first_delta_mps": speed - 10,
        "collision": False,
        "out_of_road": False,
        "terminated": False,
        "truncated": False,
    }


def episodes(scale):
    config = study()
    rows = []
    for scenario in range(2):
        for horizon in config.execution_horizons:
            cycles = config.total_window_steps // horizon
            for action in config.longitudinal_actions:
                for noise in config.noise_seeds:
                    speed = 10 + action * scale(horizon)
                    rows.append(
                        {
                            "id": f"{scenario}-{horizon}-{action}-{noise}",
                            "scenario": str(scenario),
                            "map": "S",
                            "map_seed": scenario,
                            "noise_seed": noise,
                            "g_lon": action,
                            "execution_horizon": horizon,
                            "cycles": cycles,
                            "status": "window_complete",
                            "steps": [step(speed) for _ in range(config.total_window_steps)],
                            "planner_cycles": [
                                {
                                    "plan_cycle": cycle,
                                    "checkpoint_steps": CHECKPOINT_STEPS,
                                    "forward_displacement_m": [
                                        action * (index + 1) for index in range(7)
                                    ],
                                    "first_speed_mps": action,
                                    "mean_speed_mps": action,
                                }
                                for cycle in range(cycles)
                            ],
                        }
                    )
    return rows


def test_config_validation():
    config = study()
    assert config.execution_horizons == [1, 2, 5, 10, 20]
    assert config.total_window_steps == 20
    for field, value in (
        ("noise_seeds", [0]),
        ("lateral_action", 0.5),
        ("longitudinal_actions", [-1.0, 0.0, 1.0]),
        ("execution_horizons", [1, 2, 3]),
        ("execution_horizons", [1, 2, 5]),
        ("execution_horizons", [1, 2, 5, 10, 100]),
    ):
        data = config.model_dump()
        data[field] = value
        with pytest.raises(ValueError):
            HorizonInterventionConfig.model_validate(data)


def test_horizon_verdict_tracks_executed_speed_direction():
    names = [str(i) for i in range(2)]
    small = study().model_copy(update={"required_scenarios": 1})

    mismatch = analyze_episodes(episodes(lambda h: -1.0 if h == 1 else 1.0), small, names)
    assert mismatch["gate_a"]["status"] == "strong_evidence_for_receding_horizon_mismatch"
    assert mismatch["gate_a"]["executed_speed_direction"]["1"] == "negative"
    assert mismatch["gate_a"]["executed_speed_direction"]["2"] == "positive"

    unchanged = analyze_episodes(episodes(lambda h: -1.0), small, names)
    assert unchanged["gate_a"]["status"] == "sign_unchanged_by_horizon"

    mixed = analyze_episodes(episodes(lambda h: 1.0), small, names)
    assert mixed["gate_a"]["status"] == "mixed_or_inconclusive"


def test_horizon_safety_and_incomplete_matrix():
    names = [str(i) for i in range(2)]
    small = study().model_copy(update={"required_scenarios": 1})
    rows = episodes(lambda h: -1.0 if h == 1 else 1.0)
    result = analyze_episodes(rows, small, names)
    assert result["gate_a"]["safety_clean"]
    rows[0]["steps"][3]["collision"] = True
    result = analyze_episodes(rows, small, names)
    assert not result["gate_a"]["safety_clean"]
    assert result["gate_a"]["status"] == "safety_or_proxy_failure"
    assert result["gate_a"]["unsafe_or_incomplete_episodes"] == [rows[0]["id"]]
    with pytest.raises(ValueError, match="incomplete"):
        analyze_episodes(rows[:-1], small, names)


@pytest.mark.parametrize("figures", [False, True])
def test_offline_recompute_matches_live_statistics(tmp_path, figures):
    from eco_planner.analysis.runner import analyze
    from eco_planner.artifacts import write_json
    from tests.analysis.test_reports import assert_report

    source = tmp_path / "source"
    source.mkdir()
    rows = episodes(lambda h: -1.0 if h == 1 else 1.0)
    names = [str(i) for i in range(2)]
    small = study().model_copy(update={"required_scenarios": 1})
    write_json(source / "intervention_config.json", small.model_dump())
    write_json(source / "episodes.json", {"episodes": rows})
    write_json(source / "scenarios.json", {"scenarios": [{"name": name} for name in names]})
    save_decisions(rows, small, names, source)
    original_files = {p.name: p.read_bytes() for p in source.iterdir()}
    output = tmp_path / "analysis"
    returned = analyze("guidance-horizon", source, output, figures=figures)
    result = json.loads((output / "summary.json").read_text())
    expected = analyze_episodes(rows, small, names)
    assert result == {"status": "completed", **expected}
    assert returned == {
        "status": "completed",
        "output_dir": str(output.resolve()),
        "gate_a": expected["gate_a"],
    }
    assert_report(output, figures=figures)
    assert {p.name: p.read_bytes() for p in source.iterdir()} == original_files
    payload = json.loads((output / "analysis.json").read_text())
    if figures:
        assert set(payload["figures"]) == {
            f"figures/{name}.{extension}"
            for name in ("effect-by-horizon", "horizon-speed-response", "planner-response")
            for extension in ("svg", "png")
        }
    else:
        assert payload["figures"] == []

    recorded = json.loads((source / "decisions.json").read_text())
    recorded["gate_a"]["status"] = "recorded_decision"
    recorded["horizons"]["1"]["direction"] = "positive"
    recorded["horizons"]["1"]["metrics"]["speed_mps"]["scenarios"]["0"]["passed"] = False
    write_json(source / "decisions.json", recorded)
    analyze("guidance-horizon", source, tmp_path / "again", figures=False)
    again = json.loads((tmp_path / "again" / "summary.json").read_text())
    assert again["gate_a"]["status"] == "recorded_decision"
    assert again["horizons"]["1"]["direction"] == "positive"
    assert not again["horizons"]["1"]["metrics"]["speed_mps"]["scenarios"]["0"]["passed"]
    assert (
        again["horizons"]["1"]["metrics"]["speed_mps"]["scenarios"]["0"]["effect"]
        == expected["horizons"]["1"]["metrics"]["speed_mps"]["scenarios"]["0"]["effect"]
    )
    (source / "decisions.json").unlink()
    with pytest.raises(FileNotFoundError, match="decisions.json"):
        analyze("guidance-horizon", source, tmp_path / "missing", figures=False)


def test_horizon_cli_routes(monkeypatch):
    from eco_planner.experiments.guidance.horizon import runner
    from scripts import experiments as cli

    calls = []
    monkeypatch.setattr(runner, "run", lambda *a, **kw: calls.append((a, kw)))
    args = cli.build_parser().parse_args(
        ["guidance", "horizon", "run", "--output-dir", "out", "--no-figures"]
    )
    assert args.config.is_file()
    cli.dispatch(args)
    assert calls == [((args.config, args.output_dir), {"figures": False})]


def test_horizon_planner_response_is_matched_across_horizons():
    names = [str(i) for i in range(2)]
    small = study().model_copy(update={"required_scenarios": 1})
    rows = episodes(lambda h: -1.0 if h == 1 else 1.0)
    reference = deepcopy(rows)
    analyze_horizon_episodes(reference, small, names)  # sanity: consistent data passes
    for row in rows:
        if row["execution_horizon"] == 2:
            row["planner_cycles"][0]["forward_displacement_m"][0] += 1.0
    with pytest.raises(ValueError, match="not matched"):
        analyze_horizon_episodes(rows, small, names)
