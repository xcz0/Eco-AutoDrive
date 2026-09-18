from __future__ import annotations

import json
from copy import deepcopy

import numpy as np
import pytest
from omegaconf import OmegaConf

from eco_planner._repository import CONFIG_ROOT
from eco_planner.analysis.execution_bridge import (
    analyze_bridge_episodes,
    analyze_same_state,
    zero_crossing_step,
)
from eco_planner.analysis.horizon import PLANNER_RESPONSE_CHECKPOINTS_S
from eco_planner.evaluation.policy_intervention import _prediction_response
from eco_planner.experiments.guidance.execution_bridge.diagnostics import (
    ExecutionBridgeConfig,
    analyze_episodes,
)

CHECKPOINT_STEPS = [round(value * 10) for value in PLANNER_RESPONSE_CHECKPOINTS_S]


def study() -> ExecutionBridgeConfig:
    return ExecutionBridgeConfig.model_validate(
        OmegaConf.to_container(
            OmegaConf.load(CONFIG_ROOT / "experiments/guidance/execution-bridge.yaml")
        )
    )


def small() -> ExecutionBridgeConfig:
    return study().model_copy(update={"required_scenarios": 1})


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


def forward_curve(seed: int, arm: str) -> list[float]:
    base = 1.0 + seed
    sign = 1.0 if arm == "rstress" else -1.0
    return [sign * base * (index + 1) for index in range(7)]


def episode(
    seed: int,
    scenario: str,
    arm: str,
    horizon: int,
    noise: int,
    speed: float,
    periods: int = 4,
) -> dict:
    return {
        "id": f"{seed}-{scenario}-{arm}-{horizon}-{noise}",
        "training_seed": seed,
        "scenario": scenario,
        "map": "S",
        "map_seed": int(scenario),
        "arm": arm,
        "noise_seed": noise,
        "execution_horizon": horizon,
        "cycles": periods,
        "status": "episode_terminated",
        "steps": [step(speed) for _ in range(periods)],
        "planner_cycles": [
            {
                "plan_cycle": 0,
                "checkpoint_steps": CHECKPOINT_STEPS,
                "forward_displacement_m": forward_curve(seed, arm),
                "first_speed_mps": speed,
                "mean_speed_mps": speed,
                "guidance_action": [0.0, 0.0],
            }
        ],
    }


def episodes(speed_fn) -> list[dict]:
    config = small()
    rows = []
    for seed in (0, 1):
        for noise in config.noise_seeds:
            for scenario in ("0", "1"):
                for horizon in config.execution_horizons:
                    for arm in (config.reference_arm, config.counterfactual_arm):
                        rows.append(
                            episode(
                                seed,
                                scenario,
                                arm,
                                horizon,
                                noise,
                                speed_fn(seed, arm, horizon),
                            )
                        )
    return rows


def contexts(first_median: float) -> list[dict]:
    rows = []
    for seed in (0, 1):
        for scenario in ("0", "1"):
            forward_ref = [0.0] * 80
            forward_cf = [value + 0.5 for value in forward_ref]
            forward_cf[0] = first_median
            rows.append(
                {
                    "training_seed": seed,
                    "scenario": scenario,
                    "map": "S",
                    "map_seed": int(scenario),
                    "cycle": 0,
                    "reference_arm": "r0",
                    "counterfactual_arm": "rstress",
                    "reference_label": f"r0-seed-{seed}",
                    "counterfactual_label": f"rstress-seed-{seed}",
                    "reference_policy_hash": "a" * 64,
                    "counterfactual_policy_hash": "b" * 64,
                    "g_lat_reference": 0.0,
                    "g_lon_reference": -0.02,
                    "g_lat_counterfactual": 0.02,
                    "g_lon_counterfactual": 0.08,
                    "forward_reference_m": forward_ref,
                    "forward_counterfactual_m": forward_cf,
                    "lateral_reference_m": [0.0] * 7,
                    "lateral_counterfactual_m": [0.1] * 7,
                }
            )
    return rows


def test_config_validation():
    config = study()
    assert config.execution_horizons == [1, 2, 5]
    assert config.reference_arm == "r0"
    for mutate in (
        lambda data: data.update(noise_seeds=[]),
        lambda data: data.update(execution_horizons=[1]),
        lambda data: data.update(execution_horizons=[2, 4]),
        lambda data: data["runs"][1].update(arm="r0"),
        lambda data: data["runs"][0].update(reward_profile="plannerrft_energy_band_lam64_v1"),
        lambda data: data["runs"][0].update(label="rstress-seed-0"),
        lambda data: data.update(part_b_majority_fraction=0.0),
    ):
        data = deepcopy(config.model_dump())
        mutate(data)
        with pytest.raises(ValueError):
            ExecutionBridgeConfig.model_validate(data)


def test_part_a_crossover_confirmed():
    # Rstress is slower than R0 at 0.1 s but faster at 0.5 s, for both seeds.
    def speed_fn(seed, arm, horizon):
        base = 10.0 + seed
        if arm == "r0":
            return base
        return base - 0.5 if horizon == 1 else base + 1.0

    config = small()
    result = analyze_episodes(episodes(speed_fn), [], config, ["0", "1"])
    gate = result["gate"]
    assert gate["part_a"]["directions"]["0"]["speed_mps"]["1"] == "negative"
    assert gate["part_a"]["directions"]["0"]["speed_mps"]["5"] == "positive"
    assert gate["part_a"]["status"] == "crossover_confirmed"
    assert gate["verdict"] == "execution_contract_causal_crossover_confirmed"


def test_part_a_amplifies_not_reverses():
    def speed_fn(seed, arm, horizon):
        base = 10.0 + seed
        if arm == "r0":
            return base
        return base + (0.2 if horizon == 1 else 1.0)

    config = small()
    result = analyze_episodes(episodes(speed_fn), [], config, ["0", "1"])
    assert result["gate"]["part_a"]["status"] == "amplifies_not_reverses"
    assert result["gate"]["verdict"] == "execution_contract_amplifies_but_does_not_reverse"


def test_part_b_local_temporal_bridge():
    def speed_fn(seed, arm, horizon):
        base = 10.0 + seed
        return base + (0.2 if (arm == "rstress" and horizon > 1) else 0.0)

    config = small()
    rows = contexts(0.0)
    result = analyze_episodes(episodes(speed_fn), rows, config, ["0", "1"])
    assert result["same_state"]["context_count"] == 4
    assert result["gate"]["part_b"]["status"] == "local_temporal_bridge"
    assert result["gate"]["part_b"]["seeds"]["0"]["delta_g_lon_positive_majority"]


def test_part_b_local_response_differs_verdict():
    # No Part A crossover or amplification (effect shrinks with the prefix), and a
    # positive first-waypoint local response.
    def speed_fn(seed, arm, horizon):
        base = 10.0 + seed
        if arm != "rstress":
            return base
        return base + (0.5 if horizon == 1 else 0.2)

    config = small()
    result = analyze_episodes(episodes(speed_fn), contexts(0.5), config, ["0", "1"])
    assert result["gate"]["part_b"]["status"] == "local_response_differs"
    assert (
        result["gate"]["verdict"]
        == "learned_policy_local_response_differs_from_manual_full_range_sweep"
    )


def test_same_state_analysis_and_zero_crossing():
    result = analyze_same_state(contexts(0.0))
    row = result["seeds"]["0"]
    assert row["delta_g_lon"]["positive_count"] == 2
    assert row["delta_forward"]["0.1"]["median"] == 0.0
    assert row["delta_forward"]["0.5"]["median"] == 0.5
    assert zero_crossing_step(np.array([-1.0, 0.0, 0.0, 2.0])) == 3
    assert zero_crossing_step(np.array([-1.0, -2.0])) is None


def test_prediction_response_is_along_heading():
    prediction = np.zeros((1, 5, 4), dtype=np.float64)
    prediction[0, :, 0] = [0.0, 0.1, 0.2, 0.3, 0.4]
    prediction[0, :, 1] = 0.0
    prediction[0, :, 2] = 1.0
    prediction[0, :, 3] = 0.0
    forward, lateral = _prediction_response(prediction)
    np.testing.assert_allclose(forward, [0.0, 0.1, 0.2, 0.3, 0.4])
    np.testing.assert_allclose(lateral, np.zeros(5))


def _write_source(tmp_path, rows, same_state_rows):
    from eco_planner.analysis.io import write_json

    source = tmp_path / "source"
    source.mkdir()
    config = small()
    write_json(
        source / "intervention_config.json",
        config.model_dump(mode="json"),
    )
    write_json(source / "episodes.json", {"episodes": rows})
    write_json(
        source / "scenarios.json",
        {"scenarios": [{"name": "0", "map": "S", "seed": 0}, {"name": "1", "map": "S", "seed": 1}]},
    )
    write_json(source / "same_state.json", {"contexts": same_state_rows})
    write_json(
        source / "decisions.json",
        {"gate": analyze_episodes(rows, same_state_rows, config, ["0", "1"])["gate"]},
    )
    return source


def test_offline_recompute_matches_live(tmp_path):
    from eco_planner.analysis.runner import analyze
    from eco_planner.analysis.statistics import statistics  # noqa: F401  (import surface smoke)

    def speed_fn(seed, arm, horizon):
        base = 10.0 + seed
        return base + (0.2 if (arm == "rstress" and horizon > 1) else 0.0)

    rows = episodes(speed_fn)
    same_state_rows = contexts(0.0)
    source = _write_source(tmp_path, rows, same_state_rows)
    analyze("guidance-execution-bridge", source, tmp_path / "report", figures=False)
    summary = json.loads((tmp_path / "report" / "summary.json").read_text(encoding="utf-8"))
    expected = analyze_episodes(rows, same_state_rows, small(), ["0", "1"])
    assert summary["gate"] == expected["gate"]
    assert summary["episode_count"] == len(rows)
    assert summary["metrics"]["0"]["1"]["speed_mps"]["effects"]["rstress"]["median"] == 0.0


def test_cli_routes(monkeypatch):
    from eco_planner.experiments.guidance.execution_bridge import runner
    from scripts import experiments as cli

    calls = []
    monkeypatch.setattr(runner, "run", lambda *a, **kw: calls.append((a, kw)))
    args = cli.build_parser().parse_args(
        [
            "guidance",
            "execution-bridge",
            "run",
            "--output-dir",
            "out",
            "--source-dir",
            "src",
            "--no-figures",
        ]
    )
    assert args.config.is_file()
    cli.dispatch(args)
    assert calls == [((args.source_dir, args.config, args.output_dir), {"figures": False})]
    analyze_args = cli.build_parser().parse_args(
        [
            "guidance",
            "execution-bridge",
            "analyze",
            "--output-dir",
            "out",
            "--source-dir",
            "src",
        ]
    )
    assert analyze_args.key == ("guidance", "execution-bridge", "analyze")


def test_bridge_episodes_require_matched_matrix():
    def speed_fn(seed, arm, horizon):
        return 10.0

    rows = episodes(speed_fn)
    config = small()
    analyze_bridge_episodes(
        rows,
        execution_horizons=config.execution_horizons,
        noise_seeds=config.noise_seeds,
        reference_arm=config.reference_arm,
        counterfactual_arm=config.counterfactual_arm,
        scenario_names=["0", "1"],
    )
    broken = deepcopy(rows)
    broken[0]["planner_cycles"][0]["forward_displacement_m"][0] += 1.0
    with pytest.raises(ValueError, match="not matched"):
        analyze_bridge_episodes(
            broken,
            execution_horizons=config.execution_horizons,
            noise_seeds=config.noise_seeds,
            reference_arm=config.reference_arm,
            counterfactual_arm=config.counterfactual_arm,
            scenario_names=["0", "1"],
        )
    with pytest.raises(ValueError, match="incomplete"):
        analyze_bridge_episodes(
            rows[:-1],
            execution_horizons=config.execution_horizons,
            noise_seeds=config.noise_seeds,
            reference_arm=config.reference_arm,
            counterfactual_arm=config.counterfactual_arm,
            scenario_names=["0", "1"],
        )
