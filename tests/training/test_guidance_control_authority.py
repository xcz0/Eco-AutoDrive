from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from lightning.fabric import Fabric
from omegaconf import OmegaConf
from tensordict import TensorDict

from eco_planner._repository import CONFIG_ROOT
from eco_planner.analysis.guidance import aggregate
from eco_planner.evaluation.inference.runtime import (
    FabricInferenceRuntime,
    validate_manual_guidance,
)
from eco_planner.experiments.guidance.control_authority.config import InterventionConfig
from eco_planner.experiments.guidance.control_authority.diagnostics import (
    analyze_episodes,
    matched_statistics,
)
from eco_planner.experiments.guidance.control_authority.runner import collect_group, save_decisions
from eco_planner.models import PlannerInferenceResult, parse_guidance_config
from eco_planner.models.guidance import zero_guidance_diagnostics
from eco_planner.rl.reward.config import EnergyRewardConfig
from eco_planner.runtime.envs import VectorEnvScenario


def study():
    return InterventionConfig.model_validate(
        OmegaConf.to_container(
            OmegaConf.load(CONFIG_ROOT / "experiments/guidance/control-authority/intervention.yaml")
        )
    )


def guidance():
    return parse_guidance_config(
        OmegaConf.load(CONFIG_ROOT / "components/guidance/orthogonal_policy.yaml")
    )


def step(speed):
    distance = speed * 0.1
    intensity = 32.5 * np.exp(0.036 * speed)
    return {
        "dt_s": 0.1,
        "speed_mps": speed,
        "distance_m": distance,
        "fuel_ml": intensity * distance / 1000,
        "energy_score": np.exp(-intensity / 50),
        "progress_m": distance,
        "planner_first_speed_mps": speed,
        "planner_mean_speed_mps": speed,
        "reference_first_speed_mps": 10.0,
        "target_first_delta_mps": speed - 10,
        "collision": False,
        "out_of_road": False,
        "terminated": False,
        "truncated": False,
    }


def episodes():
    rows = []
    for scenario in range(16):
        for a in study().longitudinal_actions:
            for n in study().noise_seeds:
                rows.append(
                    {
                        "id": f"{scenario}-{a}-{n}",
                        "scenario": str(scenario),
                        "map": "S",
                        "map_seed": scenario,
                        "noise_seed": n,
                        "g_lon": a,
                        "status": "window_complete",
                        "steps": [step(10 + a) for _ in range(20)],
                    }
                )
    return rows


def test_matched_statistics_and_zero_noise_boundary():
    config = study()
    values = np.repeat(np.asarray(config.longitudinal_actions)[:, None], 3, axis=1)
    result = matched_statistics(values, config)
    assert result["passed"] and result["rho"] == pytest.approx(1)
    assert result["noise_scale"] == 0 and result["effect"] == 2
    constant = matched_statistics(np.ones((5, 3)), config)
    assert not constant["passed"] and constant["rho"] is None
    noisy = matched_statistics(values + [-10, 0, 10], config)
    assert noisy["noise_scale"] == 10 and not noisy["passed"]


def test_weighted_energy_window_and_immediate():
    steps = [step(2), step(20)]
    total = aggregate(steps)
    assert total["energy_ml_per_km"] == pytest.approx(
        sum(s["fuel_ml"] for s in steps) * 1000 / sum(s["distance_m"] for s in steps)
    )
    assert total["speed_mps"] == 11
    assert aggregate(steps[:1])["speed_mps"] == 2
    assert total["energy_ml_per_km"] != pytest.approx(
        np.mean([s["fuel_ml"] * 1000 / s["distance_m"] for s in steps])
    )


def test_complete_gate_and_safety_without_sample_dropping():
    rows = episodes()
    names = [str(i) for i in range(16)]
    result = analyze_episodes(rows, study(), names)
    assert result["gate_d"]["passed"]
    assert result["transition_count"] == 4800
    rows[0]["steps"][0]["collision"] = True
    result = analyze_episodes(rows, study(), names)
    assert not result["gate_d"]["passed"]
    assert result["gate_d"]["unsafe_or_incomplete_episodes"] == [rows[0]["id"]]
    assert result["episode_count"] == 240


def test_direction_majority_and_constant_planner():
    rows = episodes()
    names = [str(i) for i in range(16)]
    for e in rows:
        if int(e["scenario"]) >= 8:
            e["steps"] = [step(10 - e["g_lon"]) for _ in range(20)]
    result = analyze_episodes(rows, study(), names)
    assert not result["gate_d"]["numerical_passed"]
    rows = episodes()
    for e in rows:
        for s in e["steps"]:
            s["planner_first_speed_mps"] = 10.0
    result = analyze_episodes(rows, study(), names)
    assert result["gate_d"]["attribution"] == "guidance_injection_or_frozen_planner_authority"


def test_temporal_reversal_and_incomplete_matrix():
    rows = episodes()
    for e in rows:
        e["steps"][0] = step(10 - e["g_lon"])
    result = analyze_episodes(rows, study(), [str(i) for i in range(16)])
    assert not result["gate_d"]["passed"]
    assert result["gate_d"]["temporal_contradictions"]
    with pytest.raises(ValueError, match="incomplete"):
        analyze_episodes(rows[:-1], study(), [str(i) for i in range(16)])


def test_offline_recompute_matches_live_statistics(tmp_path):
    from eco_planner.analysis.runner import analyze
    from eco_planner.artifacts import write_json

    source = tmp_path / "source"
    source.mkdir()
    rows = episodes()
    write_json(source / "intervention_config.json", study().model_dump())
    write_json(source / "episodes.json", {"episodes": rows})
    write_json(source / "scenarios.json", {"scenarios": [{"name": str(i)} for i in range(16)]})
    save_decisions(rows, study(), [str(i) for i in range(16)], source)
    output = tmp_path / "analysis"
    analyze("guidance-control-authority", source, output, figures=False)
    result = json.loads((output / "summary.json").read_text())
    expected = analyze_episodes(rows, study(), [str(i) for i in range(16)])
    assert result == {"status": "completed", **expected}
    assert (output / "report.md").is_file()

    # Offline rendering preserves recorded decisions without applying thresholds again.
    recorded = json.loads((source / "decisions.json").read_text())
    recorded["gate_d"]["passed"] = not recorded["gate_d"]["passed"]
    recorded["gate_d"]["attribution"] = "recorded_decision"
    recorded["windows"]["short_horizon"]["speed_mps"]["scenarios"]["0"]["passed"] = False
    write_json(source / "decisions.json", recorded)
    analyze("guidance-control-authority", source, tmp_path / "again", figures=False)
    again = json.loads((tmp_path / "again" / "summary.json").read_text())
    assert again["gate_d"] == recorded["gate_d"]
    assert not again["windows"]["short_horizon"]["speed_mps"]["scenarios"]["0"]["passed"]
    assert (
        again["windows"]["short_horizon"]["speed_mps"]["scenarios"]["0"]["effect"]
        == (expected["windows"]["short_horizon"]["speed_mps"]["scenarios"]["0"]["effect"])
    )
    (source / "decisions.json").unlink()
    with pytest.raises(FileNotFoundError, match="decisions.json"):
        analyze("guidance-control-authority", source, tmp_path / "missing", figures=False)


def test_manual_cli_and_config(monkeypatch):
    from eco_planner.experiments.guidance.control_authority import runner
    from scripts.experiments import __main__ as cli

    calls = []
    monkeypatch.setattr(runner, "run", lambda *a, **kw: calls.append((a, kw)))
    args = cli.build_parser().parse_args(
        [
            "guidance",
            "control-authority",
            "run",
            "--output-dir",
            "out",
            "--no-figures",
        ]
    )
    assert args.config.is_file()
    cli.dispatch(args)
    assert calls == [((args.config, args.output_dir), {"figures": False})]
    for field, value in (
        ("noise_seeds", [0]),
        ("lateral_action", 0.5),
        ("longitudinal_actions", [-1.0, 0.0, 1.0]),
    ):
        data = study().model_dump()
        data[field] = value
        with pytest.raises(ValueError):
            InterventionConfig.model_validate(data)


@pytest.mark.parametrize(
    "action,error",
    [
        (torch.tensor([[0.0, 1.01]]), ValueError),
        (torch.tensor([[0.0, float("nan")]]), ValueError),
        (torch.zeros(2), ValueError),
        (torch.zeros((1, 2), dtype=torch.float64), TypeError),
    ],
)
def test_manual_action_boundary(action, error):
    with pytest.raises(error):
        validate_manual_guidance(action, 1, torch.device("cpu"), guidance())


class AnalyticPlanner(torch.nn.Module):
    def forward(self, observation, noise, generators, guidance_action):
        assert torch.is_grad_enabled()
        self.last_action = guidance_action.clone()
        batch = len(guidance_action)
        prediction = torch.zeros((batch, 1, 80, 4))
        prediction[..., 2] = 1.0
        prediction[:, 0, :, 0] = (3 + guidance_action[:, 1, None]) * torch.arange(1, 81) * 0.1
        reference = prediction.clone()
        reference[:, 0, :, 0] = 3 * torch.arange(1, 81) * 0.1
        diagnostics = zero_guidance_diagnostics(
            guidance(), guidance_action, future_len=80, num_steps=5
        )
        diagnostics.longitudinal_target_speed_delta_mps = guidance_action[:, 1, None].expand(-1, 80)
        return PlannerInferenceResult(prediction, reference, guidance_action, diagnostics)


def analytic_runtime():
    return FabricInferenceRuntime(
        Fabric(accelerator="cpu", devices=1, precision="32-true"),
        AnalyticPlanner(),
        SimpleNamespace(predicted_neighbor_num=0, future_len=80, route_num=25),
        None,
        SimpleNamespace(seed=0),
        SimpleNamespace(num_steps=5),
        guidance(),
    )


def test_runtime_forwards_endpoints_without_global_rng_consumption():
    runtime = analytic_runtime()
    observation = TensorDict(
        {
            "ego_current_state": torch.zeros((2, 10)),
            "route_lanes_speed_limit": torch.zeros((2, 25, 1)),
            "route_lanes_has_speed_limit": torch.zeros((2, 25, 1), dtype=torch.bool),
        },
        batch_size=[2],
    )
    actions = torch.tensor([[0.0, -1.0], [0.0, 1.0]])
    generators = tuple(torch.Generator().manual_seed(0) for _ in range(2))
    state = torch.get_rng_state().clone()
    decision = runtime.infer_batch(
        observation, runtime.sample_noise(generators), generators, guidance_action=actions
    )
    assert torch.equal(torch.get_rng_state(), state)
    assert torch.equal(decision.audit_result()["guidance_action"], actions)
    assert decision.ego_trajectories[:, 0, 0] == pytest.approx([0.2, 0.4])


@pytest.mark.simulator
@pytest.mark.parametrize("horizon", [21, 3])
def test_real_rollout_intervention_window_and_noise_pairing(tmp_path, horizon):
    from eco_planner.contracts import ExecutionMode
    from eco_planner.runtime.envs import VectorMetaDriveEnv

    env_config = OmegaConf.to_container(OmegaConf.load(CONFIG_ROOT / "components/env.yaml"))
    env_config.update(map="S", horizon=horizon, num_scenarios=1)
    scenario = VectorEnvScenario("S0", "S", 0)
    env = VectorMetaDriveEnv(
        (env_config,),
        mode="no_traffic",
        execution_mode=ExecutionMode.ROLLOUT,
        map_query_radius_m=100.0,
        history_warmup_steps=0,
        scenarios=(scenario,),
        torch_threads_per_worker=1,
    )
    try:
        all_rows = []
        for group, noise_seed in enumerate(study().noise_seeds):
            all_rows.extend(
                collect_group(
                    env,
                    analytic_runtime(),
                    (scenario,),
                    noise_seed,
                    study(),
                    EnergyRewardConfig(reference_ml_per_km=50.0, minimum_step_distance_m=0.001),
                    tmp_path,
                    group,
                )
            )
        rows = all_rows[:5]
    finally:
        env.close()
    if horizon == 3:
        assert all(e["status"] == "episode_terminated" and len(e["steps"]) < 20 for e in rows)
        return
    assert all(e["status"] == "window_complete" and len(e["steps"]) == 20 for e in rows)
    assert aggregate(rows[0]["steps"])["duration_s"] == pytest.approx(2.0)
    for arm in range(1, 5):
        with np.load(tmp_path / "raw" / "group-000-arm-0" / "cycle-19.npz") as a:
            with np.load(tmp_path / "raw" / f"group-000-arm-{arm}" / "cycle-19.npz") as b:
                np.testing.assert_array_equal(a["initial_noise"], b["initial_noise"])
    assert aggregate(rows[-1]["steps"])["speed_mps"] > aggregate(rows[0]["steps"])["speed_mps"]
    saved = json.loads((tmp_path / "raw/group-000-arm-4/episodes.json").read_text())
    assert saved["episodes"][0]["status"] == "window_complete"

    from eco_planner.analysis.runner import analyze
    from eco_planner.artifacts import write_json

    source = tmp_path / "source"
    source.mkdir()
    config = study().model_copy(update={"required_scenarios": 1})
    write_json(source / "intervention_config.json", config.model_dump())
    write_json(source / "episodes.json", {"episodes": all_rows})
    write_json(source / "scenarios.json", {"scenarios": [{"name": scenario.name}]})
    save_decisions(all_rows, config, [scenario.name], source)
    analyze("guidance-control-authority", source, tmp_path / "report", figures=False)
    result = json.loads((tmp_path / "report/summary.json").read_text())
    assert result == {
        "status": "completed",
        **analyze_episodes(all_rows, config, [scenario.name]),
    }
