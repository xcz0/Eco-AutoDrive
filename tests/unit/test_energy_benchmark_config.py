from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import run_energy_matrix
from omegaconf import OmegaConf
from run_energy_matrix import _load_matrix, _validate_resolved_config

ROOT = Path(__file__).resolve().parents[2]


def test_energy_matrix_declares_fixed_coverage_and_guidance_profiles() -> None:
    matrix = _load_matrix(ROOT / "configs" / "benchmark" / "energy_matrix.yaml")

    assert matrix["planner_noise_seed"] == 0
    assert matrix["sampler"] == {"name": "ddim5", "ddim_stochasticity": 0.0}
    assert matrix["energy_metric"]["name"] == "metadrive_fuel_proxy"
    assert [profile["longitudinal_scale"] for profile in matrix["guidance_profiles"]] == [
        None,
        -1.0,
        0.0,
        1.0,
    ]


def test_energy_matrix_resolved_contract_matches_structures_config() -> None:
    matrix = _load_matrix(ROOT / "configs" / "benchmark" / "energy_matrix.yaml")
    job = matrix["jobs"][0]
    resolved = OmegaConf.to_container(
        OmegaConf.create(
            {
                "runtime": {"seed": 0},
                "sampler": {"name": "ddim5", "ddim_stochasticity": 0.0},
                "scenarios": [
                    {"name": item["name"], "map": item["map"], "seed": item["seed"]}
                    for item in job["scenarios"]
                ],
                "env": {
                    "trajectory_execution_steps": 5,
                    "decision_repeat": 5,
                    "physics_world_step_size": 0.02,
                    "random_agent_model": False,
                },
            }
        ),
        resolve=True,
    )
    assert isinstance(resolved, dict)

    _validate_resolved_config(resolved, matrix, job)


def test_energy_matrix_traffic_scenario_matches_hydra_config() -> None:
    matrix = _load_matrix(ROOT / "configs" / "benchmark" / "energy_matrix.yaml")
    job = next(item for item in matrix["jobs"] if item["id"] == "traffic_follow")
    scenario = job["scenarios"][0]
    resolved = OmegaConf.to_container(
        OmegaConf.load(ROOT / "configs" / "experiment" / "evaluate_energy_traffic.yaml"),
        resolve=False,
    )
    assert isinstance(resolved, dict)

    hydra_scenario = resolved["scenarios"][0]
    assert scenario["name"] == hydra_scenario["name"] == "traffic_follow"
    assert scenario["map"] == hydra_scenario["map"] == "S" * 40
    assert scenario["seed"] == hydra_scenario["seed"] == 5


def test_energy_matrix_rejects_resolved_sampler_drift() -> None:
    matrix = _load_matrix(ROOT / "configs" / "benchmark" / "energy_matrix.yaml")
    job = matrix["jobs"][0]
    resolved = {
        "runtime": {"seed": 0},
        "sampler": {"name": "dpm10", "ddim_stochasticity": 0.0},
        "scenarios": [
            {"name": item["name"], "map": item["map"], "seed": item["seed"]}
            for item in job["scenarios"]
        ],
        "env": {
            "trajectory_execution_steps": 5,
            "decision_repeat": 5,
            "physics_world_step_size": 0.02,
            "random_agent_model": False,
        },
    }

    with pytest.raises(RuntimeError, match="resolved sampler disagrees with energy matrix"):
        _validate_resolved_config(resolved, matrix, job)


def test_energy_matrix_launcher_pins_declared_sampler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[list[str]] = []

    def capture_run(command: list[str], **_: object) -> SimpleNamespace:
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_energy_matrix.subprocess, "run", capture_run)
    monkeypatch.setattr(
        run_energy_matrix,
        "_collect_run",
        lambda *_: {"status": "completed"},
    )

    returncode = run_energy_matrix.run_matrix(
        ROOT / "configs" / "benchmark" / "energy_matrix.yaml", tmp_path / "matrix"
    )

    assert returncode == 0
    assert len(commands) == 12
    assert all("planner/sampler=ddim5" in command for command in commands)
