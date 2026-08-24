from __future__ import annotations

from pathlib import Path

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
