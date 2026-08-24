from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from scripts.run_energy_matrix import _load_matrix, _validate_resolved_config

ROOT = Path(__file__).resolve().parents[2]


def test_energy_matrix_declares_required_coverage_and_guidance_profiles() -> None:
    matrix = _load_matrix(ROOT / "configs" / "benchmark" / "energy_matrix.yaml")

    assert matrix["planner_noise_seed"] == 0
    assert matrix["execution"]["simulator_hz"] == 10
    assert matrix["execution"]["planner_hz"] == 2
    assert matrix["energy_metric"] == {
        "name": "metadrive_episode_energy",
        "source_info_field": "episode_energy",
        "step_source_info_field": "step_energy",
        "unit": "mL",
        "interpretation": "MetaDrive fuel-consumption proxy",
        "accumulation_boundary": "environment reset to episode termination or horizon",
    }
    assert [item["longitudinal_scale"] for item in matrix["guidance_profiles"]] == [
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
                    "programmatic_lane_speed_limit_kmh": 50.0,
                    "traffic_density": 0.0,
                    "random_agent_model": False,
                    "vehicle_config": {
                        "wheel_friction": 0.9,
                        "max_engine_force": 800,
                        "max_brake_force": 150,
                        "max_steering": 40,
                        "max_speed_km_h": 80,
                    },
                },
            }
        ),
        resolve=True,
    )
    assert isinstance(resolved, dict)

    _validate_resolved_config(resolved, matrix, job)
