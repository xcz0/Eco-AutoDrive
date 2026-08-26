from __future__ import annotations

from pathlib import Path

import numpy as np

from ppo_reward_ab import build_training_command, load_ab_config
from summarize_ppo_reward_ab import _effect_metrics, _initial_collection_equal

ROOT = Path(__file__).resolve().parents[2]


def test_ppo_reward_ab_config_declares_a_causally_observable_matched_pair() -> None:
    config = load_ab_config(ROOT / "configs" / "matrices" / "ppo_reward_ab.yaml")

    assert config.matched_training.update_count == 4
    assert config.matched_training.transitions_per_environment == 16
    assert [(item.id, item.reward_config) for item in config.profiles] == [
        ("builtin", "metadrive_builtin_v1"),
        ("energy", "plannerrft_energy_v1"),
    ]


def test_ppo_reward_ab_commands_differ_only_by_profile_identity_and_output() -> None:
    config = load_ab_config(ROOT / "configs" / "matrices" / "ppo_reward_ab.yaml")
    commands = [
        build_training_command(config, profile, 0, 0, Path("output") / profile.id)
        for profile in config.profiles
    ]

    common = {
        argument
        for argument in commands[0]
        if not argument.startswith(("rl/reward=", "name=", "hydra.run.dir="))
    }
    assert common == {
        argument
        for argument in commands[1]
        if not argument.startswith(("rl/reward=", "name=", "hydra.run.dir="))
    }
    assert "training.update_count=4" in commands[0]
    assert "training.transitions_per_environment=16" in commands[0]


def test_ppo_reward_ab_effect_metrics_exclude_the_matched_pre_update_collection() -> None:
    initial = _update(action=0.0, distance=100.0, fuel=10.0, speed=20.0, progress=5.0)
    post_update = _update(action=0.25, distance=2.0, fuel=0.1, speed=10.0, progress=1.0)
    run = {"updates": (initial, post_update)}

    metrics = _effect_metrics(run)

    assert metrics["sample_count"] == 1
    assert metrics["longitudinal_action_mean"] == 0.25
    assert metrics["fuel_proxy_total_ml"] == 0.1
    assert metrics["fuel_proxy_ml_per_km"] == 50.0
    assert metrics["route_progress_delta"] == 1.0


def test_ppo_reward_ab_pairing_requires_exact_pre_update_common_arrays() -> None:
    initial = _update(action=0.0, distance=1.0, fuel=0.05, speed=10.0, progress=1.0)
    left = {"updates": (initial,)}
    right = {"updates": ({name: value.copy() for name, value in initial.items()},)}

    assert _initial_collection_equal(left, right)
    right["updates"][0]["guidance_action"][0, 1] = 0.1
    assert not _initial_collection_equal(left, right)


def _update(
    *, action: float, distance: float, fuel: float, speed: float, progress: float
) -> dict[str, np.ndarray]:
    return {
        "guidance_action": np.asarray([[0.0, action]], dtype=np.float32),
        "route_completion_delta": np.asarray([progress], dtype=np.float32),
        "speed_mps": np.asarray([speed], dtype=np.float32),
        "stopped": np.asarray([False]),
        "crash_vehicle": np.asarray([False]),
        "crash_object": np.asarray([False]),
        "crash_building": np.asarray([False]),
        "crash_human": np.asarray([False]),
        "crash_sidewalk": np.asarray([False]),
        "out_of_road": np.asarray([False]),
        "step_distance_m": np.asarray([distance], dtype=np.float32),
        "native_step_energy_ml": np.asarray([0.0], dtype=np.float32),
        "executed_fuel_proxy_step_energy_ml": np.asarray([fuel], dtype=np.float32),
    }
