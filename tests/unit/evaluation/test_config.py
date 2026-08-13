from __future__ import annotations

import pytest
from omegaconf import OmegaConf
from pydantic import ValidationError

from eco_planner.evaluation.config import parse_evaluation_config


def _config() -> object:
    return OmegaConf.create(
        {
            "name": "typed-evaluation",
            "map_query_radius_m": 100.0,
            "evaluation": {
                "mode": "no_traffic",
                "profile": "standard",
                "history_warmup_steps": 0,
                "evaluated_horizon_steps": 5,
                "execution": {
                    "mode": "serial",
                    "launcher": "basic",
                    "worker_count": 1,
                    "torch_threads_per_worker": None,
                    "deterministic": False,
                },
            },
            "env": {"horizon": 5, "traffic_density": 0.0},
            "model": {"args_path": "args.json", "checkpoint_path": "model.pth"},
            "runtime": {
                "accelerator": "cpu",
                "devices": 1,
                "precision": "32-true",
                "seed": 0,
            },
            "sampler": {"name": "dpm10", "implementation": "diffusers"},
            "guidance": {"name": "none"},
            "scenarios": [{"name": "straight", "map": "S", "seed": 0}],
            "video": {
                "enabled": False,
                "fps": 2,
                "screen_width": 32,
                "screen_height": 32,
                "film_width": 32,
                "film_height": 32,
                "scaling": 1.0,
            },
        }
    )


def test_parse_evaluation_config_returns_frozen_typed_boundary() -> None:
    config = parse_evaluation_config(_config())

    assert config.scenarios[0].map == "S"
    with pytest.raises(ValidationError, match="frozen"):
        config.runtime.seed = 2  # type: ignore[misc]


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("runtime.seed", "0", "runtime.seed"),
        ("runtime.devices", True, "runtime.devices"),
        ("video.enabled", 0, "video.enabled"),
        ("map_query_radius_m", float("nan"), "map_query_radius_m"),
    ],
)
def test_parse_evaluation_config_rejects_coercion_and_nonfinite_values(
    path: str, value: object, message: str
) -> None:
    config = _config()
    OmegaConf.update(config, path, value)

    with pytest.raises((ValidationError, ValueError), match=message):
        parse_evaluation_config(config)


def test_parse_evaluation_config_rejects_extra_fields() -> None:
    config = _config()
    config.video.unexpected = 1

    with pytest.raises(ValidationError, match="video.unexpected"):
        parse_evaluation_config(config)
