"""Strict parsing for the pinned official Diffusion Planner args.json."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eco_planner.models.normalization import ObservationNormalizer, StateNormalizer

_EXPECTED_KEYS = {
    "future_len",
    "time_len",
    "agent_state_dim",
    "agent_num",
    "static_objects_state_dim",
    "static_objects_num",
    "lane_len",
    "lane_state_dim",
    "lane_num",
    "map_len",
    "map_state_dim",
    "map_num",
    "route_len",
    "route_state_dim",
    "route_num",
    "encoder_drop_path_rate",
    "decoder_drop_path_rate",
    "device",
    "encoder_depth",
    "decoder_depth",
    "num_heads",
    "hidden_dim",
    "diffusion_model_type",
    "predicted_neighbor_num",
    "state_normalizer",
    "observation_normalizer",
}

_EXPECTED_DIMENSIONS = {
    "future_len": 80,
    "time_len": 21,
    "agent_state_dim": 11,
    "agent_num": 32,
    "static_objects_state_dim": 10,
    "static_objects_num": 5,
    "lane_len": 20,
    "lane_state_dim": 12,
    "lane_num": 70,
    "map_len": 10,
    "map_state_dim": 4,
    "map_num": 5,
    "route_len": 20,
    "route_state_dim": 12,
    "route_num": 25,
    "encoder_depth": 3,
    "decoder_depth": 3,
    "num_heads": 6,
    "hidden_dim": 192,
    "predicted_neighbor_num": 10,
}

_OBSERVATION_KEYS = {
    "ego_current_state",
    "neighbor_agents_past",
    "static_objects",
    "lanes",
    "lanes_speed_limit",
    "route_lanes",
    "route_lanes_speed_limit",
}


@dataclass(frozen=True)
class OfficialDiffusionPlannerConfig:
    """The architecture contract encoded by the official checkpoint metadata."""

    future_len: int
    time_len: int
    agent_state_dim: int
    agent_num: int
    static_objects_state_dim: int
    static_objects_num: int
    lane_len: int
    lane_state_dim: int
    lane_num: int
    map_len: int
    map_state_dim: int
    map_num: int
    route_len: int
    route_state_dim: int
    route_num: int
    encoder_drop_path_rate: float
    decoder_drop_path_rate: float
    checkpoint_device: str
    encoder_depth: int
    decoder_depth: int
    num_heads: int
    hidden_dim: int
    diffusion_model_type: str
    predicted_neighbor_num: int
    state_normalizer: StateNormalizer
    observation_normalizer: ObservationNormalizer

    @classmethod
    def from_json(cls, path: Path) -> OfficialDiffusionPlannerConfig:
        if not path.is_file():
            raise FileNotFoundError(f"official args file does not exist: {path}")
        with path.open("r", encoding="utf-8") as handle:
            raw: Any = json.load(handle)
        if not isinstance(raw, dict):
            raise ValueError("official args.json must contain a JSON object")
        if set(raw) != _EXPECTED_KEYS:
            missing = sorted(_EXPECTED_KEYS - set(raw))
            unexpected = sorted(set(raw) - _EXPECTED_KEYS)
            message = "official args keys mismatch; "
            message += f"missing={missing}, unexpected={unexpected}"
            raise ValueError(message)
        for name, expected in _EXPECTED_DIMENSIONS.items():
            if raw[name] != expected:
                message = f"official args field {name!r} must be {expected}"
                message += f", got {raw[name]!r}"
                raise ValueError(message)
        if raw["route_len"] != raw["lane_len"]:
            raise ValueError("route_len must equal lane_len")
        if raw["diffusion_model_type"] != "x_start":
            raise ValueError("only the official x_start diffusion model is supported")
        if raw["device"] not in {"cpu", "cuda"}:
            raise ValueError("checkpoint device must be either 'cpu' or 'cuda'")
        if not isinstance(raw["encoder_drop_path_rate"], (int, float)) or not isinstance(
            raw["decoder_drop_path_rate"], (int, float)
        ):
            raise ValueError("drop path rates must be numeric")
        encoder_drop_path_rate = raw["encoder_drop_path_rate"]
        decoder_drop_path_rate = raw["decoder_drop_path_rate"]
        if not 0 <= encoder_drop_path_rate < 1 or not 0 <= decoder_drop_path_rate < 1:
            raise ValueError("drop path rates must be in [0, 1)")
        observation = raw["observation_normalizer"]
        if not isinstance(observation, dict) or set(observation) != _OBSERVATION_KEYS:
            raise ValueError(
                "official observation normalizer keys do not match the checkpoint contract"
            )
        omitted_keys = {"device", "state_normalizer", "observation_normalizer"}
        model_fields = {name: raw[name] for name in _EXPECTED_KEYS - omitted_keys}
        return cls(
            **model_fields,
            checkpoint_device=raw["device"],
            state_normalizer=StateNormalizer(**raw["state_normalizer"]),
            observation_normalizer=ObservationNormalizer(observation),
        )
