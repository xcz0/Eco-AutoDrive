"""Shared configuration-boundary helpers for repository entrypoints."""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TypeGuard

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, Field, StrictInt

_ASSIGNMENT = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
_RESOURCE_CONFIG_GROUP = "components/resources"


class _SharedConfigModel(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)


class ScenarioConfig(_SharedConfigModel):
    """Objective-neutral identity of one map/seed scenario."""

    name: str = Field(min_length=1)
    map: str = Field(min_length=1)
    seed: StrictInt = Field(ge=0)


class ModelPathsConfig(_SharedConfigModel):
    """Filesystem inputs shared by evaluation, benchmarking, and RL."""

    args_path: str = Field(min_length=1)
    checkpoint_path: str = Field(min_length=1)


def load_local_environment(env_path: Path) -> None:
    """Load an optional strict environment file without replacing existing variables."""

    if not env_path.is_file():
        return

    for line_number, raw_line in enumerate(
        env_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _ASSIGNMENT.fullmatch(line)
        if match is None:
            raise ValueError(f"invalid .env assignment at {env_path}:{line_number}")
        name, value = match.groups()
        os.environ.setdefault(name, value)


def with_machine_resource_override(overrides: Sequence[str]) -> list[str]:
    """Select the local resource profile unless the caller already chose one."""

    resolved = list(overrides)
    if any(_is_resource_override(item) for item in resolved):
        return resolved
    machine_name = os.environ.get("MACHINE_NAME", "").strip()
    if machine_name:
        resolved.append(f"{_RESOURCE_CONFIG_GROUP}={machine_name}")
    return resolved


def load_resolved_yaml_mapping(path: Path) -> dict[str, Any]:
    """Load one YAML file, resolve required values, and require a mapping root."""

    config = OmegaConf.load(path)
    if not isinstance(config, DictConfig):
        raise TypeError(f"configuration must resolve to a mapping: {path}")
    return resolve_config_mapping(config)


def resolve_config_mapping(config: DictConfig) -> dict[str, Any]:
    """Resolve one Hydra mapping while preserving its dynamic boundary values."""

    raw = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    if not _is_string_mapping(raw):
        raise TypeError("configuration must resolve to a string-keyed mapping")
    return raw


def _is_string_mapping(value: object) -> TypeGuard[dict[str, Any]]:
    return isinstance(value, dict) and all(isinstance(key, str) for key in value)


def _is_resource_override(value: str) -> bool:
    candidate = value.lstrip("+~")
    return candidate == _RESOURCE_CONFIG_GROUP or candidate.startswith(f"{_RESOURCE_CONFIG_GROUP}=")
