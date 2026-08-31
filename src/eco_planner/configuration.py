"""Shared configuration-boundary helpers for repository entrypoints."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, TypeGuard

from omegaconf import DictConfig, OmegaConf

_ASSIGNMENT = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")


def load_local_environment(env_path: Path) -> None:
    """Load a strict environment file without replacing existing variables."""

    if not env_path.is_file():
        raise FileNotFoundError(f"missing local environment file: {env_path}")

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
