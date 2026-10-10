"""Lightweight JSON and NPZ I/O; domain schemas belong to their readers and writers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel


def write_json(path: Path, payload: BaseModel | dict[str, object]) -> None:
    """Persist stable, UTF-8 JSON for a typed research artifact."""

    value: Any = payload.model_dump(mode="json") if isinstance(payload, BaseModel) else payload
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Persist named arrays without weakening each artifact schema to ``Any``."""

    # NumPy's stub treats every dynamic keyword as its reserved allow_pickle option.
    np.savez(path, **arrays)  # pyright: ignore[reportArgumentType]


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def read_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}
