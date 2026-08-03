from __future__ import annotations

from pathlib import Path

import pytest
import torch

from eco_planner.models.checkpoint import extract_official_ema_state_dict, verify_sha256


def test_hash_verification_rejects_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "asset.bin"
    path.write_bytes(b"stage-zero")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_sha256(path, "0" * 64)


def test_hash_verification_rejects_invalid_expected_hash(tmp_path: Path) -> None:
    path = tmp_path / "asset.bin"
    path.write_bytes(b"stage-zero")
    with pytest.raises(ValueError, match="lowercase hexadecimal"):
        verify_sha256(path, "A" * 64)


def test_ema_loader_rejects_invalid_prefix() -> None:
    with pytest.raises(ValueError, match="module"):
        extract_official_ema_state_dict({"model": {}, "ema_state_dict": {"invalid": torch.ones(1)}})


def test_ema_loader_rejects_missing_ema() -> None:
    with pytest.raises(ValueError, match="exactly model"):
        extract_official_ema_state_dict({"model": {}})


def test_ema_loader_rejects_incomplete_state_dict() -> None:
    with pytest.raises(ValueError, match="276 tensors"):
        extract_official_ema_state_dict(
            {"model": {}, "ema_state_dict": {"module.partial": torch.ones(1)}}
        )
