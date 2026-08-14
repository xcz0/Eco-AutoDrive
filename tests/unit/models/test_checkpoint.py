from __future__ import annotations

import pytest
import torch

from eco_planner.models.checkpoint.loader import extract_official_ema_state_dict


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
