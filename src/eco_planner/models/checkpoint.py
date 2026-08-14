"""Official checkpoint translation and load metadata."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import torch


def extract_official_ema_state_dict(checkpoint: object) -> OrderedDict[str, torch.Tensor]:
    ema = checkpoint["ema_state_dict"]  # type: ignore[index]
    translated: OrderedDict[str, torch.Tensor] = OrderedDict()
    for original_name, value in ema.items():
        name = original_name.removeprefix("module.")
        name = name.replace("encoder.encoder.", "encoder.", 1)
        name = name.replace("decoder.decoder.", "decoder.", 1)
        translated[name] = value
    return translated


@dataclass(frozen=True)
class CheckpointLoadReport:
    ema_tensor_count: int
    parameter_count: int
