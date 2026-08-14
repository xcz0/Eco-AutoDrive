from collections import OrderedDict

import torch

from eco_planner.models.checkpoint import extract_official_ema_state_dict


def test_ema_loader_translates_official_wrapper_prefixes() -> None:
    checkpoint = {
        "ema_state_dict": OrderedDict(
            (
                ("module.encoder.encoder.layer.weight", torch.ones(2, 2)),
                ("module.decoder.decoder.layer.bias", torch.zeros(2)),
            )
        )
    }

    translated = extract_official_ema_state_dict(checkpoint)

    assert tuple(translated) == ("encoder.layer.weight", "decoder.layer.bias")
    assert translated["encoder.layer.weight"].shape == (2, 2)
