from __future__ import annotations

from pathlib import Path

import pytest
import torch

from eco_planner.models.pretrained import load_official_diffusion_planner
from eco_planner.models.synthetic import make_stage0_observation

ARGS_SHA256 = "7e62b89a50953f133d55484777e54490f7f24e58feec1efcf696bcc7b91bdf10"
CHECKPOINT_SHA256 = "7a441df91ebe1c912d8262010c40486da24f425f757e2b4228072e251ab67d45"


@pytest.mark.slow
def test_official_ema_checkpoint_cpu_smoke() -> None:
    checkpoint_dir = Path("checkpoints/DP-origin")
    if not checkpoint_dir.is_dir():
        pytest.fail("stage 0 checkpoint assets are required at checkpoints/DP-origin")
    device = torch.device("cpu")
    planner, report = load_official_diffusion_planner(
        checkpoint_dir / "args.json",
        checkpoint_dir / "model.pth",
        ARGS_SHA256,
        CHECKPOINT_SHA256,
        device,
    )
    generator = torch.Generator(device=device).manual_seed(0)
    noise = torch.randn((1, 11, 80, 4), generator=generator, device=device)
    observation = make_stage0_observation(device)
    first = planner.predict(observation, noise)
    second = planner.predict(observation, noise)
    bad_shape = dict(observation)
    bad_shape["ego_current_state"] = torch.zeros((1, 9), dtype=torch.float32, device=device)
    with pytest.raises(ValueError, match="shape"):
        planner.predict(bad_shape, noise)
    nonfinite_noise = noise.clone()
    nonfinite_noise[0, 0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        planner.predict(observation, nonfinite_noise)
    assert report.ema_tensor_count == 276
    assert report.parameter_count == 6_042_628
    assert tuple(first.shape) == (1, 11, 80, 4)
    assert torch.isfinite(first).all()
    assert torch.equal(first, second)
