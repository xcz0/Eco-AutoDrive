"""Run the standalone stage 0 Diffusion Planner baseline verification."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from eco_planner.models.pretrained import load_official_diffusion_planner
from eco_planner.models.synthetic import make_stage0_observation

ARGS_SHA256 = "7e62b89a50953f133d55484777e54490f7f24e58feec1efcf696bcc7b91bdf10"
CHECKPOINT_SHA256 = "7a441df91ebe1c912d8262010c40486da24f425f757e2b4228072e251ab67d45"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was explicitly requested but is unavailable")
    planner, report = load_official_diffusion_planner(
        args.checkpoint_dir / "args.json",
        args.checkpoint_dir / "model.pth",
        ARGS_SHA256,
        CHECKPOINT_SHA256,
        device,
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    observation = make_stage0_observation(device)
    noise = torch.randn((1, 11, 80, 4), dtype=torch.float32, device=device, generator=generator)
    first = planner.predict(observation, noise)
    second = planner.predict(observation, noise)
    if not torch.isfinite(first).all():
        raise RuntimeError("baseline prediction contains NaN or Inf")
    if not torch.equal(first, second):
        raise RuntimeError("repeated baseline prediction is not bitwise identical")
    output_bytes = first.detach().contiguous().cpu().numpy().tobytes()
    payload = {
        "args_sha256": report.args_sha256,
        "checkpoint_sha256": report.checkpoint_sha256,
        "ema_tensor_count": report.ema_tensor_count,
        "parameter_count": report.parameter_count,
        "prediction_sha256": hashlib.sha256(output_bytes).hexdigest(),
        "prediction_shape": list(first.shape),
        "prediction_sum": float(first.sum().item()),
        "runtime_device": report.runtime_device,
        "seed": args.seed,
        "status": "pass",
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
