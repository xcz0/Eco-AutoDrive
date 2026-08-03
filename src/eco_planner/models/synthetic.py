"""Fixed synthetic observation used only by the stage 0 smoke test."""

from __future__ import annotations

import torch


def make_stage0_observation(device: torch.device) -> dict[str, torch.Tensor]:
    """Create one non-degenerate batch satisfying the official raw input contract."""

    observation = {
        "ego_current_state": torch.zeros((1, 10), dtype=torch.float32, device=device),
        "neighbor_agents_past": torch.zeros((1, 32, 21, 11), dtype=torch.float32, device=device),
        "static_objects": torch.zeros((1, 5, 10), dtype=torch.float32, device=device),
        "lanes": torch.zeros((1, 70, 20, 12), dtype=torch.float32, device=device),
        "lanes_speed_limit": torch.zeros((1, 70, 1), dtype=torch.float32, device=device),
        "lanes_has_speed_limit": torch.zeros((1, 70, 1), dtype=torch.bool, device=device),
        "route_lanes": torch.zeros((1, 25, 20, 12), dtype=torch.float32, device=device),
    }
    observation["ego_current_state"][0] = torch.tensor(
        [0.0, 0.0, 1.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=device
    )
    timesteps = torch.arange(21, dtype=torch.float32, device=device)
    neighbor = observation["neighbor_agents_past"][0, 0]
    neighbor[:, 0] = 12.0 + timesteps
    neighbor[:, 2] = 1.0
    neighbor[:, 4] = 10.0
    neighbor[:, 6] = 1.8
    neighbor[:, 7] = 4.8
    neighbor[:, 8] = 1.0
    points = torch.arange(20, dtype=torch.float32, device=device)
    for name in ("lanes", "route_lanes"):
        lane = observation[name][0, 0]
        lane[:, 0] = points * 5.0
        lane[:, 2] = 1.0
        lane[:, 4] = 1.75
        lane[:, 6] = -1.75
        lane[:, 8] = 1.0
    observation["lanes_speed_limit"][0, 0, 0] = 13.89
    observation["lanes_has_speed_limit"][0, 0, 0] = True
    return observation
