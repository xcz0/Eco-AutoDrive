from __future__ import annotations

from typing import cast

import numpy as np
import torch
from tensordict import TensorDictBase

from eco_planner.rl import ExplorationPolicy, PPOUpdater

GRADIENT_GROUPS = ("actor_head", "shared_trunk", "actor", "lateral", "longitudinal")
ADVANTAGE_FORMS = ("raw", "center", "z")


def actor_gradients(policy: ExplorationPolicy) -> tuple[dict[str, np.ndarray], list[dict]]:
    groups: dict[str, list[np.ndarray]] = {key: [] for key in ("actor_head", "shared_trunk")}
    layout = []
    offset = 0
    for name, parameter in policy.named_parameters():
        if name.startswith("value_head."):
            if parameter.grad is not None:
                raise RuntimeError("actor backward reached value head")
            continue
        if parameter.grad is None:
            raise RuntimeError(f"actor parameter has no gradient: {name}")
        value = parameter.grad.detach().cpu().float().numpy().reshape(-1).copy()
        if not np.isfinite(value).all():
            raise FloatingPointError(f"nonfinite actor gradient: {name}")
        group = "actor_head" if name.startswith("actor_head.") else "shared_trunk"
        groups[group].append(value)
        layout.append(
            {"name": name, "shape": list(parameter.shape), "offset": offset, "size": value.size}
        )
        offset += value.size
    result = {key: np.concatenate(values) for key, values in groups.items()}
    result["actor"] = np.concatenate(
        [
            cast(torch.Tensor, parameter.grad).detach().cpu().float().numpy().reshape(-1)
            for name, parameter in policy.named_parameters()
            if not name.startswith("value_head.")
        ]
    )
    # forward_tensors views the four rows as [lateral/longitudinal, alpha/beta].
    for name, rows in (("lateral", slice(0, 2)), ("longitudinal", slice(2, 4))):
        result[name] = np.concatenate(
            [
                cast(torch.Tensor, parameter.grad)[rows].detach().cpu().float().numpy().reshape(-1)
                for parameter in policy.actor_head.parameters()
            ]
        )
    return result, layout


def actor_backward(
    updater: PPOUpdater, batch: TensorDictBase, advantage: torch.Tensor
) -> tuple[float, dict[str, np.ndarray], list[dict]]:
    batch["advantage"] = advantage
    updater.policy.zero_grad(set_to_none=True)
    loss = updater.loss_module(batch.to(updater.device))["loss_objective"]
    if not torch.isfinite(loss).all():
        raise FloatingPointError("actor loss must be finite")
    loss.backward()
    gradient, layout = actor_gradients(updater.policy)
    return float(loss.detach()), gradient, layout
