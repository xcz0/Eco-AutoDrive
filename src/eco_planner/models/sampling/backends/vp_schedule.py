"""Continuous-time linear VP-SDE schedule shared by diffusion samplers."""

from __future__ import annotations

import torch


class LinearVpSchedule:
    """The pinned beta schedule used by the official Diffusion Planner."""

    def __init__(self, beta_0: float = 0.1, beta_1: float = 20.0) -> None:
        self.total_n = 1000
        self.beta_0 = beta_0
        self.beta_1 = beta_1

    def log_alpha(self, timestep: torch.Tensor) -> torch.Tensor:
        beta_range = self.beta_1 - self.beta_0
        return -0.25 * timestep.square() * beta_range - 0.5 * timestep * self.beta_0

    def alpha(self, timestep: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.log_alpha(timestep))

    def sigma(self, timestep: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(1.0 - torch.exp(2.0 * self.log_alpha(timestep)))

    def lambda_(self, timestep: torch.Tensor) -> torch.Tensor:
        log_alpha = self.log_alpha(timestep)
        return log_alpha - 0.5 * torch.log(1.0 - torch.exp(2.0 * log_alpha))

    def inverse_lambda(self, value: torch.Tensor) -> torch.Tensor:
        temporary = (
            2.0
            * (self.beta_1 - self.beta_0)
            * torch.logaddexp(-2.0 * value, torch.zeros((1,), device=value.device))
        )
        delta = self.beta_0**2 + temporary
        return temporary / (torch.sqrt(delta) + self.beta_0) / (self.beta_1 - self.beta_0)
