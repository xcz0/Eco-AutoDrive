"""The official deterministic 10-step DPM-Solver++ baseline sampler."""

from __future__ import annotations

from collections.abc import Callable

import torch


class _NoiseScheduleVP:
    def __init__(self, beta_0: float = 0.1, beta_1: float = 20.0) -> None:
        self.total_n = 1000
        self.beta_0 = beta_0
        self.beta_1 = beta_1

    def log_alpha(self, timestep: torch.Tensor) -> torch.Tensor:
        beta_range = self.beta_1 - self.beta_0
        return -0.25 * timestep.square() * beta_range - 0.5 * timestep * self.beta_0

    def alpha(self, timestep: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.log_alpha(timestep))

    def std(self, timestep: torch.Tensor) -> torch.Tensor:
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


def _expand(value: torch.Tensor, dimensions: int) -> torch.Tensor:
    return value.reshape([-1] + [1] * (dimensions - 1))


class BaselineDpmSampler:
    """Exact scope used by the upstream 10-step, order-2, multistep sampler."""

    def __init__(self) -> None:
        self._schedule = _NoiseScheduleVP()

    def sample(
        self,
        initial_sample: torch.Tensor,
        model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        current_state_constraint: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        schedule = self._schedule
        device = initial_sample.device
        final_time = 1.0 / schedule.total_n
        lambda_start = schedule.lambda_(torch.tensor(1.0, device=device))
        lambda_end = schedule.lambda_(torch.tensor(final_time, device=device))
        timesteps = schedule.inverse_lambda(
            torch.linspace(lambda_start.item(), lambda_end.item(), 11, device=device)
        )
        with torch.no_grad():
            sample = initial_sample
            previous_times = [timesteps[0]]
            previous_models = [self._x_start_prediction(sample, timesteps[0], model)]
            sample = current_state_constraint(sample)
            second_time = timesteps[1]
            sample = self._first_update(
                sample, previous_times[-1], second_time, previous_models[-1]
            )
            sample = current_state_constraint(sample)
            previous_times.append(second_time)
            previous_models.append(self._x_start_prediction(sample, second_time, model))
            for step in range(2, 11):
                next_time = timesteps[step]
                sample = self._second_update(sample, previous_models, previous_times, next_time)
                sample = current_state_constraint(sample)
                previous_times = [previous_times[-1], next_time]
                previous_models = [
                    previous_models[-1],
                    previous_models[-1]
                    if step == 10
                    else self._x_start_prediction(sample, next_time, model),
                ]
            final_prediction = self._x_start_prediction(
                sample, torch.tensor(final_time, device=device), model
            )
            return current_state_constraint(final_prediction)

    def _x_start_prediction(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        batch_timestep = timestep.expand(sample.shape[0])
        alpha = _expand(self._schedule.alpha(batch_timestep), sample.dim())
        std = _expand(self._schedule.std(batch_timestep), sample.dim())
        prediction = model(sample, batch_timestep)
        noise = (sample - alpha * prediction) / std
        return (sample - std * noise) / alpha

    def _first_update(
        self,
        sample: torch.Tensor,
        start: torch.Tensor,
        end: torch.Tensor,
        prediction: torch.Tensor,
    ) -> torch.Tensor:
        h = self._schedule.lambda_(end) - self._schedule.lambda_(start)
        alpha_end = torch.exp(self._schedule.log_alpha(end))
        return (
            self._schedule.std(end) / self._schedule.std(start) * sample
            - alpha_end * torch.expm1(-h) * prediction
        )

    def _second_update(
        self,
        sample: torch.Tensor,
        predictions: list[torch.Tensor],
        times: list[torch.Tensor],
        end: torch.Tensor,
    ) -> torch.Tensor:
        previous_one, previous_zero = predictions[-2], predictions[-1]
        time_one, time_zero = times[-2], times[-1]
        lambda_one = self._schedule.lambda_(time_one)
        lambda_zero = self._schedule.lambda_(time_zero)
        h = self._schedule.lambda_(end) - lambda_zero
        r0 = (lambda_zero - lambda_one) / h
        derivative = (previous_zero - previous_one) / r0
        phi_1 = torch.expm1(-h)
        alpha_end = torch.exp(self._schedule.log_alpha(end))
        return (
            self._schedule.std(end) / self._schedule.std(time_zero) * sample
            - alpha_end * phi_1 * previous_zero
            - 0.5 * alpha_end * phi_1 * derivative
        )
