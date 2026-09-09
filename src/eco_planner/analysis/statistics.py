"""Float64 statistics with explicit ddof and undefined-vector semantics."""

import statistics as descriptive_statistics
from collections.abc import Sequence
from math import isfinite
from typing import TypedDict

import numpy as np


def statistics(value: np.ndarray, quantiles: Sequence[float], *, ddof: int = 0) -> dict:
    x = np.asarray(value, dtype=np.float64).reshape(-1)
    if x.size <= ddof or not np.isfinite(x).all():
        raise ValueError("statistics require enough finite samples")
    return {
        "mean": float(x.mean()),
        "std": float(x.std(ddof=ddof)),
        "quantiles": {
            str(q): float(v) for q, v in zip(quantiles, np.quantile(x, quantiles), strict=True)
        },
    }


def _ranks(x: np.ndarray) -> np.ndarray:
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - (counts - 1) / 2)[inverse]


def cosine(x: np.ndarray, y: np.ndarray) -> float | None:
    x, y = x.astype(np.float64), y.astype(np.float64)
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return None if denominator == 0 else float(np.dot(x, y) / denominator)


def advantage_comparison(x: np.ndarray, y: np.ndarray) -> dict:
    x, y = x.astype(np.float64).reshape(-1), y.astype(np.float64).reshape(-1)
    rx, ry = _ranks(x), _ranks(y)
    return {
        "pearson": cosine(x - x.mean(), y - y.mean()),
        "spearman": cosine(rx - rx.mean(), ry - ry.mean()),
        "sign_flip_fraction": float(np.mean(x * y < 0)),
        "zero_fraction_i": float(np.mean(x == 0)),
        "zero_fraction_j": float(np.mean(y == 0)),
    }


def rmse(x: np.ndarray, y: np.ndarray) -> float:
    delta = x.astype(np.float64) - y.astype(np.float64)
    return float(np.sqrt(np.mean(np.square(delta))))


def gradient_comparison(x: np.ndarray, y: np.ndarray) -> dict[str, float | None]:
    norm_x = np.linalg.norm(x.astype(np.float64))
    norm_y = np.linalg.norm(y.astype(np.float64))
    return {
        "cosine": cosine(x, y),
        "norm_ratio_j_over_i": None if norm_x == 0 else float(norm_y / norm_x),
    }


def paired_difference(
    reference: np.ndarray,
    comparison: np.ndarray,
    scenario_ids: np.ndarray,
    quantiles: Sequence[float],
) -> tuple[dict, np.ndarray]:
    if reference.shape != comparison.shape or reference.size != scenario_ids.size:
        raise ValueError("paired arrays and scenario index must have matching shapes")
    delta = comparison.astype(np.float64) - reference.astype(np.float64)
    return {
        "all": statistics(delta, quantiles),
        "per_scenario": {
            str(slot): statistics(delta[scenario_ids == slot], quantiles)
            for slot in np.unique(scenario_ids)
        },
    }, delta


class Measurement(TypedDict):
    samples: list[float]
    median: float
    minimum: float
    maximum: float


def measurement(samples: Sequence[float]) -> Measurement:
    values = [float(value) for value in samples]
    if not values:
        raise ValueError("benchmark measurement requires at least one sample")
    if not all(isfinite(value) for value in values):
        raise ValueError("benchmark measurement samples must be finite")
    if any(value < 0.0 for value in values):
        raise ValueError("benchmark measurement samples must be non-negative")
    return {
        "samples": values,
        "median": descriptive_statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
    }
