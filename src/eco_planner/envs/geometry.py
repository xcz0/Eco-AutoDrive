"""Shared ego-frame geometry for MetaDrive boundaries."""

from __future__ import annotations

from typing import Any

import numpy as np


def rear_axle_position(
    center_position: np.ndarray, heading_rad: float, rear_wheelbase_m: float
) -> np.ndarray:
    center = np.asarray(center_position, dtype=np.float64)
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError("center position must be a finite two-dimensional vector")
    if not np.isfinite(heading_rad):
        raise ValueError("heading must be finite")
    if not np.isfinite(rear_wheelbase_m) or rear_wheelbase_m <= 0.0:
        raise ValueError("rear wheelbase must be finite and positive")
    direction = np.array([np.cos(heading_rad), np.sin(heading_rad)], dtype=np.float64)
    return center - rear_wheelbase_m * direction


def world_points_to_local(
    points: Any, anchor_xy_m: np.ndarray, anchor_heading_rad: float
) -> np.ndarray:
    array = _finite_points(points)
    anchor = _finite_anchor(anchor_xy_m, anchor_heading_rad)
    translated = array - anchor
    return world_vectors_to_local(translated, anchor_heading_rad)


def local_points_to_world(
    points: Any, anchor_xy_m: np.ndarray, anchor_heading_rad: float
) -> np.ndarray:
    array = _finite_points(points)
    anchor = _finite_anchor(anchor_xy_m, anchor_heading_rad)
    cosine = np.cos(anchor_heading_rad)
    sine = np.sin(anchor_heading_rad)
    rotation = np.array([[cosine, -sine], [sine, cosine]], dtype=np.float64)
    return array @ rotation.T + anchor


def world_vectors_to_local(vectors: Any, anchor_heading_rad: float) -> np.ndarray:
    array = _finite_points(vectors)
    if not np.isfinite(anchor_heading_rad):
        raise ValueError("anchor heading must be finite")
    cosine = np.cos(anchor_heading_rad)
    sine = np.sin(anchor_heading_rad)
    rotation = np.array([[cosine, sine], [-sine, cosine]], dtype=np.float64)
    return array @ rotation.T


def shortest_angle_delta(angle: Any) -> np.ndarray:
    array = np.asarray(angle, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("angle delta must be finite")
    return (array + np.pi) % (2.0 * np.pi) - np.pi


def _finite_points(points: Any) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2 or not np.isfinite(array).all():
        raise ValueError("points must form a finite [N, 2] array")
    return array


def _finite_anchor(anchor_xy_m: np.ndarray, anchor_heading_rad: float) -> np.ndarray:
    anchor = np.asarray(anchor_xy_m, dtype=np.float64)
    if anchor.shape != (2,) or not np.isfinite(anchor).all():
        raise ValueError("anchor position must be a finite two-dimensional vector")
    if not np.isfinite(anchor_heading_rad):
        raise ValueError("anchor heading must be finite")
    return anchor
