"""Shared world- and ego-frame geometry operations."""

from __future__ import annotations

import numpy as np

from .arrays import Float64Array, WorldPointArray, WorldVectorArray


def rear_axle_position(
    center_position: WorldVectorArray, heading_rad: float, rear_wheelbase_m: float
) -> WorldVectorArray:
    direction = np.array([np.cos(heading_rad), np.sin(heading_rad)], dtype=np.float64)
    return center_position - rear_wheelbase_m * direction


def world_points_to_local(
    points: WorldPointArray, anchor_xy_m: WorldVectorArray, anchor_heading_rad: float
) -> WorldPointArray:
    translated = points - anchor_xy_m
    return world_vectors_to_local(translated, anchor_heading_rad)


def local_points_to_world(
    points: WorldPointArray, anchor_xy_m: WorldVectorArray, anchor_heading_rad: float
) -> WorldPointArray:
    cosine = np.cos(anchor_heading_rad)
    sine = np.sin(anchor_heading_rad)
    rotation = np.array([[cosine, -sine], [sine, cosine]], dtype=np.float64)
    return points @ rotation.T + anchor_xy_m


def world_vectors_to_local(vectors: WorldPointArray, anchor_heading_rad: float) -> WorldPointArray:
    cosine = np.cos(anchor_heading_rad)
    sine = np.sin(anchor_heading_rad)
    rotation = np.array([[cosine, sine], [-sine, cosine]], dtype=np.float64)
    return vectors @ rotation.T


def shortest_angle_delta(angle: Float64Array) -> Float64Array:
    array = np.asarray(angle, dtype=np.float64)
    return (array + np.pi) % (2.0 * np.pi) - np.pi
