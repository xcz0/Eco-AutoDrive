"""Top-down closed-loop evaluation rendering."""

from __future__ import annotations

import numpy as np

from eco_planner.envs import TrajectoryExecutionRecord, TrajectoryMetaDriveEnv
from eco_planner.evaluation.config import VideoConfig


def draw_world_polyline(
    frame: np.ndarray,
    points: np.ndarray,
    camera_position: np.ndarray,
    scaling: float,
    color: tuple[int, int, int],
    width: int,
) -> None:
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("polyline points must have shape [N, 2]")
    if points.shape[0] < 2:
        return
    height, frame_width = frame.shape[:2]
    pixels = np.empty((points.shape[0], 2), dtype=np.int64)
    pixels[:, 0] = np.rint(frame_width / 2 + (points[:, 0] - camera_position[0]) * scaling)
    pixels[:, 1] = np.rint(height / 2 - (points[:, 1] - camera_position[1]) * scaling)
    for start, end in zip(pixels[:-1], pixels[1:], strict=True):
        _draw_segment(frame, tuple(start), tuple(end), color, width)


def render_cycle_frame(
    env: TrajectoryMetaDriveEnv,
    execution: TrajectoryExecutionRecord,
    anchor_position: np.ndarray,
    video_config: VideoConfig,
    plan_index: int,
) -> np.ndarray:
    frame = env.render(
        text={"plan_cycle": plan_index, "route_completion": execution.route_completion},
        mode="top_down",
        screen_size=(video_config.screen_width, video_config.screen_height),
        film_size=(video_config.film_width, video_config.film_height),
        scaling=float(video_config.scaling),
        num_stack=20,
        history_smooth=0,
        window=False,
        center_on_map=False,
    )
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] < 3:
        raise RuntimeError("MetaDrive top-down renderer did not return an RGB frame")
    rendered = frame.copy()
    camera_position = np.asarray(env.agent.position, dtype=np.float64)
    planned = np.vstack((anchor_position, execution.world_centers))
    executed = np.vstack(
        (
            anchor_position,
            execution.substep_states[:, :2],
        )
    )
    draw_world_polyline(
        rendered, planned, camera_position, float(video_config.scaling), (40, 90, 240), 1
    )
    draw_world_polyline(
        rendered, executed, camera_position, float(video_config.scaling), (30, 210, 80), 2
    )
    return rendered


def _draw_segment(
    frame: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
    width: int,
) -> None:
    delta_x = end[0] - start[0]
    delta_y = end[1] - start[1]
    samples = max(abs(delta_x), abs(delta_y)) + 1
    xs = np.rint(np.linspace(start[0], end[0], samples)).astype(np.int64)
    ys = np.rint(np.linspace(start[1], end[1], samples)).astype(np.int64)
    height, frame_width = frame.shape[:2]
    for offset_x in range(-width, width + 1):
        for offset_y in range(-width, width + 1):
            draw_x = xs + offset_x
            draw_y = ys + offset_y
            valid = (draw_x >= 0) & (draw_x < frame_width) & (draw_y >= 0) & (draw_y < height)
            frame[draw_y[valid], draw_x[valid], :3] = color
