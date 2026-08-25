from __future__ import annotations

import numpy as np

from eco_planner.evaluation import rendering as video


def test_world_polyline_draws_on_frame() -> None:
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    video.draw_world_polyline(
        frame, np.array([[0.0, 0.0], [5.0, 0.0]]), np.array([0.0, 0.0]), 1.0, (1, 2, 3), 0
    )
    assert np.count_nonzero(frame) > 0
