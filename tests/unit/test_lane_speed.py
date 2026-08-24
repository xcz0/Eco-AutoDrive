from __future__ import annotations

from types import SimpleNamespace

from eco_planner.envs.lane_speed import ProgrammaticLaneSpeedAdapter


class _Lane:
    def __init__(self, index: str) -> None:
        self.index = index
        self.speed_limit = 1000.0

    def set_speed_limit(self, speed_limit: float) -> None:
        self.speed_limit = speed_limit


def test_block_speed_limit_profile_replaces_sentinel_by_generated_block() -> None:
    first = _Lane("first")
    first_block = SimpleNamespace(block_network=SimpleNamespace(get_all_lanes=lambda: [first]))
    profile_lanes = [_Lane(str(index)) for index in range(3)]
    blocks = [
        first_block,
        *[
            SimpleNamespace(block_network=SimpleNamespace(get_all_lanes=lambda lane=lane: [lane]))
            for lane in profile_lanes
        ],
    ]
    current_map = SimpleNamespace(
        blocks=blocks,
        road_network=SimpleNamespace(get_all_lanes=lambda: [first, *profile_lanes]),
    )
    adapter = ProgrammaticLaneSpeedAdapter(50.0, [50.0, 30.0, 50.0])

    adapter.apply(current_map)

    assert [lane.speed_limit for lane in [first, *profile_lanes]] == [50.0, 50.0, 30.0, 50.0]
    assert adapter.audit["block_speed_limit_profile_kmh"] == (50.0, 30.0, 50.0)
    assert adapter.audit["block_speed_limit_profile_applied_lane_count"] == 3
