"""Programmatic lane speed-limit contracts."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

PROGRAMMATIC_SPEED_LIMIT_SENTINEL_KMH = 1000.0
MAX_LANE_SPEED_LIMIT_KMH = 130.0


@dataclass(slots=True)
class ProgrammaticLaneSpeedAdapter:
    """Replace PGMap's unset-speed sentinel and retain reset audit metadata."""

    configured_speed_limit_kmh: float
    block_speed_limit_profile_kmh: Sequence[float] | None = None
    _sentinel_lane_ids: frozenset[str] | None = field(default=None, init=False)
    _map: object | None = field(default=None, init=False)
    _audit: dict[str, object] | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._validate_speed_limit(
            self.configured_speed_limit_kmh, "programmatic_lane_speed_limit_kmh"
        )
        profile = self.block_speed_limit_profile_kmh
        if profile is None:
            return
        if isinstance(profile, (str, bytes)) or not profile:
            raise ValueError("programmatic lane speed-limit profile must be a non-empty sequence")
        normalized = tuple(float(value) for value in profile)
        for index, speed_limit_kmh in enumerate(normalized):
            self._validate_speed_limit(
                speed_limit_kmh, f"programmatic lane speed-limit profile {index}"
            )
        self.block_speed_limit_profile_kmh = normalized

    @property
    def audit(self) -> dict[str, object]:
        if self._audit is None:
            raise RuntimeError("programmatic lane speed limits are unavailable before reset")
        return dict(self._audit)

    def clear_audit(self) -> None:
        self._audit = None

    def apply(self, current_map: object) -> None:
        road_network = getattr(current_map, "road_network", None)
        if road_network is None or not hasattr(road_network, "get_all_lanes"):
            raise RuntimeError("current map does not expose a lane road network")
        lanes = road_network.get_all_lanes()
        if not isinstance(lanes, list) or not lanes:
            raise RuntimeError("current map road network has no lanes")

        lane_ids = [repr(getattr(lane, "index", None)) for lane in lanes]
        if len(set(lane_ids)) != len(lane_ids):
            raise RuntimeError("current map exposes duplicate lane identifiers")
        if current_map is not self._map:
            self._sentinel_lane_ids = frozenset(
                lane_id
                for lane_id, lane in zip(lane_ids, lanes, strict=True)
                if getattr(lane, "speed_limit", None) == PROGRAMMATIC_SPEED_LIMIT_SENTINEL_KMH
            )
            self._map = current_map
        if self._sentinel_lane_ids is None:
            raise RuntimeError("programmatic sentinel state was not initialized")

        profile_limits = self._profile_limits(current_map, lanes)

        preserved_count = 0
        profile_applied_count = 0
        counts: dict[str, int] = {}
        for lane_id, lane in zip(lane_ids, lanes, strict=True):
            speed_limit: float = lane.speed_limit
            if not np.isfinite(speed_limit):
                raise ValueError(f"lane {lane_id} speed limit must be a finite numeric km/h value")
            current_kmh = speed_limit
            if lane_id in self._sentinel_lane_ids:
                expected_kmh = profile_limits.get(lane_id, self.configured_speed_limit_kmh)
                if current_kmh == PROGRAMMATIC_SPEED_LIMIT_SENTINEL_KMH:
                    setter = getattr(lane, "set_speed_limit", None)
                    if not callable(setter):
                        raise RuntimeError(
                            f"lane {lane_id} cannot set its programmatic speed limit"
                        )
                    setter(expected_kmh)
                elif current_kmh != expected_kmh:
                    raise RuntimeError(
                        f"lane {lane_id} no longer has the configured programmatic speed limit"
                    )
            elif 0.0 < current_kmh <= MAX_LANE_SPEED_LIMIT_KMH:
                preserved_count += 1
            else:
                raise ValueError(
                    f"lane {lane_id} speed limit {current_kmh} km/h is neither the "
                    "programmatic unset sentinel nor a legal explicit speed limit"
                )

            final_speed_limit: float = lane.speed_limit
            if not np.isfinite(final_speed_limit):
                raise RuntimeError(f"lane {lane_id} returned an invalid configured speed limit")
            final_kmh = float(final_speed_limit)
            if final_kmh == PROGRAMMATIC_SPEED_LIMIT_SENTINEL_KMH:
                raise RuntimeError(f"lane {lane_id} retained the programmatic speed-limit sentinel")
            expected_kmh = profile_limits.get(lane_id, self.configured_speed_limit_kmh)
            if lane_id in self._sentinel_lane_ids and final_kmh != expected_kmh:
                raise RuntimeError(f"lane {lane_id} did not retain the configured speed limit")
            if lane_id in profile_limits and lane_id in self._sentinel_lane_ids:
                profile_applied_count += 1
            label = f"{final_kmh:g}"
            counts[label] = counts.get(label, 0) + 1

        self._audit = {
            "speed_limit_sentinel_replaced_count": len(self._sentinel_lane_ids),
            "speed_limit_existing_preserved_count": preserved_count,
            "configured_programmatic_lane_speed_limit_kmh": self.configured_speed_limit_kmh,
            "block_speed_limit_profile_kmh": self.block_speed_limit_profile_kmh,
            "block_speed_limit_profile_applied_lane_count": profile_applied_count,
            "lane_speed_limit_kmh_counts": dict(
                sorted(counts.items(), key=lambda item: float(item[0]))
            ),
        }

    @staticmethod
    def _validate_speed_limit(speed_limit_kmh: float, name: str) -> None:
        if (
            not np.isfinite(speed_limit_kmh)
            or speed_limit_kmh <= 0.0
            or speed_limit_kmh > MAX_LANE_SPEED_LIMIT_KMH
        ):
            raise ValueError(f"{name} must be finite, positive, and no greater than 130 km/h")

    def _profile_limits(self, current_map: object, lanes: list[object]) -> dict[str, float]:
        profile = self.block_speed_limit_profile_kmh
        if profile is None:
            return {}
        blocks = getattr(current_map, "blocks", None)
        if not isinstance(blocks, list) or len(blocks) != len(profile) + 1:
            raise ValueError(
                "programmatic lane speed-limit profile must provide one value for each "
                "generated map block after the initial block"
            )
        lane_ids = {id(lane): repr(getattr(lane, "index", None)) for lane in lanes}
        result: dict[str, float] = {}
        for speed_limit_kmh, block in zip(profile, blocks[1:], strict=True):
            block_network = getattr(block, "block_network", None)
            get_all_lanes = getattr(block_network, "get_all_lanes", None)
            if not callable(get_all_lanes):
                raise RuntimeError("programmatic map block does not expose its lane network")
            block_lanes = cast(Iterable[object], get_all_lanes())
            for lane in block_lanes:
                lane_id = lane_ids.get(id(lane))
                if lane_id is None:
                    raise RuntimeError(
                        "programmatic map block exposes a lane outside the map network"
                    )
                previous = result.setdefault(lane_id, speed_limit_kmh)
                if previous != speed_limit_kmh:
                    raise RuntimeError(
                        "programmatic lane belongs to conflicting speed-limit blocks"
                    )
        return result


def model_lane_speed_limit_mps(lane: Any) -> tuple[float, bool]:
    """Return the raw model value and mask while rejecting unconfigured sentinels."""

    speed_limit: float | None = lane.speed_limit
    if speed_limit is None:
        return 0.0, False
    speed_limit_kmh = float(speed_limit)
    if not np.isfinite(speed_limit_kmh) or speed_limit_kmh < 0.0:
        raise ValueError(f"lane {lane.index!r} speed limit must be finite and non-negative")
    if speed_limit_kmh == 0.0:
        return 0.0, False
    if speed_limit_kmh == PROGRAMMATIC_SPEED_LIMIT_SENTINEL_KMH:
        raise RuntimeError(
            f"lane {lane.index!r} has raw speed limit {speed_limit!r} km/h: "
            "programmatic lane speed limit was not configured"
        )
    if speed_limit_kmh > MAX_LANE_SPEED_LIMIT_KMH:
        raise ValueError(
            f"lane {lane.index!r} speed limit {speed_limit!r} km/h exceeds "
            "the 130 km/h domain bound"
        )
    return speed_limit_kmh / 3.6, True
