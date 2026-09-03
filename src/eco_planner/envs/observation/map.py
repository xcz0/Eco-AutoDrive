"""Immutable map data consumed by the pure observation builder."""

from __future__ import annotations

from dataclasses import dataclass

from ..array_types import NumpyMapObservation


@dataclass(frozen=True, slots=True)
class MapSnapshot:
    """Reset-time map arrays, independent of the simulator that extracted them."""

    arrays: NumpyMapObservation

    def for_observation(self) -> NumpyMapObservation:
        """Return the current local-map arrays prepared by the map integration boundary."""

        return self.arrays
