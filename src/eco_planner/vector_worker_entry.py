"""Torch-free multiprocessing entrypoint for Windows MetaDrive workers."""

from __future__ import annotations

from collections.abc import Mapping
from multiprocessing.connection import Connection
from typing import Any


def worker_main(connection: Connection, launch_payload: Mapping[str, Any]) -> None:
    """Import the environment runtime only after the spawned process is ready."""

    from eco_planner.envs.vector_metadrive import _worker_main_from_payload

    _worker_main_from_payload(connection, launch_payload)
