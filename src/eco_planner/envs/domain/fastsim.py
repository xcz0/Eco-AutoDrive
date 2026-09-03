"""Offline full-trace energy measurement through FASTSim 3."""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field, StrictFloat

from eco_planner.envs.domain.metrics import EnergyMetrics, EnergyTrace

_FUEL_ENERGY_PATH = "veh.pt_type.Conv.fc.state.energy_fuel_joules"
_DISTANCE_PATH = "veh.state.dist_meters"


class FASTSimEnergyConfig(BaseModel):
    """Explicit vehicle and flat-cycle conditions for one FASTSim provider."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)

    vehicle_resource: str = Field(min_length=1)
    grade: StrictFloat
    ambient_temperature_k: StrictFloat = Field(gt=0.0)
    initial_elevation_m: StrictFloat


class FASTSimEnergyProvider:
    """Run a fresh conventional-vehicle FASTSim model over each complete trace."""

    def __init__(self, config: FASTSimEnergyConfig) -> None:
        import fastsim

        vehicle = fastsim.Vehicle.from_resource(config.vehicle_resource)
        if vehicle.veh_type() != "Conv":
            raise ValueError("FASTSim energy provider currently requires a conventional vehicle")
        self._config = config

    def measure(self, trace: EnergyTrace) -> EnergyMetrics:
        import fastsim

        cycle = fastsim.Cycle.from_pydict(self._cycle_payload(trace))
        vehicle = fastsim.Vehicle.from_resource(self._config.vehicle_resource)
        vehicle.set_save_interval(None)
        simulation = fastsim.SimDrive(vehicle, cycle)
        simulation.walk()
        result = simulation.to_pydict(flatten=True)
        model_distance_m = float(result[_DISTANCE_PATH])
        if not math.isclose(model_distance_m, trace.distance_m, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(
                "FASTSim cycle distance does not match the executed trace distance: "
                f"{model_distance_m} != {trace.distance_m}"
            )
        return EnergyMetrics(
            metric="fastsim_fuel_energy",
            distance_m=trace.distance_m,
            energy_j=float(result[_FUEL_ENERGY_PATH]),
            fuel_ml=None,
        )

    def _cycle_payload(self, trace: EnergyTrace) -> dict[str, float | list[float]]:
        point_count = trace.time_s.size
        zeros = [0.0] * point_count
        elevations = [float(self._config.initial_elevation_m)] * point_count
        return {
            "init_elev_meters": float(self._config.initial_elevation_m),
            "time_seconds": trace.time_s.tolist(),
            "speed_meters_per_second": trace.speed_mps.tolist(),
            "dist_meters": zeros,
            "grade": [float(self._config.grade)] * point_count,
            "elev_meters": elevations,
            "pwr_max_chrg_watts": zeros.copy(),
            "temp_amb_air_kelvin": [float(self._config.ambient_temperature_k)] * point_count,
            "pwr_solar_load_watts": zeros.copy(),
            "grade_interp": 0.0,
            "elev_interp": float(self._config.initial_elevation_m),
        }


__all__ = ["FASTSimEnergyConfig", "FASTSimEnergyProvider"]
