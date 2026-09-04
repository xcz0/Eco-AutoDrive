# Separate online proxy and offline FASTSim energy metrics

**Status:** Accepted and implemented
**Date:** 2026-09-03

The current kinematic MetaDrive execution needs a cheap per-transition energy signal for reward and
artifact continuity, while FASTSim is a stateful backward-looking vehicle model whose result depends
on the complete time/speed cycle. Resetting FASTSim for every 0.1 s transition would repeatedly reset
the powertrain state and would not represent one continuous drive.

Environment-domain energy providers therefore share an objective-neutral
`EnergyTrace -> EnergyMetrics` boundary in `envs.domain.metrics` while retaining distinct metric
names and native units. The provider protocol is defined with those value contracts; the online
proxy lives in `envs.domain.energy` and the offline adapter in `envs.domain.fastsim`. The online
environment uses
`metadrive_fuel_proxy`, evaluated from actual executed distance and endpoint speed. Existing reward,
trace, rollout, and evaluation fields continue to record this metric in mL, with unchanged numerical
semantics.

`fastsim_fuel_energy` is an offline full-trace provider. Its first supported vehicle is FASTSim's
bundled conventional `2012_Ford_Fusion.yaml`; grade, ambient temperature, and initial elevation are
explicit provider configuration. The adapter uses the locked FASTSim 3.0.6 `Cycle`, `Vehicle`, and
`SimDrive.walk()` interfaces and reports cumulative fuel energy in J/Wh. It does not invent a fuel
volume conversion and does not enter online reward or the default evaluation artifact schema.

Zero distance leaves per-distance intensity undefined for either provider. FASTSim may still report
positive idle or auxiliary energy for a stationary trace. Provider failures, unsupported vehicle
types, trace misses, and distance disagreement are surfaced directly; no provider fallback is used.

The two metrics may be evaluated on the same executed trace for comparison, but their totals must not
be combined or presented as the same physical quantity. A future decision is required before FASTSim
can become an online training signal or a persisted evaluation metric.
