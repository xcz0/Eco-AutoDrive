# Use the MetaDrive score for Stage-6 smoke training

Stage 6 optimizes the unmodified MetaDrive `metadrive_builtin_v1` score only to validate the
closed-loop PPO data path. The profile pins MetaDrive 0.4.3 reward settings: driving reward `1.0`,
speed reward `0.1`, success reward `10.0`, out-of-road/vehicle/object collision penalties `5.0`,
sidewalk penalty `0.0`, and lateral reward disabled. The returned scalar has unit
`dimensionless_score`. Its dense `step_reward` and any terminal overwrite delta are persisted
separately; route progress, distance, speed, stopping, termination, and execution error are audit
metrics rather than additional optimized terms.

This closes G-07 only for the Stage-6 smoke objective. It does not define the PlannerRFT paper
reward, an energy reward, or a Stage-7 comparison objective. A rising MetaDrive score is not
evidence of safety, energy improvement, or paper parity without the separately recorded audits.

The smoke runtime uses one CUDA BF16 Fabric runtime and two serial logical environment slots. Each
slot contributes 16 10-Hz transitions per update; four updates are performed. Diffusion and policy
action generators remain distinct and persistent across updates. A fixed SeedSequence namespace
derives and records both streams from the training seed. Multi-process or vectorized scale-out
remains G-09 work.

Training artifacts use independent schema version 1. They retain complete transition contexts,
actions, probability values, initial noise, RNG states, reward/audit fields, checkpoints, runtime
metadata, and classified partial failures. Only the Exploration Policy is checkpointed and updated;
the frozen planner parameter hash must remain unchanged.
