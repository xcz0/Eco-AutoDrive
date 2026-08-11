# Add reference-centered orthogonal guidance

Use one frozen official-EMA Diffusion Planner instance to produce both the reference and guided
DDIM trajectories. For each planning cycle, share one scene encoding, initial standard-normal
noise, and DDIM transition draws. Refresh the reference every planning cycle.

Define tangent from the normalized reference heading and use its left normal. Derive 10 Hz
velocity from the current point followed by 80 future points. Repeated positions are valid zero
speed; non-finite trajectories and degenerate headings fail.

Use a project-defined centered energy-gradient delta so action `(0, 0)` is exactly neutral. The
lateral target is at most `2.5 m`; the longitudinal target is at most `25%` of reference
along-track speed. Differentiate the physical ego objective through checkpoint normalization and
the frozen DiT with respect to the normalized noisy joint sample. After every DDIM transition,
apply the unit-coefficient negative gradient only to ego future channels, reapply the current-state
constraint, and detach before the next transition. Preserve and audit the masked neighbor
gradient rather than applying it.

This discretization, neutral-action correction, unit injection coefficient, and gradient scope are
project reproduction decisions because PlannerRFT does not publish those implementation details.
Active guidance is restricted to the standard-Gaussian DDIM-5 profile. DPM-10 and the isolated
`0.5 * N(0,I)` DDIM variant remain unguided controlled baselines.
