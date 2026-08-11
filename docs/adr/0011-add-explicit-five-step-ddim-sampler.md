# Add an explicit five-step DDIM sampler

Keep the official 10-step DPM-Solver++ profile as the default, immutable baseline. Add DDIM as a
separate Hydra-selected sampling boundary so its initial distribution, time schedule and
stochasticity remain visible research variables.

The PlannerRFT paper-text profile starts from standard Gaussian future noise. It makes five
continuous-time DDIM transitions, evaluating the denoiser at `t = [1.0, 0.8, 0.6, 0.4, 0.2]` and
ending each transition at `[0.8, 0.6, 0.4, 0.2, 0.0]`. Evaluation uses
`ddim_stochasticity = 0`. Non-zero stochasticity uses the episode's explicit diffusion generator;
the clean endpoint does not consume a random draw.

The paper does not publish its DDIM timestep subsequence. The uniform continuous-time schedule is
therefore a project reproduction decision, not an author fact. A separately labelled
`0.5 * N(0,I)` profile is retained only to isolate the initial-noise-scale difference from the
official baseline and cannot be reported as PlannerRFT parity.
