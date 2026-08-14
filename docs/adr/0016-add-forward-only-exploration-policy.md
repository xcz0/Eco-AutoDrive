# Add a forward-only Exploration Policy

Add a standalone Exploration Policy that consumes one frozen scene encoding, one frozen
navigation encoding, their padding masks, and the ego-local physical reference trajectory. The
reference trajectory is encoded by a configurable MLP-Mixer and used as the query for masked
cross-attention over the scene and navigation tokens. Actor and value outputs share the fused
trunk. The policy is not connected to the evaluation runner until rollout and reward contracts
exist.

Use one independent Beta distribution for lateral guidance and one for longitudinal guidance.
The stored action is the base sample `u` in `(0, 1)^2`; the audited guidance is `2u - 1` in
`(-1, 1)^2`. Training sampling uses an explicit policy RNG and reparameterized samples;
deterministic evaluation uses the Beta mean. Account for the affine transform exactly in joint
guidance log-probability and entropy. Reject boundary actions rather than clipping them.

Parameterize concentrations as `softplus(raw) + minimum_concentration`. Initialize the actor head
with zero weights and equal biases so both guidance means are exactly zero while the initial Beta
variance remains nonzero. Network dimensions, depth, attention heads, dropout, initial
concentration, and minimum concentration are required Hydra fields. These architecture,
parameterization, affine action, and initialization choices are project reproduction decisions;
PlannerRFT does not publish them.

Keep the official planner frozen and outside the policy module. Feature extraction invokes the
existing scene encoder and route encoder once under `no_grad`. Policy checkpoints contain exactly
the policy state dict and a strict format version; they never contain planner parameters,
optimizer state, rollout data, or RNG state.
