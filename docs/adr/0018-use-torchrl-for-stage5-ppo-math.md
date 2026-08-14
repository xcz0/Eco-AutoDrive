# Use TorchRL for Stage-5 GAE and PPO mathematics

Stage 5 uses the locked TorchRL 0.12.0 and TensorDict 0.12.4 APIs for generalized advantage
estimation and clipped PPO loss. Each `RolloutEpisode` is converted independently: its final item
is a recursion boundary, `terminated` suppresses bootstrap, and pure truncation or a bounded
rollout tail uses the explicitly stored tail value. Episodes are flattened only after GAE, so an
advantage cannot cross a collector boundary.

The existing Exploration Policy remains the single parameter owner. TensorDict actor and critic
adapters share that module, and a transformed Torch distribution implements the established
`u -> 2u - 1` action contract, including the affine Jacobian in log-probability and entropy. PPO
uses transformed guidance actions and stored transformed old log-probabilities; DDIM transition
probability is not part of the ratio.

Advantages are normalized once over the complete rollout batch with the sample standard
deviation. Fewer than two samples, zero variance, and non-finite values fail instead of being
clamped. The value objective is unclipped L2. Actor, value head, and their shared trunk receive the
combined policy, value, and entropy gradients. Adam uses explicit epsilon and zero weight decay;
the cosine schedule advances after every optimizer step and has an explicit optimizer-step
horizon. The policy stays in eval mode during collection and optimization so dropout cannot alter
old/new probability ratios while gradient recording remains enabled for updates.

This decision supplies a mathematical updater and a synthetic smoke profile only. It does not
select a MetaDrive optimization reward, add a closed-loop training entrypoint, persist optimizer
state, define a paper-scale scheduler horizon, or close G-07/G-09. `metadrive_builtin_v1` remains a
non-parity Stage-4 transported score until a separate reward decision is made.
