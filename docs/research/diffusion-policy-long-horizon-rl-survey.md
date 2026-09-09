# Long-horizon RL for Diffusion Policies / Planners

> Status: research note, not an accepted project decision or implementation plan.
>
> Scope: summarize reinforcement-learning methods that improve the long-horizon behavior of diffusion-based decision policies, especially autonomous-driving planners and embodied-control policies. The note distinguishes environment actions, denoising-chain actions, latent actions, value estimation, and open-source implementation references.

## 1. Core distinction: what does RL treat as the action?

A conditional diffusion policy can be written as

\[
z_N \sim \mathcal N(0,I),\qquad
z_{k-1}\sim p_\theta(z_{k-1}\mid z_k,s),\qquad
\tau=z_0.
\]

There are several different places where RL can act:

1. **Final environment action / action chunk / trajectory** \(\tau\).
2. **A denoising transition** \(z_k\rightarrow z_{k-1}\).
3. **Initial latent noise** \(z_N\).
4. **A guidance or exploration variable** that conditions denoising.
5. **A higher-level motion plan** decoded by a lower-level controller.

These choices lead to different likelihoods, critics, credit-assignment paths, and updated parameters.

For trajectory planners, it is also useful to distinguish:

- **policy action**: the generated trajectory \(\tau\);
- **environment control**: steering / throttle / low-level commands used to execute part of that trajectory;
- **committed prefix**: only the first \(K\) steps are normally executed before replanning.

Therefore a long-horizon trajectory critic is more naturally defined for a receding-horizon macro action than for an open-loop execution of the entire predicted horizon.

## 2. Method taxonomy

| Method family | RL action | Policy representation | Long-horizon signal | Main update target |
| --- | --- | --- | --- | --- |
| Diffusion-QL | final action / trajectory | implicit diffusion policy \(\pi_\theta(a\mid s)\) | Bellman \(Q(s,a)\) | diffusion actor |
| DIPO | final action, then Q-improved target | diffusion action generator | \(\nabla_a Q(s,a)\) | denoiser fitted to improved actions |
| QVPO / DPMD / SDAC | final action | conditional diffusion policy | Q / advantage weighting | denoising objective |
| DACER | final action | differentiable full sampler \(a=F_\theta(s,z)\) | critic \(Q(s,a)\) | full sampling chain by BPTT |
| DPPO | outer action chunk; inner denoising transition | reverse kernel \(p_\theta(z_{k-1}\mid z_k,s)\) | environment GAE / PPO | denoising transitions + value critic |
| DIVER / Multi-ORFT | trajectory group + denoising transitions | diffusion trajectory policy | group-relative reward / advantage | denoising policy |
| PlannerRFT | trajectory + guidance scales | DiT diffusion policy + exploration policy | GRPO for DiT; closed-loop GAE/PPO for guidance | DiT and exploration policy |
| DSRL | latent noise \(z_N\) | latent policy \(\pi_\phi(z\mid s)\) + frozen diffusion decoder | actor-critic return | latent policy only |
| REFINE-DP | high-level plan + low-level control | hierarchical diffusion planner and RL controller | task return + tracking / execution feedback | both levels |
| World-model RL | ordinary actor action | separate actor; diffusion model is environment model | imagined long-horizon return | actor, not necessarily diffusion policy |

## 3. Final-action / trajectory value methods

### 3.1 Diffusion-QL

Diffusion-QL treats the final diffusion sample as the RL action. The policy is still the marginal distribution induced by the whole denoising process:

\[
a\sim\pi_\theta(a\mid s).
\]

Its actor objective combines a diffusion behavior-modeling term with a value-improvement term:

\[
\mathcal L_{actor}
=
\mathcal L_{diffusion}
-
\lambda\,\mathbb E_{a\sim\pi_\theta}[Q_\phi(s,a)].
\]

The diffusion term keeps the actor close to the data-supported behavior distribution; the Q term increases the probability of actions with better long-term value. The denoising chain is not itself an environment-time MDP.

For a trajectory planner, the direct extension is

\[
Q(s,a)\rightarrow Q(s,\tau),
\]

where \(\tau\) is a high-dimensional future trajectory or action chunk.

Primary source: [Diffusion Policies as an Expressive Policy Class for Offline Reinforcement Learning](https://arxiv.org/abs/2208.06193).

### 3.2 DIPO

DIPO also treats the final generated sample as the environment action, but does not directly optimize the marginal diffusion-policy likelihood. Instead, after learning a critic, an action is improved by

\[
\tilde a = a + \eta\nabla_a Q(s,a),
\]

and the diffusion model is trained to fit the improved action targets. RL therefore changes the supervision data rather than introducing a denoising-chain policy-gradient objective.

Primary source: [DIPO: Diffusion Policy Optimization with Diffusion Models](https://arxiv.org/abs/2305.13122).

### 3.3 Q-weighted denoising: QVPO / DPMD / SDAC

These methods retain a diffusion-style training loss but reweight samples according to Q or advantage information:

\[
\mathcal L_\pi
=
\mathbb E\left[
  w(s,a)\,
  \|\epsilon-\epsilon_\theta(z_k,k,s)\|^2
\right],
\]

with \(w\) increasing for higher-value actions. The RL signal decides which actions should dominate diffusion fitting; denoising steps are not usually treated as separate environment actions.

Representative sources:

- [QVPO: Q-Value Regularized Diffusion Policy Optimization](https://arxiv.org/abs/2405.16173)
- [Diffusion Policy Mirror Descent](https://proceedings.mlr.press/v267/ma25d.html)

### 3.4 DACER: differentiable diffusion actor

DACER treats the entire sampler as a differentiable actor

\[
a=F_\theta(s,z_N),
\]

and backpropagates critic gradients through the complete denoising computation:

\[
\nabla_\theta J
=
\mathbb E\left[
\nabla_aQ_\phi(s,a)
\frac{\partial F_\theta(s,z_N)}{\partial\theta}
\right].
\]

Compared with DIPO, the Q gradient is not first converted into a new action target; it propagates directly through the sampling chain. The cost is memory / stability pressure when the action dimension, trajectory horizon, or denoising-step count becomes large.

Primary source: [DACER](https://arxiv.org/abs/2405.15177).

## 4. Explicit denoising-MDP methods

### 4.1 DPPO

DPPO introduces two nested time scales.

**Outer environment MDP**:

\[
s_t\rightarrow \tau_t\rightarrow s_{t+1},r_t.
\]

The diffusion policy may generate a multi-step action chunk; the environment executes only the first \(K\) actions and replans.

**Inner denoising MDP**:

\[
\tilde s_k=(s_t,z_k,k),\qquad
\tilde a_k=z_{k-1},
\]

with policy

\[
\tilde\pi_\theta(\tilde a_k\mid\tilde s_k)
=
p_\theta(z_{k-1}\mid z_k,s_t).
\]

This avoids requiring the intractable marginal likelihood \(\pi_\theta(\tau\mid s)\). PPO can instead use the transition probability of each reverse-diffusion step:

\[
\rho_{t,k}
=
\frac{p_\theta(z_{k-1}\mid z_k,s_t)}
     {p_{\theta_{old}}(z_{k-1}\mid z_k,s_t)}.
\]

The outer environment rollout provides reward, value estimates, and GAE. The same environment-time advantage is then associated with the denoising transitions that generated the executed action chunk. Implementations therefore store the whole diffusion chain in addition to ordinary PPO rollout fields.

Primary source: [Diffusion Policy Policy Optimization](https://arxiv.org/abs/2409.00588).

Official implementation: [irom-princeton/dppo](https://github.com/irom-princeton/dppo).

### 4.2 Group-relative diffusion optimization

For high-dimensional trajectories, learning a precise \(Q(s,\tau)\) critic can be difficult. A different family generates several trajectories under the same state,

\[
\tau^{(1)},\ldots,\tau^{(G)}\sim\pi_\theta(\tau\mid s),
\]

scores each candidate, and computes a group-relative advantage such as

\[
\hat A_i
=
\frac{R_i-\bar R}{\operatorname{std}(R)+\epsilon}.
\]

The policy then increases the likelihood of denoising transitions associated with above-average trajectories and decreases it for below-average trajectories. This removes the need for a separate state-value baseline for the trajectory branch, although it requires enough reward variance within each group.

Representative autonomous-driving works include DIVER, PlannerRFT, and Multi-ORFT.

## 5. PlannerRFT: dual-branch optimization for autonomous-driving diffusion planners

PlannerRFT is particularly useful for separating two different RL roles.

### 5.1 Exploration Policy: guidance action + PPO

The pretrained / reference planner produces a reference trajectory \(x^{ref}\). A separate Exploration Policy outputs parameters of Beta distributions for lateral and longitudinal guidance scales:

\[
\eta=(\eta_{lat},\eta_{lon})
\sim
\pi_\phi(\eta\mid s,x^{ref}).
\]

The guidance scales condition energy-based classifier guidance during denoising. This branch is optimized by closed-loop PPO. The Exploration Policy also contains a value head \(V_\psi(s_t)\), and future rewards are propagated through GAE. Therefore this branch explicitly uses environment-time long-horizon feedback to decide which exploration direction is appropriate in the current scenario.

PlannerRFT stores closed-loop tuples including state, selected guidance scale, reward, and value prediction. Only the first action of the selected planning trajectory is executed before the next planning step.

### 5.2 Fine-tuned DiT: trajectory group + GRPO

The trajectory branch generates a group of candidate trajectories under different guidance samples. The paper evaluates the trajectories over a prediction horizon and uses GRPO to update the Diffusion Transformer.

Following DPPO, each DDIM transition is modeled as a Gaussian policy:

\[
\pi_\theta(x_{s-1}\mid x_s)
=
\mathcal N(x_{s-1};\mu_\theta(x_s,s),\sigma_s^2 I).
\]

The group-relative trajectory advantage is applied to the log probability of the denoising transitions, with a denoising discount factor. A behavior-cloning term is retained to reduce policy collapse.

PlannerRFT uses 5-step DDIM during RFT to preserve stochastic exploration while keeping the number of denoising steps small. During evaluation, it uses deterministic sampling and can remove the reference / exploration modules, leaving the fine-tuned diffusion planner itself.

### 5.3 Survival reward and reward horizon

For hard scenarios, a pure terminal reward can collapse an entire candidate group to the same zero reward once all candidates eventually fail. PlannerRFT therefore introduces a survival reward that accumulates valid rewards until terminal failure, so a trajectory that fails later can still rank above one that fails immediately.

The reported ablation shows that a 2 s trajectory-reward horizon is weaker, while 4 s and 6 s are similar; the paper uses 4 s in its main setting. The Exploration Policy and DiT therefore receive different temporal forms of supervision:

- Exploration Policy: closed-loop environment return + GAE;
- DiT trajectory branch: group-relative score over a moderate prediction horizon + survival reward.

Primary source: [PlannerRFT](https://arxiv.org/abs/2601.12901), project page [OpenDriveLab PlannerRFT](https://opendrivelab.com/PlannerRFT/).

## 6. Latent / guidance-policy RL

### 6.1 DSRL

DSRL freezes the pretrained diffusion policy and changes the distribution of its initial latent noise:

\[
z_N\sim\pi_\phi^Z(z\mid s),
\qquad
\tau=F_{\theta_0}(s,z_N).
\]

The RL action is the latent \(z_N\), while the environment executes the trajectory decoded by the frozen diffusion policy. A critic can be defined over \(Q(s,z_N)\). RL therefore learns where to sample on the pretrained model's latent manifold without modifying the denoiser weights.

Primary source: [Steering Your Diffusion Policy with Latent Space Reinforcement Learning](https://arxiv.org/abs/2506.15799).

### 6.2 Guidance-policy optimization

PlannerRFT's Exploration Policy belongs to a related family: the RL action is not the final trajectory itself, but a low-dimensional guidance variable that changes how the diffusion generator explores. The effective policy is hierarchical:

\[
\pi(\tau,\eta\mid s)
=
\pi_\phi(\eta\mid s,x^{ref})
\pi_\theta(\tau\mid s,\eta).
\]

This separates long-horizon selection of an exploration mode from short-horizon / trajectory-level diffusion optimization.

## 7. Hierarchical and world-model extensions

### 7.1 REFINE-DP

REFINE-DP separates a high-level diffusion motion planner and a low-level RL controller. The high-level output is a future motion plan / command sequence; the low-level action is the robot-control command that must track that plan. Both levels are optimized so that high-level policy improvement does not move the planner distribution outside the low-level controller's executable region.

Primary source: [REFINE-DP](https://arxiv.org/abs/2603.13707).

### 7.2 World4RL

World4RL uses a diffusion model as a learned world model rather than necessarily as the policy. A separate actor is improved from imagined long-horizon rollouts generated by the frozen diffusion world model. The long-horizon benefit comes from replacing costly real interaction with model-generated rollout, but performance then depends on world-model fidelity and error accumulation.

Primary source: [World4RL](https://arxiv.org/abs/2509.19080).

## 8. Long-horizon value functions for trajectory planners

### 8.1 \(V(s)\): remaining value under the current policy

\[
V^\pi(s_t)
=
\mathbb E_\pi\left[
\sum_{i=0}^{T-t-1}\gamma^i r_{t+i}
+
\gamma^{T-t}r_T^{terminal}
\mid s_t
\right].
\]

For long-distance driving, the state representation must distinguish otherwise similar local scenes with different remaining tasks. Candidate state variables therefore include route progress, remaining route distance, elapsed time, navigation / road preview, and dynamic traffic context in addition to local ego state.

If the observation is not Markov, the value function is better interpreted as \(V(h_t)\) over an encoded observation / action history.

### 8.2 \(Q(s,\tau)\): value of committing to a trajectory prefix

If the planner predicts \(H\) future steps but only the first \(K\) are executed before replanning, a trajectory-value definition is

\[
Q_K^\pi(s_t,\tau_t)
=
\mathbb E\left[
\sum_{i=0}^{K-1}\gamma^i r_{t+i}
+
\gamma^K V^\pi(s_{t+K})
\mid s_t,\tau_t
\right].
\]

This is more accurate than assuming the full \(H\)-step trajectory is executed open loop. The trajectory \(\tau\) is a macro action; \(K\) is the actual commitment horizon.

A standard training target is

\[
y_t^Q
=
\sum_{i=0}^{K-1}\gamma^i r_{t+i}
+
\gamma^K V_{\bar\phi}(s_{t+K}),
\]

with

\[
\mathcal L_Q=(Q_\psi(s_t,\tau_t)-y_t^Q)^2.
\]

A high-dimensional trajectory can be encoded separately:

\[
z_s=f_s(s),\qquad z_\tau=f_\tau(\tau),\qquad Q=g(z_s,z_\tau),
\]

where \(f_\tau\) can be a temporal Transformer, MLP-Mixer, or another sequence encoder.

### 8.3 Relation between \(V\), \(Q\), and advantage

\[
V^\pi(s)
=
\mathbb E_{\tau\sim\pi(\cdot\mid s)}Q^\pi(s,\tau),
\]

\[
A^\pi(s,\tau)=Q^\pi(s,\tau)-V^\pi(s).
\]

In PPO / DPPO, \(V(s)\) is the natural baseline for GAE. A trajectory-conditioned critic is more directly useful for candidate ranking, Q-guided sampling, off-policy actor-critic learning, or action-gradient methods.

## 9. Long-distance metrics and critic targets

### 9.1 Additive metrics

Energy, progress, travel time, and comfort can be represented as step-wise increments:

\[
r_t^E=-\Delta E_t,\qquad
r_t^P=\Delta d_t,\qquad
r_t^T=-\Delta t,\qquad
r_t^J=-j_t^2\Delta t.
\]

Corresponding value heads estimate remaining expected energy, progress, time, or comfort cost.

### 9.2 Terminal / event metrics

Collision, completion, off-road, and similar events can be modeled as probabilities:

\[
V_{collision}(s)
=P(\text{collision before termination}\mid s),
\]

\[
Q_{collision}(s,\tau)
=P(\text{collision after committing to }\tau\mid s,\tau).
\]

These can be learned as cost critics or probabilistic heads rather than being forced into the same scalar regression target as energy.

### 9.3 Ratio metrics

A route-level metric such as energy intensity

\[
\frac{E_{total}}{D_{total}}
\]

is not additive. Using the local ratio \(-\Delta E_t/\Delta d_t\) as an immediate reward can be unstable near zero progress and is not generally equivalent to optimizing the episode-level ratio.

For fixed route length, minimizing total energy is equivalent to minimizing energy per route distance. For variable route completion, the numerator and denominator can instead be estimated separately, e.g. remaining energy and remaining progress / distance.

### 9.4 Multi-head value estimation

A vector critic can preserve the semantics and scales of different objectives:

\[
\mathbf V(s)
=
[V_E,V_P,V_C,V_J,V_T]^T,
\]

and similarly

\[
\mathbf Q(s,\tau)
=
[Q_E,Q_P,Q_C,Q_J,Q_T]^T.
\]

Scalarization or constraints can then be applied at the policy-objective level rather than hiding all objectives inside one value target.

This is a research extension, not a claim that multi-head critics are always preferable to a scalar critic.

## 10. Open-source implementation references

### Verified useful repositories

1. **DPPO** — [irom-princeton/dppo](https://github.com/irom-princeton/dppo)  
   Useful for diffusion-chain storage, reverse-step log probabilities, PPO ratios, GAE/value learning, and receding-horizon action chunks.

2. **Diffusion-QL** — [Zhendong-Wang/Diffusion-Policies-for-Offline-RL](https://github.com/Zhendong-Wang/Diffusion-Policies-for-Offline-RL)  
   Useful for a diffusion actor with Q-based policy improvement in offline RL.

3. **Diffusion Planner** — [ZhengYinan-AIR/Diffusion-Planner](https://github.com/ZhengYinan-AIR/Diffusion-Planner)  
   Baseline autonomous-driving diffusion planner; relevant for trajectory representation, scene conditioning, joint ego/neighbor generation, and classifier guidance.

### Papers / methods whose code availability should be checked separately before reuse

- DIPO
- QVPO / DPMD / SDAC
- DACER
- DIVER
- PlannerRFT / nuMax
- Multi-ORFT
- DSRL
- REFINE-DP
- World4RL

The method descriptions above do not assume that these repositories are currently released, complete, or directly reusable.

## 11. Autonomous-driving-specific observations from PlannerRFT and Diffusion Planner

The original Diffusion Planner jointly generates ego planning and neighboring-agent predictions as a future-trajectory diffusion problem. It predicts an 8 s horizon at 10 Hz and supports training-free classifier guidance for target speed, comfort, collision avoidance, and drivable-area compliance.

PlannerRFT starts from this type of IL-pretrained diffusion planner and changes the RFT sampling process in several important ways:

- duplicates and freezes a reference planner during RFT;
- uses a 5-step stochastic DDIM sampler instead of the original ODE-based DPM-Solver during RL fine-tuning;
- adds lateral / longitudinal energy-guidance variables;
- learns those guidance variables with a Beta-distribution Exploration Policy and PPO;
- updates the fine-tuned DiT with group-relative trajectory optimization over Gaussian denoising transitions;
- adds survival reward to preserve ranking information in hard cases;
- reports stronger results with a moderate trajectory-reward horizon than with a very short 2 s horizon;
- removes the extra reference / exploration modules at deployment and keeps the fine-tuned planner.

This distinction matters when comparing long-horizon RL strategies: the Exploration Policy solves an environment-time credit-assignment problem, while the DiT branch solves a trajectory-distribution refinement problem.

## 12. Research interpretation boundaries

The surveyed methods answer different questions and should not be treated as interchangeable implementations of the same objective:

- a longer predicted trajectory does **not** by itself provide long-horizon RL credit;
- inference-time classifier / cost guidance is **not** RL unless its parameters are updated from return, advantage, Q, or another RL objective;
- explicit diffusion-MDP methods solve the likelihood / credit path through denoising, but still require a meaningful environment-time return;
- \(Q(s,\tau)\) methods require a critic that can generalize over a high-dimensional trajectory action;
- group-relative methods avoid a trajectory critic but need informative within-group reward variation;
- latent / guidance policies restrict RL to a lower-dimensional interface and may leave the base diffusion model frozen;
- hierarchical and world-model methods move part of the long-horizon problem outside the diffusion actor itself.

## References

- Zheng et al., **Diffusion-based Planning for Autonomous Driving with Flexible Guidance**, ICLR 2025. [Paper](https://arxiv.org/abs/2501.15564) · [Code](https://github.com/ZhengYinan-AIR/Diffusion-Planner)
- Ren et al., **Diffusion Policy Policy Optimization**, ICLR 2025. [Paper](https://arxiv.org/abs/2409.00588) · [Code](https://github.com/irom-princeton/dppo)
- Li et al., **PlannerRFT: Reinforcing Diffusion Planners through Closed-Loop and Sample-Efficient Fine-Tuning**, 2026. [Paper](https://arxiv.org/abs/2601.12901) · [Project](https://opendrivelab.com/PlannerRFT/)
- Wang et al., **Diffusion Policies as an Expressive Policy Class for Offline Reinforcement Learning**, 2022. [Paper](https://arxiv.org/abs/2208.06193) · [Code](https://github.com/Zhendong-Wang/Diffusion-Policies-for-Offline-RL)
- Yang et al., **DIPO**, 2023. [Paper](https://arxiv.org/abs/2305.13122)
- QVPO, 2024. [Paper](https://arxiv.org/abs/2405.16173)
- DACER, 2024. [Paper](https://arxiv.org/abs/2405.15177)
- Wagenmaker et al., **Steering Your Diffusion Policy with Latent Space Reinforcement Learning**, 2025. [Paper](https://arxiv.org/abs/2506.15799)
- **REFINE-DP**, 2026. [Paper](https://arxiv.org/abs/2603.13707)
- **World4RL**, 2025. [Paper](https://arxiv.org/abs/2509.19080)
