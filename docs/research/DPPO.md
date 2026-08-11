不是“完整预测轨迹全部运行完后才获得一次奖励”。DPPO 代码里更准确的粒度是：

> 每个 environment rollout step，扩散策略先生成一个完整 action horizon，但只执行前 `act_steps` 个动作；这 `act_steps` 个底层环境步分别产生 reward，随后被求和成一个外层 reward \(R_t\)。

训练代码明确先生成 `samples.trajectories`，截取 `output_venv[:, :self.act_steps]`，然后一次传给 `self.venv.step(action_venv)`；该返回值被记录为 `reward_trajs[step]`。

而 `MultiStep` wrapper 内部其实逐个执行这些动作：

```python
for act in action:
    observation, reward, done, info = self.env.step(act)
    self.reward.append(reward)

reward = aggregate(self.reward, "sum")
```

因此默认：

\[
\boxed{
R_t=\sum_{j=0}^{N_a-1}r_{t,j}
}
\]

其中 \(N_a=\text{act\_steps}\)。如果中间 episode 已终止，则后面的动作不会继续执行。

这点对理解 DPPO 很重要。

### 三种时间尺度要分开

假设：

\[
\text{horizon\_steps}=16,\qquad
\text{act\_steps}=4,\qquad
K=10
\]

一次外层决策实际上是：

```text
环境状态 s_t

        ↓

随机噪声 x_K
        ↓ denoise
x_{K-1}
        ↓
...
        ↓
x_0

x_0 = [a0, a1, ..., a15]
        ↓

只执行 [a0, a1, a2, a3]

        ↓
env.step(a0) → r0
env.step(a1) → r1
env.step(a2) → r2
env.step(a3) → r3

        ↓

R_t = r0 + r1 + r2 + r3

        ↓

得到新状态 s_{t+1}

        ↓

重新运行一次 diffusion
```

所以，**扩散去噪的 \(K\) 步没有环境 reward**。环境甚至不知道 \(x_K\rightarrow\cdots\rightarrow x_0\) 这个过程存在。

环境只看到最终生成出来的动作。

---

## Reward 最终怎么传给 diffusion 各步？

这是 DPPO 的另一个关键点。

在环境层面，第 \(t\) 个 rollout step 只有一个：

\[
R_t
\]

然后像普通 PPO 一样计算 critic：

\[
V(s_t)
\]

以及 GAE：

\[
\delta_t
=
R_t+\gamma V(s_{t+1})-V(s_t)
\]

\[
A_t
=
\delta_t+
\gamma\lambda A_{t+1}
\]

代码就是这样反向遍历 `reward_trajs` 计算 `advantages_trajs`。

注意：这里的 \(t\) 是 **environment step**，不是 diffusion step。

之后训练 diffusion policy 时，对于这个 environment step 对应的完整扩散链：

\[
x_K\rightarrow x_{K-1}\rightarrow\cdots\rightarrow x_0
\]

每一个 diffusion transition：

\[
x_k\rightarrow x_{k-1}
\]

都使用这个环境状态对应的优势：

\[
A_t
\]

但再乘一个 diffusion MDP 内部的 discount：

\[
A_{t,k}
=
\gamma_{\text{denoising}}^{K-k-1}A_t
\]

代码中就是：

```python
discount = gamma_denoising ** (...)
advantages *= discount
```



所以可以把它理解成：

\[
\boxed{
\text{环境 reward}
\rightarrow
A_t
\rightarrow
\text{分配给该环境决策对应的所有 diffusion steps}
}
\]

而不是：

\[
x_K \to r_K,\quad
x_{K-1}\to r_{K-1},\dots
\]

扩散过程本身没有独立的环境奖励。

---

## 一次 DPPO rollout 的伪代码

下面基本对应 `TrainPPODiffusionAgent.run()` 的代码结构。

```python
# ==========================================
# One DPPO rollout / iteration
# ==========================================

obs = env.current_observation

# rollout buffers
obs_buffer = []
reward_buffer = []
done_buffer = []
diffusion_chain_buffer = []

# ------------------------------------------
# 1. Collect environment rollout
# ------------------------------------------

for env_t in range(n_steps):

    # ---- 当前环境状态 ----
    s_t = obs

    # ---- Diffusion policy sampling ----
    # x_K ~ N(0, I)
    #
    # x_K -> x_{K-1} -> ... -> x_0
    #
    # x_0 是完整 action horizon

    with no_grad():

        trajectory, chain = diffusion_policy(
            condition=s_t,
            return_chain=True
        )

        # trajectory:
        # [horizon_steps, action_dim]
        #
        # chain:
        # [K+1, horizon_steps, action_dim]


    # ----------------------------------
    # 只执行前 act_steps 个动作
    # ----------------------------------

    action_chunk = trajectory[:act_steps]


    # MultiStep wrapper 内部等价于：

    R_t = 0

    for j in range(act_steps):

        a_tj = action_chunk[j]

        next_obs, r_tj, done, info = \
            low_level_env.step(a_tj)

        R_t += r_tj

        if done:
            break


    # ----------------------------------
    # 保存这一外层 environment step
    # ----------------------------------

    obs_buffer.append(s_t)

    # 保存整个 diffusion 去噪链
    diffusion_chain_buffer.append(chain)

    # 一个 action chunk 对应一个外层 reward
    reward_buffer.append(R_t)

    done_buffer.append(done)

    obs = next_obs


# ==================================================
# 2. Rollout完成后：计算 value / old log probability
# ==================================================

for env_t in range(n_steps):

    s_t = obs_buffer[env_t]

    V_t = critic(s_t)

    chain = diffusion_chain_buffer[env_t]

    # 对 diffusion chain 的每个 transition：
    #
    # x_K -> x_{K-1}
    # ...
    # x_1 -> x_0
    #
    # 计算旧策略 log probability

    old_logprob[env_t] = get_logprobs(
        s_t,
        chain
    )


# ------------------------------------------
# 3. GAE：在 Environment MDP 上计算
# ------------------------------------------

advantage = 0

for env_t in reversed(range(n_steps)):

    R_t = reward_buffer[env_t]

    if terminal:
        V_next = 0
    else:
        V_next = value_of_next_state

    delta = (
        R_t
        + gamma * V_next
        - V[env_t]
    )

    advantage = (
        delta
        + gamma * gae_lambda
        * advantage
    )

    A[env_t] = advantage

    returns[env_t] = (
        A[env_t] + V[env_t]
    )


# ==================================================
# 4. PPO update
# ==================================================

for update_epoch in range(num_epochs):

    # 每个训练样本实际上是：
    #
    # (environment step, diffusion step)
    #
    # 即：
    #
    # s_t,
    # x_k,
    # x_{k-1},
    # A_t

    for minibatch in batches:

        env_t, diffusion_k = sample_indices()

        s = obs_buffer[env_t]

        x_k = chain[env_t][diffusion_k]
        x_k_minus_1 = chain[env_t][diffusion_k + 1]

        # 当前策略重新计算：
        #
        # log p_theta(x_{k-1} | x_k, s)

        new_logprob = policy.logprob(
            x_k_minus_1,
            condition=(x_k, s)
        )

        old_lp = old_logprob[
            env_t,
            diffusion_k
        ]


        # ----------------------------------
        # Environment advantage
        # ----------------------------------

        advantage = A[env_t]


        # Diffusion MDP discount
        advantage *= (
            gamma_denoising
            ** diffusion_discount(diffusion_k)
        )


        # ----------------------------------
        # PPO likelihood ratio
        # ----------------------------------

        ratio = exp(
            new_logprob
            - old_lp
        )


        # clipped PPO objective

        loss1 = -advantage * ratio

        loss2 = -advantage * clip(
            ratio,
            1 - epsilon,
            1 + epsilon
        )

        policy_loss = max(
            loss1,
            loss2
        )


        # critic
        value_loss = (
            critic(s)
            - returns[env_t]
        ) ** 2


        total_loss = (
            policy_loss
            + value_coef * value_loss
            + bc_coef * bc_loss
        )

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
```

仓库真实更新逻辑也确实把总样本数定义为：

\[
N_{\text{steps}}
\times
N_{\text{env}}
\times
K_{\text{denoising}}
\]

然后把随机索引拆成：

```python
batch_inds_b, denoising_inds_b
```

也就是说，一个 PPO minibatch 样本对应的是“**某个环境决策 + 某个 diffusion denoising step**”。

还有一个值得注意的细节：DPPO 默认

\[
\text{reward\_horizon}=\text{act\_steps}
\]

并且在 loss 中：

```python
newlogprobs = newlogprobs[:, :reward_horizon, :]
oldlogprobs = oldlogprobs[:, :reward_horizon, :]
```

即虽然 diffusion 生成的是整个 `horizon_steps` 动作序列，PPO 梯度默认只统计**真正执行、因而真正影响当前 reward 的前 `act_steps` 个动作位置**。

因此可以把一次 DPPO rollout 最简洁地写成：

\[
\boxed{
s_t
\xrightarrow{\text{diffusion }K\text{ steps}}
a_{t:t+H}
\xrightarrow{\text{execute first }N_a}
R_t
\rightarrow s_{t+1}
}
\]

重复 \(n_{\text{steps}}\) 次以后：

\[
\boxed{
R_{0:T}
\rightarrow
GAE
\rightarrow
A_{0:T}
\rightarrow
\text{PPO update each diffusion transition}
}
\]

对于轨迹规划场景也应该特别注意这一点：如果模型生成 8 秒轨迹，但闭环中只实际跟踪其中 0.5 秒，那么 DPPO 原始语义对应的是“**0.5 秒执行产生的 reward → 更新此次生成过程**”，而不是先把完整 8 秒轨迹开环执行完再给奖励。
