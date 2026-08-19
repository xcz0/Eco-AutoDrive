# Use TorchRL for GAE and PPO mathematics

Exploration Policy 的 PPO 更新使用 TorchRL/TensorDict 提供的 GAE 和 clipped PPO 数学实现，而不是在项目中维护另一套自定义 PPO 公式。

每个 `RolloutEpisode` 独立执行 GAE。Episode 最后一个 transition 是递归边界：

- 真实 `terminated` transition 不进行 bootstrap；
- truncation 或 rollout-limit tail 使用显式保存的 tail value bootstrap；
- 不同 episode 只有在各自 GAE 完成后才能 flatten 或拼接。

因此，advantage 不得跨 collector 或 episode boundary 泄漏。

Exploration Policy 是 actor 和 critic 的单一参数所有者。TensorDict actor/critic adapter 共享同一个 policy module，而不是维护相互独立的网络副本。

Actor 的基础随机变量位于 Beta distribution 的 `[0, 1]` 区间，并通过

`g = 2u - 1`

映射到 guidance action 的 `[-1, 1]` 区间。PPO 使用变换后 action space 中的 joint log-probability，因此 log-probability 和 entropy 都必须包含 affine transform 的 Jacobian。

PPO ratio 使用 rollout 时保存的 transformed old log-probability 与当前 policy 重新计算的 transformed log-probability：

`ratio = exp(new_log_prob - old_log_prob)`

Diffusion/DDIM transition probability 不属于这个 PPO ratio。扩散规划器在这里提供被冻结的规划与 policy context，而 PPO 优化的随机策略是 Exploration Policy。

GAE 产生的 advantage 在所有 episode 完成 GAE 并组成当前 PPO batch 后统一标准化一次。标准化使用 sample standard deviation。样本数不足、方差为零或出现非有限统计时直接失败，而不是通过 clamp 隐藏退化 batch。

Value objective 使用 unclipped L2。Policy loss、value loss 和 entropy term 的梯度共同更新 actor head、value head 以及共享 trunk。

Collection 和 PPO optimization 均保持 policy 的确定性网络模式，使 dropout 等训练模式随机性不会污染 old/new probability ratio；这不妨碍优化阶段记录梯度。

TorchRL/TensorDict 的具体版本、optimizer 参数、scheduler horizon、minibatch 大小和 update 数量属于训练配置和当前系统契约，而不是本 ADR 的架构不变量。

Policy export checkpoint 与 resumable training-state checkpoint 是两个不同边界。前者用于保存可加载的 Exploration Policy 参数；后者用于恢复一次训练运行，还必须包含 optimizer、scheduler、训练随机状态以及训练循环进度等状态。不能用 policy export 的格式约束推断训练是否可恢复。

Reward 定义、MetaDrive rollout 配置、并行采样规模和具体 smoke profile 由其他 ADR 或实验配置决定。
