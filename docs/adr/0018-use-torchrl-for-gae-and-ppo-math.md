# Use TorchRL for GAE and PPO mathematics

选择 TorchRL/TensorDict 的 GAE 与 clipped PPO，避免项目维护第二套通用数学实现。
项目仍拥有 scenario/episode 的顺序、递归边界与 tail value；成熟库不能代替这些领域保证。
Actor/critic 共享一个 policy 参数所有者，避免独立网络副本使 collection 与更新失配。

PPO 优化的是 Exploration Policy 的 guidance 概率，不是 Diffusion/DDIM 转移概率。
当时采用完整 batch 的一次 advantage 标准化、unclipped L2 value objective，
并保持 collection/update 网络模式确定，以免 dropout 污染 old/new ratio。
退化 batch 显式失败的选择保留了问题可见性。公式与软件记账分别归 training protocol/contract。

**历史文字澄清（#102）：** 原文将 Beta 基础动作与 guidance 写成 `[0,1]`、`[-1,1]`，
与 [ADR 0016](0016-add-forward-only-exploration-policy.md) 拒绝端点的决定冲突。
这里记录该表述问题，不将其解释为允许端点或放宽动作域；有效域统一见 training contract。
固定人工干预的闭区间接口不属于 Beta 概率空间。

Policy export 与 resumable checkpoint 的区别仍保留；优化参数、库版本、reward 与运行规模
由各自配置及规范拥有，不从本篇推断当前实现。

规范归属：[Training protocol](../research/protocols/training.md)、[Training contract](../contracts/training.md)、[Artifacts contract](../contracts/artifacts.md)。

> 本篇保存设计理由与历史决定；现行要求由上述 Protocol/Contract 拥有，读取路由见 [AGENTS](../../AGENTS.md)。
