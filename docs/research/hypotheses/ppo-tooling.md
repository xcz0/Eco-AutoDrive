# PPO 工具与迁移假设

本篇合并旧 PPO design gates / support plan 中尚未解决的工具问题；它不是当前能力清单，
也不自动建立实施任务。Reward、credit 与多头 critic 只在 [reward-objective](reward-objective.md)
维护；信息消融、encoder 提取与微调只在 [information-representation](information-representation.md)
维护。总体候选顺序见 [ablation-plan](ablation-plan.md)。

## 稳定性与学习方向能否迁移

历史候选的机械稳定性、objective 可辨识性和 learned behavior 必须分别理解，
已有证据见 [Findings](../findings.md)。改变 reward scale/structure、observation、encoder
trainability、critic、traffic 或训练 horizon 后，旧结论可迁移到什么范围仍是研究问题。

候选研究可先检查 finite diagnostics、Beta 边界、闭环行为与 held-out retention，
再判断是否需要独立 optimizer 诊断；不默认重做完整超参搜索，也不把旧 support plan E
升级成所有任务的强制复验清单。具体预算与判据在接受任务时确定。

已接受的 canonical-cadence transfer 由 [Issue #83](https://github.com/xcz0/Eco-AutoDrive/issues/83)
跟踪，状态与缺少原始 E-048 记录的核验边界见
[Findings](../findings.md#canonical-cadence-transfer-的证据边界)。该执行工作不重新包装为未接受假设。

## Guidance 的局部响应能否外推

旧 G-P3 不能继续笼统写作“完全未知”：
[prefix 与 learned-policy bridge 证据](../findings.md#guidance-与-execution-prefix-如何共同决定方向)
已解释部分条件下的方向错位。剩余问题包括 state-dependent 异质性、常量干预的 SC off-route、
更丰富场景下的 action sensitivity，以及新 cadence 训练后的 objective→behavior 方向。
这些问题不预设改变 guidance range/action definition；必要干预应在对应实验中显式声明。

## 何时需要扩大运行规模

只有研究信号值得扩大样本量、且统计预算或 simulator throughput 构成实际瓶颈时，才考虑
更大 batch、更长训练或额外 orchestration。旧 support plan F / G-P6 提及的 vector/process
能力不是一份尚未实现清单；当前机制归 code/config，新增能力需先确认真实缺口。

## 旧计划与历史决定的归属

旧 Task 1A 方案包含“新增 frozen band / substep 审计 / replay_id 修复 / E-048 配置”的实施步骤。
[#102 固定 revision 盘点](https://github.com/xcz0/Eco-AutoDrive/issues/102#issuecomment-5806230219)
已定位这些机制在 `eaafe5b` 存在；本次只按文档整理，不重新审计源码或重复实施。
旧“已规划、未执行”不能同时代表机制状态和实验状态。未完成工作的唯一执行入口为 #83；
独有的旧方案细节仍可读取
[固定 revision 的 Task 1A 计划](https://github.com/xcz0/Eco-AutoDrive/blob/eaafe5bf911390ef3263bea808aadc40215068a6/docs/research/issue83-task-1a-identifiability-transfer-plan.md)，
不保留第二份 active checklist。已报告运行的核验边界由 Findings 单独记录。

已关闭的历史 design gates 不重建状态表：采样理由见 [ADR 0011](../../adr/0011-add-explicit-five-step-ddim-sampler.md)，
reference/guidance 见 [ADR 0013](../../adr/0013-add-reference-centered-orthogonal-guidance.md)，
policy/action 见 [ADR 0016](../../adr/0016-add-forward-only-exploration-policy.md)，
PPO 数学选择见 [ADR 0018](../../adr/0018-use-torchrl-for-gae-and-ppo-math.md)。
旧 G-01 的 [ADR 0017](../../adr/0017-add-10hz-rollout-contract.md) cadence 已由
[ADR 0039](../../adr/0039-unify-closed-loop-cadence.md) 取代，不能借“Closed”复活旧训练步长。
方法与软件保证分别引用 [training protocol 草案](../protocols/training.md) 和
[training contract 草案](../../contracts/training.md)；两者在全局入口切换前仍为 proposed。
