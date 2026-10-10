# 研究导航

核心问题：如何利用闭环长程反馈降低规划器的能耗，同时避免安全、有效进度、速度和旅行效率出现不可接受退化？
研究分为 objective/credit 与 information/representation 两条主线，再按证据需要研究交互效应。

## 按问题读取

| 要回答的问题 | 入口 |
| --- | --- |
| 概念与解释边界 | [Semantics](semantics.md) |
| 规划、采样、正式 cadence、指标与 matched evaluation | [Planning/evaluation protocol](protocols/planning-and-evaluation.md) |
| 优化对象、reward、GAE/PPO 与训练统计 | [Training protocol](protocols/training.md) |
| Fixed-batch、guidance/执行干预与归因设计 | [Diagnostic studies protocol](protocols/diagnostic-studies.md) |
| ABI、执行、RNG/resume、artifact 软件保证 | [Data/model](../contracts/data-and-model.md)、[Execution](../contracts/execution.md)、[Training](../contracts/training.md)、[Artifacts](../contracts/artifacts.md) |
| 当前证据支持什么，哪些归因被修正 | [Findings](findings.md) → 所需原始 record |
| 为什么采用某项长期设计 | [ADR](../adr/) |
| 如何登记与检索真实运行 | [实验记录](../experiments/README.md) |
| 已接受的工作与验收 | 对应 GitHub Issue；当前 cadence-transfer / λ 研究见 [#83](https://github.com/xcz0/Eco-AutoDrive/issues/83) |

规范读取与冲突处理遵循 [AGENTS](../../AGENTS.md)：active Protocol/Contract 拥有规范要求，
当前具体实现由 observation/code/tests/config 回答；二者不一致时调查 conformance mismatch。

## 未解决研究问题

- [Reward/objective](hypotheses/reward-objective.md)：scalar reward 的结构、尺度、权重与长程 credit 是否能产生可接受 trade-off；何时需要多头 critic 或约束方法。
- [Information/representation](hypotheses/information-representation.md)：ego/reference、local scene、road/navigation preview 与 traffic 的独立价值；何时值得适配 encoder。
- [候选消融设计](hypotheses/ablation-plan.md)：先分离主效应，再研究 objective × information × representation；候选收益与非劣判据不作为默认规范。
- [PPO 工具与迁移](hypotheses/ppo-tooling.md)：跨条件稳定性、guidance 局部响应、运行规模及旧计划归属。

研究假设不等于已接受实施计划；机械有效、训练稳定、行为分离、节能和信息贡献也不是同一层结论。
具体结果与限制统一读 Findings，不在本入口维护阶段完成表或能力清单。

## 非规范性参考资料

- [PlannerRFT PPO 一手资料核查](plannerrft-ppo/plannerrft-ppo-primary-sources.md)
- [Diffusion policy / planner 长程 RL 综述](diffusion-policy-long-horizon-rl-survey.md)
- [RL energy-management 综述笔记](rl-ems-survey-notes.md)

参考资料保留原核查范围、公开/未公开信息与启发性建议；文献结论不等于本项目 Finding，
建议验证顺序不自动成为 Protocol 或 Skill。本次整理没有重新查证文献。
