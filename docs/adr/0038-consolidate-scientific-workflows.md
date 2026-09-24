# 0038 — 按当前科研工作流整合实验架构

- 状态：接受
- 日期：2026-09-14
- 来源：Issue #97
- 部分取代：ADR 0034、0035、0036、0037

> 关于人工 intervention/rollout 0.1 s 与普通 evaluation 0.5 s 边界保持不变的决定，
> 已由 [ADR 0039](0039-unify-closed-loop-cadence.md) 取代；其他决定按下述范围保留。

## 背景

历史 study 把采集、reward 校准、backward、人工干预和实验设计放在同层，
又通过 reference-dir 和冻结数值串联，导致底层机制依赖研究命名，新 batch 不能独立诊断。

## 决定

选择让 experiments 只编排研究 protocol，共享机制回到拥有它的稳定下层，
配置组合保持轻量，不建立通用调度框架。Fixed-batch、reward 变换、优化诊断与人工干预
各有独立所有者；analysis 只消费持久化测量，避免通用机制反向依赖具体实验。

当时将校准与 energy-band 改为按源 batch 和显式配置重算并保存实际尺度，
以独立数值测试替代历史 reference-dir／expected-value 守卫。
这描述该次选择；后续显式 frozen-band 设计与 batch-derived 的区别由 diagnostic protocol
拥有，不能将本篇解释成所有未来诊断都必须重估 band。

训练诊断保留预算、预声明阈值及确定候选规则，测量先保存，报告不重裁 gate。
CLI 收口为薄入口并删除旧 studies、stage A/B/C、top-N 晋升、pruning、stability 数据库分析
及独立 reproducibility 目录协议，没有保留兼容层。具体 current topology 不在 ADR 维护。

## 保留的约束与取代范围

仅取代 ADR 0034–0037 的通用机制位置、旧入口／历史工作流保留、强制参考链和冻结来源检查。
保留轻量离线分析、原始证据、source/output 分离、typed summaries、配对／随机性、
逐 seed bootstrap、undefined 与解释边界；规范要求收口到对应 Protocol/Contract。

当时没有改变 planner、PPO、reward 与仿真含义；原文“0.1 s rollout / 0.5 s evaluation 不变”
只代表当时范围，已由 ADR 0039 局部取代。历史记录仍保留原命令、路径和结果，
组织调整及软件验收不代表历史结果已重现或节能结论成立。

规范归属：[Execution contract](../contracts/execution.md)、[Artifacts contract](../contracts/artifacts.md)、[Diagnostic protocol](../research/protocols/diagnostic-studies.md)、[Planning/evaluation protocol](../research/protocols/planning-and-evaluation.md)。

> #102 迁移阶段：上述新规范仍为 proposed，生效入口遵循 [AGENTS](../../AGENTS.md)。
> 本篇保存设计理由与历史决定，不作为第二套现行规范；本次收口不激活新 owner。
