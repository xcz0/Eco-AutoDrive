# 0039 — 统一 closed-loop cadence 与 transition 时间语义

- 状态：接受
- 日期：2026-09-21
- 来源：Issue #96 Task E
- 部分取代：ADR 0017 的分离 cadence 决定；ADR 0038 中“人工 intervention/rollout 的 0.1 s 与普通 evaluation 的 0.5 s 边界保持不变”的表述

## 背景

[ADR 0017](0017-add-10hz-rollout-contract.md) 曾让 PPO 每次执行一个 0.1 s 点，
而 evaluation 每周期执行五点、0.5 s。分离 cadence 让 training 与 evaluation 实际运行
不同 closed-loop system，也把 gamma=0.99 隐含解释为 per 0.1 s transition。
重复 execution-mode plumbing 还让这一差异散落到多个边界。

## 决定

选择统一正式 training/evaluation 为五个子步、0.5 s 的 decision cadence，
让一个 PPO transition 对应一次 decision 及其实际 execution prefix。
这取代 ADR 0017 的分离 cadence，以及
[ADR 0038](0038-consolidate-scientific-workflows.md) 中保留旧时间边界的表述，
不取代它们其余 DDIM、buffer、RNG、bootstrap 或证据决定。

删除两套 execution mode 与重复校验，保留显式 matched diagnostic prefix 干预，
是为同时表达正式方法与因果诊断，而不让诊断覆盖静默变成 baseline。
具体 cadence、终止短前缀和配置保证由 Protocol/Contract 拥有。

多子步 reward 选择逐子步求值后归约，在线与离线重评分共用同一数学，
并保存足够的子步审计输入。理由是 min gate 与 base sum 的乘积一般不能恢复实际
gated reward，聚合后一次缩放也不能代替非线性子步重评分。
归约表、字段与有效前缀只在 training protocol/contract 定义，不在 ADR 维护第二份规则。

Gamma 保持 per-transition，原配置 0.99 不做 rebase：物理折扣视界由约 10 s 变为约 50 s。
这是有意接受的科学语义变更，不能在文档整理中“补偿”回旧物理时间尺度。

## 后果与可比性

Evaluation 原有五子步数值行为不因这一 cadence 统一改变；但旧训练从 0.1 s 改成 0.5 s 后，
同模拟时长的 transition 数约为原先 1/5，每 transition reward 尺度及折扣视界约为原先 5 倍。
旧 10 Hz training 与此后的训练产物不能直接比较；这些尺度是后果解释，不是实验收益结论。

统一 cadence 没有合并 train/eval 的 seed、RNG lifecycle、episode 编排、schema 或索引单位。
历史 records 保持原样，不回填，也不声称旧 E-039/E-040/E-047 已按新 cadence 重新验证。
[#83](https://github.com/xcz0/Eco-AutoDrive/issues/83) 的 transfer validation
仍须独立证据，不能由本次文档收口宣称通过。

规范归属：[Planning/evaluation protocol](../research/protocols/planning-and-evaluation.md)、[Training protocol](../research/protocols/training.md)、[Execution contract](../contracts/execution.md)、[Training contract](../contracts/training.md)。

> #102 迁移阶段：上述新规范仍为 proposed，生效入口遵循 [AGENTS](../../AGENTS.md)。
> 本篇保存设计理由与历史决定，不作为第二套现行规范；本次收口不激活新 owner。
