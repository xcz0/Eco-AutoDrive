# Restructure evaluation artifacts and analysis

**Status:** Superseded by ADR 0021
**Date:** 2026-08-14

Evaluation 的在线执行、持久化和离线分析曾位于同一平面包，容易让离线分析意外依赖仿真
或模型 runtime。已验证数据在分析端退化为 dict 后再次解释，也造成重复校验和含义分叉。

当时选择 runtime、artifacts、analysis 三个方向：在线与离线都消费 typed artifact boundary，
而 artifact 不反向依赖执行或分析。共享 trace 字段声明服务 recording 与 validation，
同时保留项目自己的跨数组关系检查，避免引入泛化 schema 框架。

物理布局已由 [ADR 0021](0021-flatten-evaluation-package.md) 取代；
该 ADR 明确保留依赖方向。这里保留分层原因与替代关系，不维护当前模块地图。
字段、reader 验证与轻量导入保证由 artifacts contract 拥有。

规范归属：[Execution contract](../contracts/execution.md)、[Artifacts contract](../contracts/artifacts.md)。

> #102 迁移阶段：上述新规范仍为 proposed，生效入口遵循 [AGENTS](../../AGENTS.md)。
> 本篇保存设计理由与历史决定，不作为第二套现行规范；本次收口不激活新 owner。
