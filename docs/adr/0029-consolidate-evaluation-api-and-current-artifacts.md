# Consolidate the evaluation API and current artifact contract

**Status:** Accepted and implemented; physical layout superseded by ADR 0031
**Date:** 2026-09-02

[ADR 0021](0021-flatten-evaluation-package.md) 压平后，消费者仍依赖多个内部文件，
在线 recorder 与离线 schema/reader 共用含 Torch 的导入路径。Issue #78 的 engine、
metrics 与 experiment ownership 收口使剩余组织负担更明显。

当时选择统一在线／离线入口、分离 recorder 与轻量 typed I/O，并通过延迟导出隐藏内部布局，
降低调用方与执行依赖的耦合。旧内部路径直接移除，不保留转发。

同时选择只读取当前科研产物，不引入全局 artifact 版本及历史兼容层；completed metrics
只保留一个表示，并集中 I/O 结构验证。这些选择减少重复数据与解释分叉，
不取消 summary/trace 等科研语义核验。

[ADR 0031](0031-layer-evaluation-package.md) 仅取代物理布局；
artifact、依赖与验证的设计理由并未因此失效。具体字段和失败含义由 artifacts contract 拥有。

规范归属：[Execution contract](../contracts/execution.md)、[Artifacts contract](../contracts/artifacts.md)。

> 本篇保存设计理由与历史决定；现行要求由上述 Protocol/Contract 拥有，读取路由见 [AGENTS](../../AGENTS.md)。
