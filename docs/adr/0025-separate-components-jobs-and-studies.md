# Separate reusable components, runnable jobs, and research studies

**Status:** Accepted and implemented; terminology and entrypoint layout amended by ADR 0030
**Date:** 2026-08-27

入口曾混合薄 CLI、benchmark 实现、研究实验和测试 helper，配置也混合 reusable groups、
jobs 与研究 manifests，导致接口所有权不清。因此选择 component/job/experiment 分层：
分别表达可复用组合、可执行研究条件和实验编排，物理目录不决定 resolved schema。

当时把 repository-only benchmarking/studies/analysis 放在 scripts 的决定已由
[ADR 0027](0027-use-internal-application-modules-and-resource-overlays.md) 取代；
[ADR 0030](0030-align-cli-and-experiment-config-layout.md) 修订术语和入口布局。
配置分层的理由保留，单实现不预建 selectable group 的取舍避免无实际需求的扩展机制。

内部入口选择直接切换而不保留旧 aliases；历史命令仍作为 provenance。
这些组织选择没有授权改变随机流、实验参数或训练／评测产物的研究含义。

规范归属：[Execution contract](../contracts/execution.md)。

> 本篇保存设计理由与历史决定；现行要求由上述 Protocol/Contract 拥有，读取路由见 [AGENTS](../../AGENTS.md)。
