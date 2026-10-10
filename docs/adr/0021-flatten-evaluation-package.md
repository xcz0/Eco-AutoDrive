# Flatten the evaluation package while preserving dependency direction

**Status:** Superseded by ADR 0029
**Date:** 2026-08-25

[ADR 0020](0020-restructure-evaluation-artifacts-and-analysis.md) 的分层在每层文件很少时，
产生了超过实际职责需要的目录、空初始化文件和转发成本。因此当时选择压平为职责模块，
直接切换内部导入，不保留兼容 forwarder。

压平是物理组织选择，没有取消 artifact 作为轻量 typed 边界、analysis 只消费产物、
在线路径单向依赖 artifact，以及 rendering 延迟导入的理由。
配置、随机流、仿真与 artifact 语义不在该次组织变更范围内。

本篇布局已由 [ADR 0029](0029-consolidate-evaluation-api-and-current-artifacts.md) 取代。
历史文件列表不再作为 current topology 镜像；持续软件保证统一引用 contracts。

规范归属：[Execution contract](../contracts/execution.md)、[Artifacts contract](../contracts/artifacts.md)。

> 本篇保存设计理由与历史决定；现行要求由上述 Protocol/Contract 拥有，读取路由见 [AGENTS](../../AGENTS.md)。
