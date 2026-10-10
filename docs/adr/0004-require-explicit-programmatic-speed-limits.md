# Require explicit programmatic speed limits

Treat a programmatic lane speed as an explicit experiment condition and replace only MetaDrive's exact unset sentinel while preserving legitimate lane limits. This avoids silently changing map semantics or letting an unset value enter the model as an extreme valid speed.

规范归属：[Data/model contract](../contracts/data-and-model.md)。

> 本篇保存设计理由与历史决定；现行要求由上述 Protocol/Contract 拥有，读取路由见 [AGENTS](../../AGENTS.md)。
