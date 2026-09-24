# Optimize the evaluation host boundary and artifact writing

**Status:** Accepted and implemented
**Date:** 2026-08-14

长程评测的目标是降低整个进程墙钟时间和峰值内存。当时同周期数据沿多条路径重复传回 CPU，
sampler 数值检查频繁同步，trace 在结束时集中 stack/concatenate，带来了同步与临时分配成本。

选择保留原始 CPU observation 作为持久化来源，并集中推理结果的 host boundary，
使执行与审计不各自复制完整数据。把完整数值校验移到最终 CPU boundary，是为了减少
热循环同步而保持失败可见，不是放弃有限性检查。

当时采用预分配 trace 与未压缩 NPZ，以可预测的连续存储换取较低写盘 CPU 成本和峰值内存，
并让 complete、partial、empty 使用同一记录机制。这些是当时优化方案的理由；
具体容量、布局和传输调度由代码／配置拥有，不作为持续维护的 ADR 实现清单。
离线 reader 仍承担外部输入验证。

规范归属：[Execution contract](../contracts/execution.md)、[Artifacts contract](../contracts/artifacts.md)。

> 本篇保存设计理由与历史决定；现行要求由上述 Protocol/Contract 拥有，读取路由见 [AGENTS](../../AGENTS.md)。
