# Separate online proxy and offline FASTSim energy metrics

**Status:** Accepted and implemented
**Date:** 2026-09-03

运动学执行需要便宜的在线能耗信号，而 FASTSim 是依赖完整速度／时间序列和连续动力系统
状态的后向车辆模型。若逐 0.1 s 重置它，会把一个连续行程变成反复初始化的片段。

因此选择保留实际执行 distance/speed 驱动的在线 fuel proxy，并把 FASTSim 放在完整 trace
的离线比较边界。两者共享 objective-neutral 输入／结果抽象，但保留不同 metric 名称与单位，
不假设 proxy mL 与 fuel energy J/Wh 是同一个研究量。

首个 FASTSim adapter 选择 bundled conventional Ford Fusion，环境条件显式配置；
具体库 API 和版本由代码／lock 拥有，不在 ADR 维护调用镜像。
完整 trace 还能解释静止时的 idle/auxiliary energy，而零距离强度仍不可定义。

选择让缺失轨迹、距离不一致和 provider 失败可见，而不 fallback，是为保护比较含义。
将 FASTSim 接入在线 reward 或默认 evaluation artifact 需要新的研究决定；
本篇没有作出这一扩展。

规范归属：[Semantics](../research/semantics.md)、[Planning/evaluation protocol](../research/protocols/planning-and-evaluation.md)、[Execution contract](../contracts/execution.md)、[Artifacts contract](../contracts/artifacts.md)。

> #102 迁移阶段：上述新规范仍为 proposed，生效入口遵循 [AGENTS](../../AGENTS.md)。
> 本篇保存设计理由与历史决定，不作为第二套现行规范；本次收口不激活新 owner。
