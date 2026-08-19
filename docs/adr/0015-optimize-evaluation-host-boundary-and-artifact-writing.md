# Optimize the evaluation host boundary and artifact writing

**Status:** Accepted and implemented
**Date:** 2026-08-14

长程闭环 evaluation 的主要性能目标是降低整个独立评测进程的墙钟时间和峰值内存，而不是单独优化模型 forward。

此前一个 planning cycle 中存在多次重复的 device-to-host copy 和同步：模型输入、prediction、ego execution 数据以及 trace 数据可能沿不同路径分别复制。Sampler 热循环中还执行同步式数值检查，而 recorder 使用 Python record 列表累计数据，最终再整体 stack 或 concatenate。这些操作会增加 CUDA synchronization、Python 分配以及 episode 结束时的峰值内存。

因此，evaluation 明确设置一个 host boundary。

原始 observation 在进入模型前已经是 CPU 数据，因此保留 CPU representation 作为持久化来源；设备副本只服务于模型计算。模型推理完成后，需要写入 trace 或交给环境执行的结果统一返回 host，尽量合并 device-to-host transfer 和 synchronization，而不是为不同消费者分别复制。

完整数值有限性检查放在最终 CPU boundary。Sampler 的 transition 热路径只执行保证数值更新安全所必需的结构检查，避免每个 diffusion transition 都触发设备同步。

Episode trace 使用预分配数组，而不是累积 Python record 后再整体拼接。Recorder 在 episode 开始时根据最大容量分配存储，并在运行期间直接写入对应槽位；episode 结束时仅返回实际有效区域的 view 或 slice。

这一设计具有三个目的：

1. 将内存复杂度从大量临时 Python/NumPy 对象转换为可预测的连续数组；
2. 避免 finalize 阶段的大规模 stack/concatenate；
3. 使 partial、complete 和 empty episode 使用同一记录机制。

Trace 持久化继续使用 NumPy NPZ。写入路径采用标准未压缩 `np.savez`，因为长程 evaluation 更关注写入 CPU 时间和峰值内存，而当前数据不需要通过压缩格式换取额外复杂度。离线 reader 在读取外部数据时仍执行完整结构和数值验证。

具体 host result 类型、trace 字段、容量、dtype、mixed-precision 设置以及硬件相关优化参数属于当前系统契约和 evaluation 配置，而不是本 ADR 的长期约束。
