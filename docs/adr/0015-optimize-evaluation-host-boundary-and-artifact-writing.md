# Optimize the evaluation host boundary and Artifact v3 writing

**Status:** Accepted and implemented
**Date:** 2026-08-14

长程 traffic 评测的首要目标是独立进程总墙钟。此前每个规划周期保留设备 observation，随后为
trace 将其复制回 CPU；prediction 又分别为 ego 执行和 trace 做主机复制。采样器还在每个 DPM/DDIM
transition 上执行同步式有限性归约，recorder 以 Python record 列表累计并在 `finalize()` 中整体
`stack`/`concatenate`，最后用压缩 NPZ 消耗 CPU 时间和额外峰值内存。

评测边界现在保留 adapter 产生的原始 CPU `float32` observation 作为 Artifact 输入；Fabric 设备
副本只供模型使用，并允许 autocast 产生 FP16/BF16 计算张量。`FabricInferenceRuntime.infer()` 返回
CPU-resident `HostInferenceResult`：noise、prediction、reference 和 guidance diagnostics 在同一 CUDA
stream 上排队复制并只同步一次，随后统一转为 Artifact v3 dtype 和检查有限性。ego 轨迹是 prediction
的视图，不再单独复制。最终 CPU 边界负责拒绝非有限推理结果；采样 transition 热循环只验证必要的
shape、device 和 dtype 结构。

`EpisodeTraceRecorder` 在创建时接收 planning 与 warmup 最大容量，按 Artifact v3 shape/dtype
预分配并直接写槽位。`finalize()` 只返回有效切片，并继续区分 `complete`、`partial` 和 `empty`。
producer 做增量边界检查和最终结构检查；`load_trace_artifact` 仍对外部/落盘数组执行完整字段、shape、
精确 dtype、有限性和跨数组关系校验。

Artifact schema 保持 v3，文件名、JSON 字段、NPZ key、shape 和 dtype 均不改变。`trace.npz` 使用标准
未压缩 `np.savez`；现有 NumPy reader 无需格式迁移。evaluation 根包不再重导出会装载 runtime 的
公共符号，summary、episode writer、runtime metadata 与 reader 分离；离线 reader 不导入
Lightning、MetaDrive 或模型 runtime。

Windows RTX 3050 的两个 300-step traffic 场景筛选中，CUDA
`torch.set_float32_matmul_precision("high")` 的进程总墙钟中位数相对 `highest` 降低 5.56%，且下降量
超过两组 MAD；两组均保持相同场景状态、终止原因、planning cycles 和 simulator steps，因此 CUDA
评测路径采用 `high`。该设置允许浮点差异，不构成跨设备逐位一致性承诺。
