# Use TorchRL ParallelEnv for vector MetaDrive execution

**Status:** Accepted and implemented
**Date:** 2026-08-25

Evaluation 与 policy rollout 都需要在 Windows `spawn` 进程中运行多个固定
`MetaDriveEnvSlot`，支持 full/partial reset-step、共享 observation/action tensor、worker failure
以及确定的关闭生命周期。项目原有实现自行维护 `Process`、`Pipe`、连接轮询、消息协议、共享内存
和 terminate/join；这些机制与 TorchRL 已提供的环境编排重复，并扩大了 slot 领域语义之外的维护面。

因此，生产 `VectorMetaDriveEnv` 使用固定版本 TorchRL 0.13.3 的 `ParallelEnv`，配置为 CPU、
`spawn`、`shared_memory=true`、`use_buffers=true`、`serial_for_single=false`。TorchRL 拥有进程、
共享 TensorDict buffer、partial `_reset`/`_step` mask、健康检查和 close/join/terminate。项目 façade
只拥有完整 scenario catalog、scenario index 与物理 slot 映射、请求顺序恢复，以及 reset、execution、
traffic audit、failure 和 timing 的领域契约。它以单一 `reset(..., slots=...)` / `step(..., slots=...)`
入口返回按请求顺序组织的批 TensorDict；planner observation 位于嵌套 `observation`，不再拆成逐 slot
结果对象。Evaluation 继续拥有动态 scenario refill；rollout 继续
拥有逻辑 waves、独立 RNG、episode、GAE、bootstrap 和 artifact 语义。本决定只取代 ADR 0022 中
关于自定义 IPC client/worker protocol 的部分，不改变 `MetaDriveEnvSlot` 的所有权决定。

共享 buffer 路径只承载固定 shape/dtype tensor。固定依赖 TensorDict 0.13.0 在 partial mask 与
`NonTensor` output 同时出现时会在内部调用不存在的 `TensorDict.maybe_to_stack`；因此当前 domain
dataclass 和可捕获 failure 在同一 façade 操作锁内通过 TorchRL 自带 remote-method channel 读取；
façade 取得请求 slot 的 tensor batch 后才把这些 sidecar 作为 `operation_results` NonTensor 附加到
父进程 TensorDict，因此 NonTensor 不进入 shared buffer 或 partial-mask 实现。这不是项目 Pipe、
额外 worker 进程或自定义消息协议。

可捕获的 worker Python 异常保存 operation 与真实 traceback，façade 标注 slot、关闭整个 pool 并
抛出 `VectorMetaDriveWorkerError`。硬进程退出只包装 TorchRL 原始异常和当前 operation。B=1 仍使用
独立进程；关闭幂等。`VectorEnvTiming` 只保留 worker environment/observation service time，parent
从 batch wall 与最慢 worker busy 计算 `transport_sync_s`，不再暴露自定义 IPC send/receive/wait
字段。

TorchRL Collector 可以拥有常规 reset/done lifecycle、固定 frame batch 和 tensor transition storage，
但不能在每个 simulator step 后取得 remote-method channel 上的 execution/traffic/reward audit；批次
结束时只剩最后一次 operation result。它也不能自然表达逻辑 scenario 多于物理 worker 时的精确
per-scenario transition 配额、逐 slot diffusion/policy generator、延迟 audit transfer 和单独 bootstrap
profiling。把这些职责编码回 policy output 或固定 tensor spec 会扩大 MetaDrive/Diffusion-Planner 特有
数据面，因此 rollout 保留项目 collector，但 serial/vector 共用同一 episode lifecycle helper；项目
collector 不重新实现进程、共享内存、partial mask、GAE 或 PPO。
