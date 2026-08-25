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
traffic audit、failure 和 timing 的领域契约。Evaluation 继续拥有动态 scenario refill；rollout 继续
拥有逻辑 waves、独立 RNG、episode、GAE、bootstrap 和 artifact 语义。本决定只取代 ADR 0022 中
关于自定义 IPC client/worker protocol 的部分，不改变 `MetaDriveEnvSlot` 的所有权决定。

共享 buffer 路径只承载固定 shape/dtype tensor。固定依赖 TensorDict 0.13.0 在 partial mask 与
`NonTensor` output 同时出现时会在内部调用不存在的 `TensorDict.maybe_to_stack`；因此当前 domain
dataclass 和可捕获 failure 在同一 façade 操作锁内通过 TorchRL 自带 remote-method channel 读取。
这不是项目 Pipe、额外 worker 进程或自定义消息协议。若未来依赖版本消除此限制，是否改回同一
TensorDict 返回属于独立实现选择，不改变本 ADR 的进程与调度所有权。

可捕获的 worker Python 异常保存 operation 与真实 traceback，façade 标注 slot、关闭整个 pool 并
抛出 `VectorMetaDriveWorkerError`。硬进程退出只包装 TorchRL 原始异常和当前 operation。B=1 仍使用
独立进程；关闭幂等。`VectorEnvTiming` 只保留 worker environment/observation service time，parent
从 batch wall 与最慢 worker busy 计算 `transport_sync_s`，不再暴露自定义 IPC send/receive/wait
字段。

本次迁移不采用 TorchRL Collector，不改变 planner batching、explicit-generator Beta、PPO、RNG、
reward、trajectory execution、evaluation artifact 或训练 TensorDict schema。旧自定义 IPC 文件和
独立 Collector PoC 在切换后删除。
