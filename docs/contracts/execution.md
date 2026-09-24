---
kind: contract
status: proposed
scope: [execution, configuration, resources, rng, ownership]
read_when:
  - changing execution timing simulator slots or resource boundaries
  - changing diffusion random streams or runtime ownership
---

# Execution contract

这是 [Issue #102](https://github.com/xcz0/Eco-AutoDrive/issues/102) 的 proposed 草案，尚不替换旧入口。
MUST／MUST NOT 表示拟保留的软件保证。正式 cadence、sampling 与 no-repair 方法的唯一草案定义在
[planning/evaluation protocol](../research/protocols/planning-and-evaluation.md)；数据 ABI 在
[data/model](data-and-model.md)，训练特有 RNG 在 [training](training.md)。

## 时间、实际状态与客观事实

MUST 验证 physics step `0.02 s × decision_repeat 5` 等于 Protocol 的 simulator 子步。
执行前缀默认采用 canonical cadence；只有明确的 matched diagnostic intervention 才能覆写长度。
正式 training config 必须显式保存 simulator_step_s、closed_loop_execution_steps、decision_interval_s，
并在 typed boundary 拒绝非 canonical 值。共享 ABI 不在各模块另定义一套常量。

有限 float32 ego 局部轨迹转换到世界坐标后，每个实际子步写入 vehicle center、heading、相邻 center 有限差分的 velocity 与最短 heading 差的 angular velocity。
下一观测／规划锚点 MUST 来自最后实际执行状态。Terminal/truncation 时立即停止 prefix，保留真实执行长度和两个原始 flags；不得以计划长度伪造未执行状态。
Route edge 集同时定义 out-of-road 边界；terminal route padding 条件引用 data/model contract。

Simulator state、raw observation、prediction 与 energy 必须分离。
执行边界 MUST 唯一推导客观 motion、behavior、energy 与误差事实（TransitionMetrics）；
reward、training、evaluation 不得另算同名 speed/stopped/wrong-direction/collision。
研究定义见 planning/evaluation 指标表。
Domain metrics 不携带 reward policy，training 按其 Protocol 求 score，evaluation 只记录客观事实。

## Slot、交通与进程生命周期

Reset MUST 原子完成场景 reset、history warmup 与首 observation；step 原子完成实际 prefix、traffic history 更新和下一 observation。Serial/vector 共享这些边界。
换图时环境、history、observation 与 map cache 一起重建，不能残留上个场景状态。

交通首次推理前固定 ego，推进背景交通直到满足 data/model history ABI。
Warmup 状态、reward、termination、对象数量与正式窗口分开；ego 位移达到 `1e-3 m` 或预热提前结束立即失败。No-traffic 的真实空场景约束由 data/model 定义。
既有 reset 对 navigation 损坏的恢复仅限专用 local-route exception：重建环境后重试一次，不能扩展为一般异常 retry、旧 observation fallback 或跳过坏场景。

Partial reset/step MUST 只推进指定物理 slots，按请求 slot 顺序返回；未选 slot 的状态和 RNG 不变。
逻辑 scenario 身份不随物理 worker 复用而丢失；完整训练 batch 的保证另见 training contract。
Worker 只拥有 CPU 环境，不加载 planner/CUDA；B=1 vector 也使用独立 worker。
第三方环境接口要求的零 reward 占位不等于科研 reward，训练不得消费它。

Worker 操作失败 MUST 带回 slot、operation 和可用的原始异常／traceback，关闭整个 pool 后传播；
硬退出保留第三方原始错误，不伪造远端 traceback。Close 幂等，进程关闭／join／terminate 由成熟并行机制管理。
跨边界副本不能继续别名引用随后被 worker 改写的共享 buffer。

同 job vector 与 job-parallel 不得嵌套，二者关闭 video。Job-parallel 中每进程独立持有环境、单设备推理 runtime 和 writer；CPU worker×thread 预算必须验证。
既有 job-level CUDA 条件仅允许共享一张可见 GPU，要求显式 deterministic 与正式运行前显存 preflight；这是该专用执行模式的条件，不新增普通任务通用 preflight。
Serial/vector/job-level 的显式 deterministic 设置 MUST 一致作用于相应后端。

## 配置、资源与依赖方向

Hydra/OmegaConf 只在 composition 边界存在，领域执行接收严格 typed config。
第三方开放 env 映射可保留，但项目消费的 horizon、traffic、evaluation 字段必须交叉验证。
完整 semantic job 是场景、sampler、reward 等的唯一声明；experiment 选择 job 并声明配对、干预、搜索或显式 overrides，不复制整份 job 再另行对账。

Resources 只提供 worker／slot／thread 容量，不能决定科学参数或 execution topology；换机器不得静默改变 PPO、reward、sampler、guidance 或场景定义。Semantic job 可以无机器 profile 地 compose/validate，执行边界需要预算时缺少 profile 则失败，不合成默认预算。
显式 resources override 优先；进程环境不被 `.env` 覆盖；自动机器选择仅在无显式 override 时使用。

稳定 application logic 属于 `src/eco_planner`，scripts 仅做参数、bootstrap、展示与退出码。
通用机制归最低稳定层，共享配置不由 evaluation 私有拥有；底层及 analysis 不依赖具体 experiments。
Reward 数学归 reward 层，collector 消费 domain facts，环境／worker 不执行 reward profile。
业务实现不得从只读 `ref/` 导入。CLI 或 offline reader 的轻量依赖保证见 artifacts contract。

配置、文件及第三方返回值首次进入 typed domain 时校验／转换一次。内部受控数据流依赖类型与 producer tests，不重复做同层防御检查；有限性、冻结参数、RNG 等科学语义检查仍必须保留。

## Diffusion RNG、推理设备与传输

每个逻辑 slot MUST 有独立持久 diffusion generator；每周期先抽标准正态 initial noise，
随机 DDIM transitions 再按顺序从该 slot 流抽样。Batch composition 不得改变其他 slot 消费顺序。
Reference/guided pass 使用同一批已取得的 transition draws，而非各自再抽一次。
Stochasticity=0 MUST 不消费 transition RNG；非零值要求显式同设备 generator。
Evaluation 每 episode generator 使用该 job runtime seed；训练跨 episodes 的持续状态由 training contract 定义。
Runtime seed 同时进入全局运行时 seeding，不能据此合并 map/action/minibatch 的独立流。

Sampler MUST 使用项目 VP-SDE 离散化的 trained betas，不采用第三方默认 schedule；模型时间由实际 sigma 恢复，不能把 scheduler 离散索引直接当连续时间。后端声明 `implementation=diffusers`，
项目不维护另一套 solver 更新公式。研究 schedule 仍由 Protocol 拥有。

推理由单进程、单设备 runtime 统一拥有模型设备、观测传输与 forward precision。
自动设备只解析 CPU/CUDA；precision auto 在 CPU 为 32-true，CUDA 优先 bf16-mixed，不支持 BF16 时按已声明解析规则为 16-mixed。
请求值与解析值都必须记录；严格 FP32 baseline 需显式 32-true。
相同 seed 的复现范围限定在相同设备／精度／配置，不承诺跨条件逐位相同。
允许浮点差异的 matmul 性能路径不得改变终止原因或 planning/simulator 计数。

每次决策同步传回执行所需 ego float32 trajectory；其余完整 prediction/noise/guidance audit 可延迟传输，但 simulator step 后由 artifact/replay 消费方显式取得已完成副本，保持原规划周期身份，按 trace dtype 检查有限性。CPU audit 生命周期不得决定仍供 PPO 使用的 device training fields 生命周期。

## Energy 与 profiling 边界

Energy provider 输入为有限、非负且时间严格递增的`time_s[N+1], speed_mps[N+1], step_distance_m[N]`；结果包含 metric 名称、实际距离和可选 J／mL。
Wh 与每公里量从同一结果派生。在线 proxy 从相邻实际 center 和执行速度求值，native reset-bounded 累计值单独保存；离线 FASTSim 采用完整 trace，失败／不支持车型／轨迹缺失／距离不一致直接暴露，不得换 provider fallback。指标窗口和单位不可混合，定义见 planning/evaluation protocol。

关闭 profiling MUST 不创建 CUDA Event、不额外同步、不改变调度、随机流或数值语义。
启用 profiling 的额外 bootstrap 同步不能进入普通 collection。不同时间量的解释归
[diagnostic protocol](../research/protocols/diagnostic-studies.md#benchmark-的测量语义)。

## 来源与待确认边界

来源：[旧 system contract](../agents/system-contract.md)、[runtime](../agents/contracts/runtime.md)、
[planner](../agents/contracts/planner.md)、[training](../agents/contracts/training.md)。
接受依据包括 [ADR 0028](../adr/0028-consolidate-runtime-ownership-and-evaluation-topology.md)、
[0038](../adr/0038-consolidate-scientific-workflows.md)、[0039](../adr/0039-unify-closed-loop-cadence.md)。

[ADR 0024](../adr/0024-add-plannerrft-energy-reward.md) 已将 slot/worker 求 reward 标为历史位置。
所有权的迁移依据是：[ADR 0022](../adr/0022-unify-metadrive-environment-slots.md) 将 RL reward
留给环境调用方；旧 system/training contract 明确了后续 collector-side 求值边界；
[ADR 0039](../adr/0039-unify-closed-loop-cadence.md) 接受在线／离线共享子步归约及审计输入。
这些来源支持客观 execution facts 与 reward 求值分离，不等于 ADR 0024 整体失效，
也不把原位置决定改写成新位置。本文仍为 proposed，等待全局 owner 原子切换；未做源码 conformance 审计。

## 代码与测试导航

| 修改面 | 实现／配置定位 | 相关测试（未运行） |
| --- | --- | --- |
| Cadence / execution | [ABI](../../src/eco_planner/contracts.py)、[execution](../../src/eco_planner/envs/metadrive/execution.py)、[domain](../../src/eco_planner/envs/domain/) | [execution consistency](../../tests/simulation/test_execution_consistency.py) |
| Config / resources | [jobs](../../src/eco_planner/jobs.py)、[runtime](../../src/eco_planner/runtime/)、[resources](../../configs/components/resources/) | [jobs](../../tests/configuration/test_jobs.py) |
| Slot / worker 生命周期 | [runtime envs](../../src/eco_planner/runtime/envs/)、[MetaDrive](../../src/eco_planner/envs/metadrive/) | [simulation tests](../../tests/simulation/) |
| RNG / sampler | [runtime random](../../src/eco_planner/runtime/random.py)、[sampling](../../src/eco_planner/planning/diffusion/sampling.py) | [sampling](../../tests/planning/test_sampling.py)、[rollout](../../tests/training/test_rollout.py) |
| Profiling / transfer | [host transfer](../../src/eco_planner/runtime/host_transfer.py)、[rollout profiling](../../src/eco_planner/rl/rollout/profiling.py) | [benchmark](../../tests/benchmarking/test_rollout.py) |
