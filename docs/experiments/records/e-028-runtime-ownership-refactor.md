# E-028 runtime ownership 与 evaluation topology 重构验证（RTX 3050 Laptop）

**日期 / 类型 / 目的**：2026-09-01 / 本机实现验证与诊断性性能复测 / 验证 runtime
所有权、evaluation topology、rollout 分层和 TorchRL worker 拆分不改变数值与执行语义，并观察
持久 CUDA audit stream 候选。

## 代码与环境

- Git HEAD：`e9c086c1c12f0c5fb603e7524fa0529ce5cf4d0e`；benchmark artifact 的 tracked diff
  保存对应运行时的实现与测试修改；本记录、ADR 与 system contract 在复测后写入当前工作树。
- Windows 10 build 26200；Python 3.10.20；PyTorch 2.12.1+cu126；Lightning 2.6.5；
  MetaDrive 0.4.3；NVIDIA GeForce RTX 3050 Laptop GPU（4,294,508,544 bytes）；CUDA 12.6；
  BF16 mixed precision；16 logical CPU cores、8 Torch threads。
- 官方 checkpoint：`checkpoints/DP-Origin/model.pth` 与 `args.json`。

## 正确性验证

- shared standard-normal sampler 与旧逐 slot loop 输出逐值相等，且每个 generator 的最终 state
  逐值相等。
- `just lint` 通过；`just typecheck` 为 0 errors、0 warnings。
- `just test-all-cpu`：80 passed、8 deselected。
- `just test-sim`：6 passed、82 deselected。
- `just test-gpu`：1 passed、87 deselected。

这些验证覆盖 config composition、topology→resource 派生、evaluation artifacts、host transfer、
rollout decision/profiling、worker failure traceback，以及真实 MetaDrive vector 生命周期。

## 固定 workload

rollout 使用 no-traffic、DDIM-5 policy guidance、B=2/4、每 slot 16 transitions、4 PPO epochs、
minibatch 16、1 warmup update、3 measured updates、3 repeats；scenario/noise/policy-action seed
bases 为 `0/0/10000`。throughput 使用 planner B=1/2/4、vector workers=1/2/4、3 warmup cycles、
10 measured cycles、3 repeats。完整 resolved config、overrides、raw samples 和 tracked diff 均保存
在下列产物目录。

## 诊断结果

| 变体 | B | collection 中位 (s) | transitions/s 中位 | audit copy accelerator 中位 (s) | audit resolve 中位 (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 重构前 baseline | 2 | 4.370 | 7.32 | 0.0218 | 0.0176 |
| 重构前 baseline | 4 | 4.994 | 12.82 | 0.0244 | 0.0234 |
| 持久 audit stream 候选 | 2 | 4.555 | 7.02 | 0.0217 | 0.0189 |
| 持久 audit stream 候选 | 4 | 5.054 | 12.66 | 0.0172 | 0.0262 |
| 最终独立 audit stream | 2 | 3.608 | 8.87 | 0.0156 | 0.0164 |
| 最终独立 audit stream | 4 | 3.910 | 16.37 | 0.0139 | 0.0211 |

持久候选的 evaluation planner throughput 为 B=1/2/4 `7.07/13.99/24.74 samples/s`，vector
environment 为 workers=1/2/4 `155.53/267.49/364.77 steps/s`；对应重构前值为
`5.35/10.24/20.94 samples/s` 与 `121.03/159.25/301.55 steps/s`。

同一机器不同运行时段的 planner 与 rollout 主阶段差异明显大于 stream lifetime 这一局部改动可
可靠归因的量，因此这些数值只作为本机诊断，不作为硬性性能验收或加速结论。持久候选在 B=2
没有显示一致 audit-copy 改善，且两个 batch 的 resolve wall 均未改善；最终不保留 stream 复用，
只保留 runtime-owned `HostTransfer` 结构边界。

## 结论边界

- 支持本次内部直接切换通过当前 CPU、simulator、GPU 契约验证；不声称统计上的端到端加速。
- 随机流、planner 输入输出、轨迹执行、reward、termination、checkpoint 和正式 artifact schema
  未改变；resolved execution config 与 benchmark-only schema 按计划改变。
- 持久 CUDA stream 结果只适用于本机当次状态；未来若重新考虑，需独立、交错并控制机器状态的
  A/B 证据，不能复用本记录作为默认值依据。

## 产物

- `outputs/benchmarks/runtime-refactor-2026-09-01/baseline-rollout/`
- `outputs/benchmarks/runtime-refactor-2026-09-01/baseline-throughput/`
- `outputs/benchmarks/runtime-refactor-2026-09-01/final-rollout/`（持久 stream 候选）
- `outputs/benchmarks/runtime-refactor-2026-09-01/final-throughput/`（持久 stream 候选）
- `outputs/benchmarks/runtime-refactor-2026-09-01/final-ephemeral-rollout/`（最终实现）
