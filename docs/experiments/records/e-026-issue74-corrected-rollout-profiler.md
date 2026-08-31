# E-026 Issue #74 corrected rollout profiler（RTX 3050 Laptop）

**日期 / 类型 / 目的**：2026-08-28 / 本机正式基准 / 修正 policy-guided rollout 的
PPO scheduler horizon 与 CUDA 异步计时边界，并在真实训练形状下建立 B=2/B=4 基线。

## 代码与环境

- Git HEAD：`f20976ebbfa7b0f2c6207936abbc7605fe219866`，运行时 tracked diff 保存本次
  profiler、benchmark schema、测试与 system contract 修改。
- Windows 10 build 26200；Python 3.10.20；PyTorch 2.12.1+cu126；Lightning 2.6.5；
  MetaDrive 0.4.3；Pydantic 2.13.4。
- NVIDIA GeForce RTX 3050 Laptop GPU（4,294,508,544 bytes），CUDA 12.6，BF16 mixed
  precision；CPU 为 AMD64 Family 25 Model 80、16 logical cores、8 Torch threads。

## 配置与命令

no-traffic、DDIM-5 policy guidance，两个逻辑 scenarios 循环扩展到 B=2/B=4；每 slot
16 transitions，每个 update 为 4 PPO epochs、minibatch 16，即 B=2 时 8 optimizer
steps/update、B=4 时 16 optimizer steps/update。每个 batch 独立执行 1 warmup update、
3 measured updates、3 repeats；scenario/noise/policy-action seed bases 为 `0`、`0`、`10000`。

```powershell
just benchmark rollout `
  "benchmark.batch_sizes=[2,4]" `
  "benchmark.collector_modes=[vector]" `
  "benchmark.transitions_per_slot=16" `
  "benchmark.ppo_epochs=4" `
  "benchmark.ppo_minibatch_size=16" `
  "benchmark.warmup_updates=1" `
  "benchmark.measured_updates=3" `
  "benchmark.repeats=3" `
  "hydra.run.dir=outputs/benchmarks/issue74-2026-08-28/corrected-profiler/run"
```

## 结果

下表均为 9 个 measured update samples 的中位数。decision 是每 update 16 个 planning
rounds 的 host wall 总和；bootstrap 是末尾一个 batch 的完整 host wall，不再只量 kernel
launch。audit accelerator copy 位于独立 stream，不能与 host wall 相加。

| B | Collection (s) | PPO (s) | End-to-end (s) | transitions/s | Decision (s) | Bootstrap (s) | Environment (s) | Unattributed (s) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 4.718 | 0.355 | 5.023 | 6.78 | 4.415 | 0.099 | 0.098 | 0.091 |
| 4 | 4.721 | 0.597 | 5.313 | 13.56 | 4.323 | 0.104 | 0.114 | 0.159 |

当前 stream 的 accelerator phase 总量如下；它们用于比较 GPU 工作，不与 host wall 或
独立 audit stream 直接相加。

| B | Prepare/reference DDIM (s) | Policy forward (s) | Action sampling (s) | Guided DDIM (s) | Execution D2H (s) | Bootstrap prepare (s) | Bootstrap policy (s) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 1.515 | 0.080 | 0.044 | 2.681 | 0.0048 | 0.094 | 0.0051 |
| 4 | 1.509 | 0.077 | 0.046 | 2.560 | 0.0047 | 0.098 | 0.0050 |

其余边界中，collate wall 为 B=2 `0.0021 s`、B=4 `0.0027 s`；deferred audit
accelerator copy 为 `0.0193 s` / `0.0202 s`，resolve host wall 为 `0.0179 s` /
`0.0224 s`。所有 collection residual samples 均为非负，最小值分别为 `0.0842 s` 与
`0.1508 s`。cold end-to-end 中位数为 B=2 `12.983 s`、B=4 `15.077 s`，与 steady
state 分开保留。

## 结论边界

- scheduler horizon 已覆盖全部 warmup/measured updates 及其真实 epoch/minibatch optimizer
  steps，未再出现 `PPO update would exceed the configured scheduler horizon`。
- corrected profiler 确认 guided DDIM 是 decision 最大 accelerator phase，prepare/reference
  DDIM 次之；bootstrap 的主要成本同样是 prepare/reference DDIM，而不是 policy value head。
- B=4 在 collection wall 基本不变时将 throughput 提高到 B=2 的约 2 倍；该结论只适用于
  本机、当前两个 scenarios、DDIM-5、BF16 与给定 seeds/workload。
- `outputs/benchmarks/rollout/perf-diagnosis-collection-shape/` 的旧 end-to-end/throughput
  仍是其历史 workload 的 host wall；旧 `planner_decision_wall_s`、
  `planner_bootstrap_wall_s` 与 `collection_overhead_s` 因异步归因和边界混合而不再用于结论。
- 本实验没有修改 planner、critic、sampler、guidance、随机流或 resource profile，也不构成
  性能优化 before/after；Issue #74 的执行路径优化仍未完成。

## 产物

- `outputs/benchmarks/issue74-2026-08-28/corrected-profiler/run/rollout_throughput.json`
- 同目录 `resolved_config.yaml`、`.hydra/` 与 `tracked_diff.patch`
