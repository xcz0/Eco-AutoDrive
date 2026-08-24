# E-022 Persistent vector evaluation benchmark (RTX 3050)

**日期 / 类型 / 目的**：2026-08-24 / 本机正式基准 / 为 Issue #65（父 Issue #62）比较旧版 chunked vector、当前 persistent + refill vector 与 serial evaluation，并记录 worker-pool 生命周期、batch fill 和 worker timing。

**状态**：完成。基准只改变 execution strategy；planner、sampler、guidance、checkpoint、场景、seed、precision、horizon 和 video 设置保持不变。

## 代码与配置

- 机器：Windows 10 build 26200；AMD64 Family 25 Model 80（16 logical cores）；NVIDIA GeForce RTX 3050 Laptop GPU（4 GiB，driver 596.49）；Python 3.10.20；PyTorch 2.12.1+cu126；MetaDrive 0.4.3。
- 模型：`checkpoints/DP-Origin/model.pth`，EMA 6,042,628 parameters，reference DPM-10，`bf16-mixed`。
- 场景：6 个 traffic scenarios（long straight / long mixed 各 map seeds `0,1,2`），history warmup 20、evaluated horizon 300、video 关闭、`deterministic=true`、2 vector slots、每 worker 8 PyTorch threads。
- 历史 baseline 使用 #65 前的 `088e5f3`，并将 #66 的 deterministic CUDA resource setup 应用到临时 benchmark worktree；当前 persistent 与 serial 使用 `090d243` 加本 Issue 的未提交 profiling 和 reset 修复。各 runtime metadata 均保存实际 HEAD 与 tracked diff。

基准入口为 `scripts/benchmark_vector_evaluation.py`，仅包装真实 `VectorMetaDriveEnv` 以写出 `vector_benchmark.json`；正常 evaluation 运行不采集这些 timing。旧版 runner 将 6 个 scenarios 按 2 slots 分为 3 个 chunk；当前 runner 用一个 pool 从同一队列 refill。serial 使用同一 resolved config，并将 `vector_env_slots=null`。

## 结果

| Execution | Matrix wall (s) | Pool init / teardown | Batch fill | Batch wall median (ms) | Worker wait median (ms) | Imbalance median (ms) |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| serial | 95.44 | n/a | n/a | n/a | n/a | n/a |
| chunked vector | 96.70 | 3 / 3 | 1.00 | 35.47 | 240.75 | 16.66 |
| persistent + refill | 82.02 | 1 / 1 | 1.00 | 34.48 | 243.44 | 16.46 |

chunked vector 的 pool initialization 合计 20.68 s、teardown 合计 6.18 s；persistent 路径分别为 6.87 s 和 2.13 s。persistent 比 chunked 少 14.68 s（15.2%），也比同配置 serial 少 13.42 s（14.1%）。两条 vector 路径的 worker wait 与 imbalance 中位数处于同一量级。

六个场景在三条路径中都完成，并一致保存 60 planning cycles 和 300 simulator steps。逐值 trace 会因实际进程调度产生微小数值差异；但 scenario identity、termination status、horizon、sampler/guidance 和所有主要执行边界一致。基准场景均到达相同 horizon，因此实际 batch fill 始终为 1.00；提前完成 slot 的 refill 边界由 `tests/unit/evaluation/test_episode.py` 的 3-scenario / 2-slot 回归测试覆盖：第一个 slot 在第 1 个 planning cycle 结束后立即 reset 为第三个 scenario，且 batch size 连续保持 2。

本次 bring-up 还发现同 map slot refill 会在 MetaDrive reset 的第 0 步留下 `{agent: None}` action map，过去被误判为 trajectory continuation。`KinematicTrajectoryPolicy` 现在只在该 reset 边界允许空 cache；非零 step 仍明确失败，回归测试见 `tests/unit/test_execution_policy.py`。

## 产物与结论边界

- `outputs/benchmarks/issue65-2026-08-24/baseline-source/outputs/benchmarks/issue65-baseline/`
- `outputs/benchmarks/issue65-2026-08-24/persistent-fixed/`
- `outputs/benchmarks/issue65-2026-08-24/serial/`

本记录支持在本 RTX 3050、该 6-scenario traffic benchmark 下使用 persistent 2-slot vector evaluation；它不支持将该 wall-time 比、slot 数或默认 execution mode 外推到 RTX A4000、不同 scenario matrix、不同 horizon 或不同硬件。E-016 的历史 vector 墙钟仍保留其原有结论边界。
