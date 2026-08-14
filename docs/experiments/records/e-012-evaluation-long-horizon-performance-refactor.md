# E-012 Evaluation 长程端到端性能重构

**日期 / 类型 / 目的**：2026-08-14；本机性能验收；在不改变 Artifact v3、场景状态、终止原因、
planning-cycle 数或 simulator-step 数的条件下，降低两个长程 traffic 回合的独立进程总墙钟。

**代码**：候选为本次未提交的 evaluation 重构，运行时 `git_head` 为
`44db88b0aeede50d350acb1a0cd77c38d65a117a`，具体未提交 diff 见每个候选输出目录的
`tracked_diff.patch`。基线是同一 detached `44db88b` worktree，仅应用 NumPy 数值标量、Fabric
mixed-precision 与模型 BF16 buffer/dtype 的必要正确性修复；保留旧的推理结果传输、trace record
累积/拼接、热循环有限性检查和压缩 NPZ。基线运行不设置 `matmul_precision=high`；候选 CUDA 路径设置
`torch.set_float32_matmul_precision("high")`。

**环境**：Windows 10 build 26200；Python 3.10.20；PyTorch 2.12.1+cu126；Lightning 2.6.5；
MetaDrive 0.4.3；RTX 3050 Laptop GPU；CUDA BF16；headless、视频关闭。

**数据 / 模型**：`traffic/full` 的 `long_straight` 和 `long_mixed`，地图与噪声 seed 均为 0，
`traffic_density=0.05`、`traffic_mode=trigger`、query radius 100 m；checkpoint
`checkpoints/DP-Origin/model.pth`，EMA 276 tensors、6,042,628 parameters；DPM-10、guidance none。

**配置**：默认 `configs/evaluation/traffic.yaml`；`video.enabled=false`。每组独立新进程运行 5 次；
每次均包含两个场景、20-step traffic warmup 和 Artifact 落盘。计时是从 Python 进程启动到退出的
PowerShell stopwatch；`run_evaluation` 墙钟取 `runtime_metadata.json` 的 `elapsed_seconds`。

**命令**：候选使用：

```powershell
.venv\Scripts\python.exe scripts\evaluate.py --config-name evaluation/traffic `
  video.enabled=false hydra.run.dir=outputs/perf/evaluation-full-candidate-runN
```

基线从 `outputs/perf/evaluation-baseline-worktree` 执行同一入口，以 `PYTHONPATH` 指向该 worktree，
并显式覆盖 checkpoint 的绝对路径及 `hydra.run.dir=...evaluation-full-baseline-runN`。

**结果**：

| 指标 | correctness-only 基线（5） | 候选（5） | 中位变化 |
| --- | ---: | ---: | ---: |
| 进程总墙钟 (s) | 108.88，MAD 0.94 | 103.60，MAD 1.92 | -4.85%（-5.28 s） |
| `run_evaluation` (s) | 99.53，MAD 0.60 | 94.35，MAD 1.04 | -5.21% |
| CUDA peak allocated | 38,554,112 B | 38,401,024 B | -153,088 B |
| 单场景 `trace.npz` 中位大小 | 7,709,854 B（压缩） | 38,421,831 B（未压缩） | +398% |

所有 10 个基线回合和 10 个候选回合的场景结果完全一致：`long_straight` 为
`crash_vehicle`、328 planning cycles、1,640 simulator steps；`long_mixed` 为 `crash_vehicle`、
149 planning cycles、741 simulator steps。20 个 trace 均由 `load_trace_artifact` 严格读取；候选的
未压缩 NPZ 保持 Artifact v3 的文件名、key、shape 和 dtype。候选不依靠提前结束、执行更少 simulator
steps、跳过 trace 字段或掩盖失败获得速度。

先导 300-step 双场景筛选中，`matmul_precision=high` 的进程中位为 37.12 s，`highest` 为 39.31 s，
下降 5.56%，且差值大于两组 MAD（1.20 s、0.03 s），因此纳入正式候选。

**产物**：

- `outputs/perf/evaluation-full-baseline-run1` 至 `run5`
- `outputs/perf/evaluation-full-candidate-run1` 至 `run5`
- `outputs/perf/evaluation-refactor-300-run1` 至 `run3`
- `outputs/perf/evaluation-refactor-high-300-run1` 至 `run3`

**结论边界**：在此机器、GPU、checkpoint、DPM-10、headless traffic/full 配置下，候选通过“中位改善
同时大于 3% 和两组 MAD”的门槛。该结果不证明其他 GPU、其他 sampler、guidance、视频开启或不同
traffic seed 的绝对耗时；未压缩 NPZ 明显增加磁盘占用，这是以总墙钟为首要目标下接受的 Artifact v3
保持策略。
