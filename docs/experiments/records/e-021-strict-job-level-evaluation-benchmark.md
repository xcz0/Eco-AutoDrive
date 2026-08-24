# E-021 Strict job-level evaluation benchmark (RTX 3050)

**日期 / 类型 / 目的**：2026-08-24 / 本机正式基准 / 为 Issue #66（父 Issue #62）在 E-015 开发机上以完全相同的评测语义比较 serial 和 job-level execution，并验证 resource profile 的 worker/thread 预算。

**状态**：完成。此记录只比较 execution mode；不改变 planner、sampler、guidance、reward 或场景语义。

## 代码与环境

- Git HEAD：`c2e08a3`；运行时工作区含 Issue #66 的未提交修改：动态 joblib launcher、CUDA deterministic 对齐、对应测试和文档。每个 job 的 `tracked_diff.patch` 保存实际 diff。
- Windows 10 build 26200；Python 3.10.20；PyTorch 2.12.1+cu126；Lightning 2.6.5；MetaDrive 0.4.3；Pydantic 2.13.4。
- CPU：AMD64 Family 25 Model 80，16 logical cores。GPU：NVIDIA GeForce RTX 3050 Laptop GPU，4 GiB，driver 596.49，CUDA 12.6。
- 模型：`checkpoints/DP-Origin/model.pth`，EMA 6,042,628 parameters；sampler 为 reference DPM-10，precision 为 `bf16-mixed`。

## 相同配置

两次运行均为 traffic matrix：seeds `0,1,2` × densities `0.05,0.10`，共 6 个 Hydra jobs；每个 job 评测 `long_straight` 和 `long_mixed`，history warmup 20、evaluated horizon 300、video/render 关闭、`vector_env_slots=null`、`deterministic=true`、`torch_threads_per_worker=8`。每次运行开始前分别写入 resolved config；比较 job 0 的两份 resolved config，唯一差异为 `evaluation.execution.mode`。

`rtx3050_laptop` profile 设置 `evaluation_job_worker_count=2` 和 `torch_threads_per_worker=8`，所以 job-level CPU 预算为 `2 × 8 = 16`，恰等于可见 logical CPU count。Hydra launcher 的 `n_jobs` 和所有 job-level runtime metadata 的 `worker_count` 均为 2。

命令：

```powershell
just eval-matrix "hydra/launcher=basic" "evaluation.execution.mode=serial" "evaluation.execution.vector_env_slots=null" "evaluation.execution.deterministic=true" "hydra.sweep.dir=outputs/benchmarks/issue66-2026-08-24/serial-strict/jobs"
just eval-matrix "evaluation.execution.mode=parallel" "evaluation.execution.vector_env_slots=null" "evaluation.execution.deterministic=true" "hydra.sweep.dir=outputs/benchmarks/issue66-2026-08-24/job-level-strict/jobs"
```

## 结果

| Mode | Launcher | Worker count | Matrix wall (s) | Per-job wall median (s) | GPU mean / max | CPU mean / max |
| --- | --- | ---: | ---: | ---: | --- | --- |
| serial | basic | 1 | 227.70 | 32.16 | 39.22% / 99% | 25.10% / 55.01% |
| job-level | joblib `loky` | 2 | 147.28 | 38.72 | 42.94% / 99% | 27.09% / 46.19% |

job-level 的 matrix wall 比 serial 少 80.42 s（35.3%）。GPU/CPU 值来自运行期间的一秒外部采样：serial 109 个、job-level 68 个样本；GPU active fraction 分别为 83.5% 与 80.9%，active-only GPU median 分别为 43% 与 53%。每个目录保存完整的 `runtime_metadata.json`、`resolved_config.yaml`、`summary.json`、episode trace，以及 `gpu-utilization.csv` / `cpu-utilization.csv`。

## 产物与结论边界

- `outputs/benchmarks/issue66-2026-08-24/serial-strict/jobs/`
- `outputs/benchmarks/issue66-2026-08-24/job-level-strict/jobs/`

本次实测支持在此 RTX 3050 Laptop、此 6-job traffic matrix 和当前软件栈上使用 `rtx3050_laptop` 的 2-worker / 8-thread job-level profile；因此它可作为该机器选择 job-level execution 的依据。它不支持将该 wall-time 比值、worker 数或 execution mode 外推到 RTX A4000、其他 CPU/GPU、其他 scenario matrix 或其他 horizon。E-016 的历史 serial/job-level 墙钟仍不满足同配置条件，继续不用于 performance comparison。
