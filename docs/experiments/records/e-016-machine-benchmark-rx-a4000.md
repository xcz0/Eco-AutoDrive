# E-016 Machine scaling baseline (RTX A4000)

**日期 / 类型 / 目的**：2026-08-21 / 本机性能基线（远程训练机） / 为 Issue #55（父 Issue #49）在不同于 E-015 开发机的第二台机器上复现 planner、vector environment 与 RL rollout scaling，据实给出该机器的 worker/batch 建议区间。

**状态**：本机正式基线完成；覆盖 E-015 中“在远程训练机复现实验并给出 worker/batch 建议”一项。planner 饱和点已在 B>16 量到；三种 evaluation 执行模式（serial / job-level / vector）的墙钟对照均已正式运行并合并成 `scripts/report_evaluation_benchmark.py` 的正式报告。job-level 此前被本机并发多进程加载 torch 的 `c10.dll` 初始化崩溃阻塞，已定位并修复根因（import 顺序违反“Torch 必须先于 MetaDrive/Panda3D 加载”约定），补齐数据。

## 代码与环境

- Git HEAD：`96bcb152d6323c2165a3d992a88c31948366735d`；运行时另有 E-015 记录的一个未提交小编辑，与本实验无关；每个正式报告目录内保存了当时的 `git status --short` 与 `tracked_diff.patch`。job-level 墙钟运行在 HEAD 上加了两处未提交改动：`src/eco_planner/evaluation/artifacts/io.py` 的 torch-before-metadrive import 顺序修复，以及回归测试 `tests/unit/evaluation/test_import_order.py`。
- Windows 10 build 26200；Python 3.10.20；PyTorch 2.12.1+cu126；Lightning 2.6.5； MetaDrive 0.4.3；Pydantic 2.13.4。
- CPU：Intel(R) Xeon(R) w5-2455X，12 physical / 24 logical cores；PyTorch intra-op/inter-op threads 均为 12（自动取 logical cores 的一半）。
- GPU：NVIDIA RTX A4000，17,170,956,288 bytes；driver 556.18；CUDA runtime 12.6；推理使用 `bf16-mixed`。
- 模型：`checkpoints/DP-Origin/model.pth`，EMA 6,042,628 parameters。

## 配置与命令

与 E-015 相同的正式运行条件（关 render、3 次 warmup、10 次 measured cycles、3 repeats；rollout 为 1 warmup update、3 measured updates、3 repeats）：

```powershell
just benchmark-throughput hydra.run.dir=outputs/benchmarks/acceptance/local-throughput/run
just benchmark-throughput-traffic hydra.run.dir=outputs/benchmarks/acceptance/local-throughput-traffic/run
just benchmark-rollout hydra.run.dir=outputs/benchmarks/acceptance/local-rollout/run
```

Planner/environment 运行使用 reference DPM-10，planner batch sizes 为 `[1, 2, 4, 8, 16]`，environment worker counts 为 `[1, 2, 4, 8]`。无交通场景为 `S`、scenario seed 0；交通场景为 `SCSCSCSCSC`、density 0.05、trigger mode、scenario seed 0。planner noise seeds 为 `0..15`。

Rollout 使用 DDIM-5 和 `orthogonal_policy`，场景为 `S`/`SC`，collector modes 为 `serial`/`vector`，batch sizes 为 `[1, 2, 4, 8]`，每 slot 8 transitions；scenario、noise、policy-action seed bases 分别为 0、0、10000。完整参数以各运行目录中的`resolved_config.yaml` 为准。

CPU/GPU utilization 由运行外部每秒采样，覆盖模型加载、warmup 和正式测量。本次采样只得到可用的 GPU 利用率；CPU `Get-Counter` 采样在 PowerShell 5.1 下恒返回 0，CPU utilization 未纳入本次结论。

## Planner reference-only scaling

下表均为 30 个 measured samples 的中位数；wall time 是一次 batch inference 的墙钟时间，throughput 为 batch samples/s。

| Batch | Batch wall (ms) | Samples/s | CUDA peak allocated (MiB) |
| ---: | ---: | ---: | ---: |
| 1 | 71.04 | 14.08 | 35.68 |
| 2 | 77.06 | 25.95 | 38.63 |
| 4 | 80.25 | 49.85 | 45.17 |
| 8 | 84.86 | 94.27 | 56.95 |
| 16 | 77.48 | 206.50 | 80.32 |

B=8→16 从 94.27 翻倍到 206.50，仍接近线性扩展且显存远未耗尽，未测到 planner saturation。相比 E-015（RTX 3050）本机 B=16 已明显更高，但仍不可据此宣称饱和值或跨机器默认值。

## Planner scaling B>16（本机饱和点）

在 B=16 之后继续扩大 batch，测到吞吐饱和。同一 reference DPM-10、关 render、3 次 warmup、10 次 measured、3 repeats；`benchmark.batch_sizes=[16,32,64,128,256,512,1024,2048,4096]`、`benchmark.worker_counts=[4]`，输出`outputs/benchmarks/acceptance/local-throughput-bpast16/run`。下表中位数均为 30 个 measured samples：

| Batch | Batch wall (ms) | Samples/s | CUDA peak allocated (MiB) |
| ---: | ---: | ---: | ---: |
| 16 | 79.2 | 201.99 | 76.6 |
| 32 | 84.4 | 379.10 | 121.1 |
| 64 | 87.3 | 732.85 | 211.0 |
| 128 | 100.4 | 1275.23 | 392.4 |
| 256 | 127.1 | 2014.08 | 752.4 |
| 512 | 222.6 | 2300.48 | 1472.5 |
| 1024 | 423.5 | 2417.71 | 2908.5 |
| 2048 | 840.4 | 2437.06 | 5783.2 |
| 4096 | 4054.7 | 1010.18 | 11535.2 |

吞吐增长在 B=128→256 后明显放缓，B≈512–2048 进入约 2300–2440 samples/s 的平台区，B=4096 反而回退到 1010 samples/s；峰值显存在 B=4096 达约 11.3 GiB，仍远未耗尽 16 GiB。因此本机 planner 饱和是**计算/带宽受限而非显存受限**，饱和点约 B=512–2048；B=16 只是量程内近似线性的低端，不是饱和值。该饱和点仍只针对本机，不写入跨机器默认值。

## Vector environment scaling

`env steps/s` 是完整 batch step 的 worker environment throughput；`command wait` 是每个step 主进程实际等待 worker command 完成的时间。

| Traffic | Workers | Env steps/s | Command wait/step (ms) | Worker imbalance/step (ms) |
| --- | ---: | ---: | ---: | ---: |
| no traffic | 1 | 189.32 | 0.69 | 0.00 |
| no traffic | 2 | 313.25 | 1.31 | 0.29 |
| no traffic | 4 | 454.73 | 2.52 | 0.52 |
| no traffic | 8 | 485.25 | 7.03 | 0.26 |
| traffic | 1 | 73.58 | 1.52 | 0.00 |
| traffic | 2 | 167.02 | 1.61 | 0.31 |
| traffic | 4 | 265.54 | 2.64 | 1.08 |
| traffic | 8 | 256.78 | 10.77 | 2.61 |

无交通模式下 8 workers 仍略有增长（485 > 455），但增幅有限且 command wait 从 2.52ms 升到 7.03ms；交通模式在 4 workers 达到本次量程内峰值，8 workers 回退且 wait/imbalance 明显升高。综合两种 traffic mode，本机建议 environment workers 为 4，建议区间 2–4；无交通模式到 8 仍有边际增益，但不支持据此推高默认。该结论只针对本机。

环境 sensor 边界与 E-015 一致：无交通环境 `env.engine.sensors == {}`；交通环境只保留 IDM 需要的共享 lidar。

## RL rollout scaling

`transitions/s` 按实际 collected transitions 计算。Serial 与 vector 是两个真实 collector 路径；B=1 vector 不作为 serial 的替代标签。

| Collector | Batch | Transitions/s | Collection wall (s) | End-to-end wall (s) |
| --- | ---: | ---: | ---: | ---: |
| serial | 1 | 6.11 | 1.31 | 1.33 |
| serial | 2 | 6.08 | 2.63 | 2.67 |
| serial | 4 | 6.11 | 5.24 | 5.27 |
| serial | 8 | 6.00 | 10.67 | 10.73 |
| vector | 1 | 0.92 | 8.66 | 8.70 |
| vector | 2 | 1.45 | 11.04 | 11.08 |
| vector | 4 | 2.78 | 11.53 | 11.57 |
| vector | 8 | 4.78 | 13.39 | 13.54 |

所有 vector decision fill ratio 均为 1.0；B=8 policy-guided planner decision throughput 中位数为 62.09 samples/s。Virtual throughput 随 batch 增长，但在 B≤8 仍未超过 serial；vector collection 的 overhead（含每次 collection 重建 worker pool）在短 rollout 中占主导。与 E-015 一致，不把 vector collector 设为默认，也不据此修改 collector 或 PPO 数值行为。

## 整段运行的 GPU 利用率采样

| Run | n samples | GPU mean | GPU max | GPU active fraction | Active-only GPU median |
| --- | ---: | ---: | ---: | ---: | ---: |
| no-traffic throughput | 95 | 12.8% | 40% | 92% | 13.0% |
| traffic throughput | 108 | 2.2% | 43% | 13% | 7.0% |
| rollout | 393 | 14.6% | 86% | 79% | 14.0% |

GPU 均值均较低，主要因采样覆盖大量 CPU 环境阶段；Active-only median 说明 inference 确实占用 GPU。CPU utilization 因采样工具在 PowerShell 5.1 下失效而不列入结论。

## 三种 evaluation 执行模式的墙钟对照

对相同 scenario matrix（traffic、seeds `0,1,2` × densities `0.05,0.10` = 6 个 job，每个 job 评测 long_straight / long_mixed 两个场景，evaluated horizon 300、history warmup 20、video 关闭、DPM-10、bf16-mixed）正式运行串行、Joblib job-level 与 #54 vector 三种 execution.mode。墙钟为整次 matrix 的进程总时间。对应命令：

```powershell
just eval-matrix "hydra/launcher=basic" "evaluation.execution.mode=serial" "evaluation.execution.torch_threads_per_worker=null" "evaluation.execution.deterministic=false" "hydra.sweep.dir=outputs/evalmodes/serial"
just eval-matrix "evaluation.execution.mode=parallel" "evaluation.execution.torch_threads_per_worker=8" "evaluation.execution.deterministic=true" "hydra.sweep.dir=outputs/evalmodes/job_level"
just eval-matrix "hydra/launcher=basic" "evaluation.execution.mode=serial" "evaluation.execution.vector_env_slots=2" "evaluation.execution.torch_threads_per_worker=null" "evaluation.execution.deterministic=false" "hydra.sweep.dir=outputs/evalmodes/vector"
```

| Mode | execution.mode | launcher | vector_env_slots | Matrix wall (s) | 状态 |
| --- | --- | --- | ---: | ---: | --- |
| serial | serial | basic | null | 127.6 | 完成 |
| vector | serial | basic | 2 | 196.8 | 完成 |
| job-level | parallel | joblib_two（2 loky） | null | 145.2 | 完成 |

- 产物根：`outputs/evalmodes/serial`、`outputs/evalmodes/job_level`、`outputs/evalmodes/vector`，每根各含 6 个 `runtime_metadata.json`；`scripts/report_evaluation_benchmark.py` 正式对照报告已生成到 `outputs/evalmodes/evaluation_modes.json`（三根产物均可由脚本严格校验解析）。
- 短矩阵下 vector（196.8 s）比 serial（127.6 s）慢，与 E-015/E-016 rollout 中 “vector overhead 在短运行占主导”的结论一致；本矩阵不构成否定 #54 vector 的依据，也不据此修改 evaluation 默认 execution。
- job-level（2-worker 并行，deterministic=true，145.2 s）在本短矩阵下未跑赢与它设置不直接可比的 serial（127.6 s，deterministic=false）。job 层面真实并行的存在性已确认（2 worker 并发完成 6 job，外壁钟 = 2 批串行的和，job 各自墙钟中位 33.2 s），但本矩阵配置下不构成把 job-level 并行设为默认的依据，也不据此修改 evaluation 默认 execution。

### job-level 阻塞根因与修复

`joblib_two`（2 个 loky 并发进程）启动 worker 时此前于 `torch` 加载 `c10.dll` 处崩溃，报 `OSError WinError 1114` 与 `Windows fatal exception: access violation`，仅在 `n_jobs=2` 触发且沙箱内外一致复现。

定位后根因是 **import 顺序**：`src/eco_planner/evaluation/artifacts/io.py` 在模块顶层 `from metadrive.utils.doc_utils import generate_gif`（加载 Panda3D），早于其后的 `config → models → torch` 导入链。当 worker 进程把 `runner`（进而 `io`）作为第一个 `eco_planner` 模块导入时，Panda3D 先于 torch 加载，随后 torch 的 `c10.dll` 初始化失败（WinError 1114）。这在 2 个 worker 并发启动时更容易触发；serial 路径不崩只是因为 `evaluate.py` 先导入 `config`（→torch）再导入 `runner`。

修复：在 `io.py` 顶部、metadrive import 之前显式 `import torch`（`# noqa: I001, F401`），与仓库既有的 `envs/__init__.py` “Torch 必须先于 MetaDrive/Panda3D 加载”约定一致。新增回归测试 `tests/unit/evaluation/test_import_order.py`：在全新子进程中以自定义 `sys.meta_path` 探针导入 `io`，断言 `torch` 先于 `metadrive` 入 sys.modules；去掉该 import 时测试按预期失败（复现 WinError 1114）。

## 产物与验证

- `outputs/benchmarks/acceptance/local-throughput/run/throughput.json`
- `outputs/benchmarks/acceptance/local-throughput-traffic/run/throughput.json`
- `outputs/benchmarks/acceptance/local-rollout/run/rollout_throughput.json`
- `outputs/benchmarks/acceptance/local-throughput-bpast16/run/throughput.json`（planner 饱和点 B=16..4096）
- `outputs/evalmodes/serial/`、`outputs/evalmodes/job_level/`、`outputs/evalmodes/vector/`（evaluation 墙钟对照；每个 job 目录含 `runtime_metadata.json` 与 `resolved_config.yaml`）；`outputs/evalmodes/evaluation_modes.json`（三模式正式对照报告）
- 每个运行目录包含 `resolved_config.yaml` 与 `tracked_diff.patch`；父目录包含 `gpu-utilization.csv` 和 `cpu-utilization.csv`。
- 各正式报告均成功结束且 JSON 可由严格 schema 解析（脚本在结束前正常写出 JSON）。

## 结论边界与后续

本记录支持本机（RTX A4000、Xeon w5-2455X）使用 planner B=8–16（正式饱和区 B≈512–2048，但一般推理不采用如此大批次）、environment workers=2–4（实测峰值 4 workers，交通模式 8 已回退），并在现有短 rollout 设置下继续使用 serial collector，在现有短 evaluation matrix 下保持 serial execution 为默认（vector 与 job-level 在该矩阵均未跑赢 serial）。它不支持修改生产推理、环境调度、collector、evaluation 或 PPO 默认语义，也不支持把本机 planner 饱和点或任一墙钟比值写成跨机器不变量。

Issue #55 / #49 需完成的 item 已全部补齐：job-level（2-worker 并行）evaluation 墙钟 145.2 s 已测得，阻塞该测量的 `c10.dll` import 顺序问题已定位并修复、加了回归测试；serial / job-level / vector 三模式已合并成 `outputs/evalmodes/evaluation_modes.json` 的正式对照报告。
