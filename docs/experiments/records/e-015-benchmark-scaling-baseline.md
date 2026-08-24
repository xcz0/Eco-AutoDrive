# E-015 Benchmark scaling 本机基线

**日期 / 类型 / 目的**：2026-08-21 / 本机性能基线 / 为 Issue #55（父 Issue #49）
建立可追溯的 planner、vector environment 与 RL rollout scaling 基线，并据实给出本机建议区间。

**状态**：本机正式基线完成；Issue #55 部分验收。远程训练机、planner 饱和点和三种
evaluation 执行模式的正式墙钟对照尚未完成。

## 代码与环境


- Windows 10 build 26200；Python 3.10.20；PyTorch 2.12.1+cu126；Lightning 2.6.5；
  MetaDrive 0.4.3；Pydantic 2.13.4。
- CPU：AMD64 Family 25 Model 80，16 logical cores；PyTorch intra-op/inter-op threads
  均为 8。
- GPU：NVIDIA GeForce RTX 3050 Laptop GPU，4,294,508,544 bytes；driver 596.49；
  CUDA runtime 12.6；推理使用 `bf16-mixed`。
- 模型：`checkpoints/DP-Origin/model.pth`，EMA 6,042,628 parameters。

## 配置与命令

正式运行均关闭 render，并使用 3 次 warmup、10 次 measured cycles、3 repeats（rollout
为 1 warmup update、3 measured updates、3 repeats）。命令为：

```powershell
just benchmark-throughput hydra.run.dir=outputs/benchmarks/acceptance/local-throughput-v3/run
just benchmark-throughput-traffic hydra.run.dir=outputs/benchmarks/acceptance/local-throughput-traffic/run
just benchmark-rollout hydra.run.dir=outputs/benchmarks/acceptance/local-rollout/run
```

Planner/environment 运行使用 reference DPM-10，planner batch sizes 为
`[1, 2, 4, 8, 16]`，environment worker counts 为 `[1, 2, 4, 8]`。无交通场景为
`S`、scenario seed 0；交通场景为 `SCSCSCSCSC`、density 0.05、trigger mode、scenario
seed 0。planner noise seeds 为 `0..15`。

Rollout 使用 DDIM-5 和 `orthogonal_policy`，场景为 `S`/`SC`，collector modes 为
`serial`/`vector`，batch sizes 为 `[1, 2, 4, 8]`，每 slot 8 transitions；scenario、
noise、policy-action seed bases 分别为 0、0、10000。完整参数以各运行目录中的
`resolved_config.yaml` 为准。

CPU/GPU utilization 由运行外部每秒采样，覆盖模型加载、warmup 和正式测量。该采样只用于
判断整段运行的资源占用边界；它会漏掉短 CUDA kernel，不能解释为单次 inference 的 GPU
utilization。

## Planner reference-only scaling

下表均为 30 个 measured samples 的中位数；wall time 是一次 batch inference 的墙钟时间，
throughput 为 batch samples/s。

| Batch | Batch wall (ms) | Samples/s | CUDA peak allocated (MiB) |
| ---: | ---: | ---: | ---: |
| 1 | 134.04 | 7.46 | 34.03 |
| 2 | 137.33 | 14.56 | 36.84 |
| 4 | 145.96 | 27.41 | 43.08 |
| 8 | 139.56 | 57.32 | 54.31 |
| 16 | 140.09 | 114.21 | 76.60 |

B=16 相比 B=8 仍接近线性扩展，且显存远未耗尽，因此本实验没有测到 planner saturation。
本机当前已验证的高吞吐区间是 B=8–16；不能据此声称 B=16 是饱和值或跨机器默认值。

## Vector environment scaling

`env steps/s` 是完整 batch step 的 worker environment throughput；`command wait` 是每个
step 主进程实际等待 worker command 完成的时间，不与 worker busy time 重复命名。

| Traffic | Workers | Env steps/s | Command wait/step (ms) | Worker imbalance/step (ms) |
| --- | ---: | ---: | ---: | ---: |
| no traffic | 1 | 149.40 | 0.92 | 0.00 |
| no traffic | 2 | 194.29 | 2.02 | 0.14 |
| no traffic | 4 | 293.56 | 4.22 | 0.31 |
| no traffic | 8 | 279.63 | 11.90 | 2.43 |
| traffic | 1 | 74.50 | 1.16 | 0.00 |
| traffic | 2 | 110.68 | 1.88 | 0.08 |
| traffic | 4 | 173.42 | 3.61 | 0.74 |
| traffic | 8 | 158.05 | 15.64 | 3.38 |

两种 traffic mode 都在 4 workers 达到本次量程内峰值；8 workers 吞吐回退，同时 command
wait 和 imbalance 明显升高。因此当前开发机建议使用 4 environment workers，建议区间为
2–4；该结论不外推到远程训练机。

环境 sensor 边界保持不变：无交通环境的 `env.engine.sensors == {}`；交通环境只保留 IDM
需要的共享 lidar，planner observation 不消费该 lidar。本实验没有以性能推测为理由删除
traffic sensor。

## RL rollout scaling

`transitions/s` 按实际 collected transitions 计算。Serial 与 vector 是两个真实 collector
路径；B=1 vector 不作为 serial 的替代标签。

| Collector | Batch | Transitions/s | Collection wall (s) | End-to-end wall (s) |
| --- | ---: | ---: | ---: | ---: |
| serial | 1 | 4.37 | 1.83 | 1.85 |
| serial | 2 | 4.42 | 3.62 | 3.66 |
| serial | 4 | 4.34 | 7.37 | 7.44 |
| serial | 8 | 4.29 | 14.90 | 15.02 |
| vector | 1 | 0.77 | 10.36 | 10.41 |
| vector | 2 | 1.42 | 11.23 | 11.31 |
| vector | 4 | 2.49 | 12.85 | 12.99 |
| vector | 8 | 3.46 | 18.48 | 18.68 |

所有 vector decision fill ratio 均为 1.0；B=8 policy-guided planner decision throughput
中位数为 29.04 samples/s。Vector throughput 随 batch 增长，但在 B≤8 仍未超过 serial。
当前真实 collector 每次 collection 都重新创建 worker pool，启动成本在短 rollout 中占主导；
因此本实验不支持把 vector collector 设为默认，也不据此修改 collector 或 PPO 数值行为。

## 整段运行的资源采样

| Run | GPU mean | GPU max | GPU active fraction | Active-only GPU median | CPU median | CPU max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| no-traffic throughput | 4.85% | 43% | 15% | 33.0% | 30.78% | 83.59% |
| traffic throughput | 4.47% | 43% | 14% | 33.5% | 32.99% | 92.23% |
| rollout | 14.21% | 97% | 40% | 36.0% | 31.72% | 89.12% |

由于采样包括大量启动和 CPU 环境阶段，三组整段运行的 GPU utilization median 均为 0；表中
另列 active-only median，避免把 0 误读为 inference 没有使用 GPU。

## 产物与验证

- `outputs/benchmarks/acceptance/local-throughput-v3/run/throughput.json`
- `outputs/benchmarks/acceptance/local-throughput-traffic/run/throughput.json`
- `outputs/benchmarks/acceptance/local-rollout/run/rollout_throughput.json`
- 每个运行目录包含 `resolved_config.yaml` 与 `tracked_diff.patch`；父目录包含
  `gpu-utilization.csv` 和 `cpu-utilization.csv`。
- 最终三组正式报告均成功结束且 JSON 可由严格 schema 解析。相关 benchmark/runtime/
  collector 单测 46 项通过；vector environment simulator 定向测试 5 项通过。
- 正式运行前的失败诊断暴露并修正了 benchmark 默认 horizon 与 MetaDrive seed domain；只有
  上述 `local-throughput-v3`、`local-throughput-traffic`、`local-rollout` 被用于性能结论。

## 结论边界与后续

本记录支持当前开发机使用 planner B=8–16、environment workers=2–4，并在现有短 rollout
设置下继续使用 serial collector；其中 environment 的实测峰值是 4 workers。它不支持修改
生产推理、环境调度、collector 或 PPO 默认语义。

Issue #55 / #49 仍需完成：

1. 在 B>16 继续 planner scaling，直到测到吞吐或显存饱和；
2. 在远程训练机复现实验并给出该机器的 worker/batch 建议；
3. 对相同 scenario matrix 正式运行 serial、job-level 和 vector evaluation 墙钟对照。
