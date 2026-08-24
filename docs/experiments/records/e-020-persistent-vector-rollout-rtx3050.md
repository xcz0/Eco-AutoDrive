# E-020 Persistent vector rollout（RTX 3050 Laptop）

**日期 / 类型 / 目的**：2026-08-24 / 本机验证 / 为 Issue #63 在 E-015 的 RTX 3050 Laptop
条件下验证跨 PPO update 复用 MetaDrive vector worker pool，并分开记录 cold-start 与稳态。

## 代码与环境

- Git HEAD：`440819b5db705b16601ea600d82b8e3c2c1fe356`；运行时 `git status --short` 为空。
- Windows 10 build 26200；Python 3.10.20；PyTorch 2.12.1+cu126；Lightning 2.6.5；
  MetaDrive 0.4.3；Pydantic 2.13.4。
- GPU：NVIDIA GeForce RTX 3050 Laptop GPU，4,294,508,544 bytes；driver 596.49；CUDA
  runtime 12.6；推理使用 `bf16-mixed`。
- CPU：AMD64 Family 25 Model 80，16 logical cores；PyTorch intra-op/inter-op threads 均为 8。

## 配置与命令

配置与 E-015 rollout 基线相同：no-traffic `S` 场景，batch `[1, 2, 4, 8]`，serial/vector，
每 slot 8 transitions，1 warmup update、3 个稳态 measured updates、3 repeats，DDIM-5
policy-guided planner。scenario/noise/policy-action seed bases 分别为 `0`、`0`、`10000`。

```powershell
just benchmark-rollout hydra.run.dir=outputs/benchmarks/issue-63/rtx3050-rollout-persistent-workers/run
```

## 结果

下表为同次运行的稳态中位数；`cold collection` 是每个 repeat 的第一个 update，包含 worker
初始化。`startup` 单独测量只包括建池时间。

| Collector | Batch | Steady transitions/s | Steady collection (s) | Cold collection (s) | Worker startup (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| serial | 1 | 3.44 | 2.33 | 2.36 | — |
| serial | 2 | 3.40 | 4.71 | 4.68 | — |
| serial | 4 | 3.42 | 9.37 | 9.79 | — |
| serial | 8 | 3.39 | 18.86 | 19.09 | — |
| vector | 1 | 3.68 | 2.17 | 11.94 | 9.72 |
| vector | 2 | 5.95 | 2.69 | 13.54 | 10.69 |
| vector | 4 | 10.82 | 2.96 | 16.48 | 13.19 |
| vector | 8 | 26.66 | 2.40 | 13.68 | 10.93 |

所有 vector decision batch fill ratio 为 1.0；worker wait 中位数为每 transition 0.25–0.33 s，
worker busy imbalance 中位数为 0.00 s。本次执行显示：建池成本在首轮仍占主导，但 worker
pool 跨 update 复用后，vector 在所有测量 batch 上均超过本次同配置 serial 路径。

相对于 E-015 当时的 vector 数值（0.77、1.42、2.49、3.46 transitions/s），本次稳态 vector
为 3.68、5.95、10.82、26.66 transitions/s。该对照支持 Issue #63 解决了短 rollout 中的固定
worker lifecycle 成本；E-015 与本次的 serial 绝对值不同，因此不将跨运行的 serial 差异解释为
本 Issue 的效果。

## 产物与结论边界

- `outputs/benchmarks/issue-63/rtx3050-rollout-persistent-workers/run/rollout_throughput.json`
- 同目录的 `resolved_config.yaml`、`.hydra/` 与 `tracked_diff.patch`。

本结果只覆盖 RTX 3050 Laptop、no-traffic、当前短 rollout profile 和给定 batch 范围；不外推到
traffic mode、不同 PPO batch/资源 profile 或 RTX A4000。Issue #63 的另一台机器 benchmark 仍需
在该机器上独立执行和登记。
