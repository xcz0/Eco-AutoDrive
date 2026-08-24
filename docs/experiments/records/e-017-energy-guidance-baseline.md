# E-017 固定能耗场景矩阵与 longitudinal guidance 基线

[返回实验索引](../README.md)

**日期 / 类型 / 目的**：2026-08-24 / 正式固定 seed 基线 / 完成 Issue #1 的 6 场景、
4 guidance 能耗 benchmark，并验证相同配置与 seed 的重复性。

**代码**：两轮运行基于 `e4ac73abd209c5ccdb108c6ae8d220c12d8da7b4`，工作区包含
本次 `traffic_follow` 的 `S×40` 路线修正及对应配置/simulator 测试。每个 job 的
`runtime_metadata.json` 记录相同 Git HEAD 与 dirty state，`tracked_diff.patch` 保存运行时
实际实现。最终实验记录随完成 Issue #1 的后续提交入库。

**环境 / 模型**：Windows 10；Python 3.10.20；PyTorch 2.12.1+cu126；Lightning 2.6.5；
MetaDrive 0.4.3；NVIDIA GeForce RTX 3050 Laptop GPU 4 GiB，driver 596.49；单卡
`cuda:0`、`bf16-mixed`。checkpoint 为 `checkpoints/DP-Origin/model.pth`，EMA 276 tensors /
6,042,628 parameters。

**配置**：`configs/benchmark/energy_matrix.yaml` version 1；标准高斯 deterministic DDIM5
（`ddim_stochasticity=0`）；planner noise seed 0；scenario/map seeds 0–5；10 Hz trajectory
execution、2 Hz replanning、0.02 s physics world step、5-step execution prefix。车辆
`random_agent_model=false`。交通场景使用 seed 5、`S×40`、trigger traffic、density 0.05、
20-step warmup 和 600-step evaluation，实际路线 2421.569 m；限速场景使用 50→30→50 km/h
block profile。四组 profile 为无额外 guidance、longitudinal -1、0、+1。

**命令**：

```powershell
just energy-matrix outputs/energy_matrix/e-017-energy-guidance-baseline-run-1
just energy-matrix outputs/energy_matrix/e-017-energy-guidance-baseline-run-2
```

两轮分别累计 217.282 s 和 222.239 s 的 12 个 job runtime。每轮均包含 12 jobs、24 个
episode，launcher failure、runtime/reset failure 均为 0。两轮 12 组完整 episode payload
逐值相同；离散终止状态以及 energy、distance、route progress、mean speed 等连续指标均无漂移。

**Run 1 逐回合结果**（Run 2 逐值相同；energy 为 mL，单位距离指标为 mL/km）：

| 场景 | Guidance | 终止 | Distance (m) | Energy (mL) | mL/km | Progress | Mean speed (m/s) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| cruise | baseline | arrive | 262.516 | 12.933 | 49.265 | 0.983307 | 11.514 |
| cruise | negative | arrive | 267.738 | 12.888 | 48.137 | 0.982944 | 10.710 |
| cruise | zero | arrive | 262.516 | 12.933 | 49.265 | 0.983307 | 11.514 |
| cruise | positive | arrive | 269.086 | 14.768 | 54.883 | 0.983529 | 13.942 |
| curve | baseline | arrive | 238.720 | 11.153 | 46.720 | 0.980749 | 9.626 |
| curve | negative | out_of_road | 138.717 | 6.344 | 45.730 | 0.557576 | 8.563 |
| curve | zero | arrive | 238.720 | 11.153 | 46.720 | 0.980749 | 9.626 |
| curve | positive | arrive | 244.707 | 12.301 | 50.267 | 0.980300 | 11.073 |
| intersection | baseline | time_truncation | 182.277 | 7.950 | 43.617 | 0.835300 | 6.076 |
| intersection | negative | out_of_road | 137.134 | 5.925 | 43.204 | 0.607400 | 5.552 |
| intersection | zero | time_truncation | 182.277 | 7.950 | 43.617 | 0.835300 | 6.076 |
| intersection | positive | time_truncation | 215.234 | 10.156 | 47.184 | 0.930733 | 7.174 |
| merge_lane_change | baseline | arrive | 204.802 | 9.748 | 47.597 | 0.978764 | 10.503 |
| merge_lane_change | negative | arrive | 212.995 | 9.893 | 46.446 | 0.980010 | 9.551 |
| merge_lane_change | zero | arrive | 204.802 | 9.748 | 47.597 | 0.978764 | 10.503 |
| merge_lane_change | positive | arrive | 208.776 | 10.810 | 51.778 | 0.981197 | 12.502 |
| speed_limit_50_30_50 | baseline | arrive | 258.642 | 12.687 | 49.054 | 0.983867 | 11.394 |
| speed_limit_50_30_50 | negative | arrive | 263.595 | 12.634 | 47.929 | 0.981883 | 10.586 |
| speed_limit_50_30_50 | zero | arrive | 258.642 | 12.687 | 49.054 | 0.983867 | 11.394 |
| speed_limit_50_30_50 | positive | arrive | 265.369 | 14.285 | 53.830 | 0.985852 | 13.471 |
| traffic_follow | baseline | time_truncation | 662.494 | 32.151 | 48.530 | 0.275626 | 11.042 |
| traffic_follow | negative | out_of_road | 98.850 | 4.716 | 47.705 | 0.041532 | 10.297 |
| traffic_follow | zero | time_truncation | 662.494 | 32.151 | 48.530 | 0.275626 | 11.042 |
| traffic_follow | positive | time_truncation | 753.733 | 39.481 | 52.380 | 0.307441 | 12.562 |

全部 24 回合无 collision。negative 的 curve、intersection 和 traffic_follow 为
`out_of_road`；其余结果为 15 个 arrive 和 6 个 time truncation。交通场景四组运行均完成
20-step warmup，88–99% planning frames 观察到背景参与者。

**Guidance 观察**：zero 与 baseline 在全部场景及两轮运行中逐值相同，符合 neutral guidance
契约。negative 在 6/6 场景降低 mean speed，并在三个场景同时降低 progress、提前出界；
positive 在 6/6 场景提高 mean speed 和 mL/km，在 intersection/traffic_follow 提高 progress。
因此 longitudinal guidance 对执行速度/进度有明确可观察作用，但 negative 的较低 total energy
主要与少走和提前失败混合，不能解释为节能改善。

**验证**：相关配置与新增长路线定向测试 21 passed；实验工作区的 `just test` 为 233 passed、
33 deselected，合并远端 import-order 回归后最终为 234 passed、33 deselected；`just test-sim`
为 21 passed、245 deselected；`just lint` 与 `just format-check` 通过。

**产物**：

- `outputs/energy_matrix/e-017-energy-guidance-baseline-run-1/`
- `outputs/energy_matrix/e-017-energy-guidance-baseline-run-2/`

每个根目录含 matrix manifest/summary；每个 job 含 resolved config、Hydra overrides、runtime
metadata、tracked diff、job/episode summary 和完整 trace。

**结论边界**：本记录证明固定场景、seed、DDIM5 与 guidance 配置在当前单 GPU BF16 环境可
严格重复，并提供 #3/#59 所需的 energy、distance、progress、speed 和失败/终止基线。energy
仅是从 0.1 s 实际运动学执行 trace 重算的 MetaDrive fuel-consumption proxy，不是真实车辆能耗；
本实验不比较其他能耗模型，不决定 PPO reward normalization，也不支持 FASTSim 或真实动力系统结论。
