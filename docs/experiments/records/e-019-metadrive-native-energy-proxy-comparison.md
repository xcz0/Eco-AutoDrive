# E-019 MetaDrive native energy proxy 与执行 trace 对照

[返回实验索引](../README.md)

**日期 / 类型 / 目的**：2026-08-24 / 正式固定 seed 指标审计 / 完成 Issue #3：在
Issue #1 固定场景矩阵上比较 MetaDrive 原生 `step_energy`/`episode_energy` 与当前
`metadrive_fuel_proxy`，并为 #59 第一版 reward sanity check 给出临时 normalization
依据。

**代码与环境**：Git HEAD `8ac5476bd690963f1c74c7047a20eee664ce22ee`；运行时工作区
含本记录对应的 trace-audit 改动，完整 dirty state 及 patch 分别在每个 job 的
`runtime_metadata.json` 和 `tracked_diff.patch`。Windows 10 build 26200；Python
3.10.20；PyTorch 2.12.1+cu126；Lightning 2.6.5；MetaDrive 0.4.3；CUDA `cuda:0`，
`bf16-mixed`。checkpoint 为 `checkpoints/DP-Origin/model.pth`，EMA 276 tensors /
6,042,628 parameters。

**配置与命令**：沿用 `configs/benchmark/energy_matrix.yaml` version 1：DDIM5，
`ddim_stochasticity=0`，planner noise seed 0，6 个固定 scenario/map seed，10 Hz
运动学执行、2 Hz replanning、5-step execution prefix，且 `random_agent_model=false`。
四组 profile 是 baseline、longitudinal -1、0、+1。因 launcher 子进程在三个完成 job
后退出，余下 job 以同一 resolved config/overrides 单独运行；所有 12 个 job 均成功，
总 job runtime 为 389.195 s。

```powershell
just energy-matrix outputs/energy_matrix/e-018-native-energy-proxy-comparison
.venv/Scripts/python.exe scripts/evaluate.py --config-name=experiment/evaluate_energy_<job> planner/sampler=ddim5 planner/guidance=<profile> hydra.run.dir=outputs/energy_matrix/e-018-native-energy-proxy-comparison-run-1/<job>/<profile>
```

`step_energy` 和 `episode_energy` 由 MetaDrive `BaseVehicle.after_step()` 产生，单位都是
mL；前者是当前 simulator 子步、后者从 episode reset 起累计。项目的
`metadrive_fuel_proxy` 不使用该上游值，而是以实际保存的 0.1 s executed position 与
执行速度按同一公式重算；其 `total_ml` 只累计 evaluation execution，不含 traffic
history warmup。`ml_per_km = total_ml * 1000 / episode_distance_m`，分母明确为实际执行
路径长度，而不是 route length 或 route progress。

## 原生与重算 proxy 对照

全部 24 回合、6501 个 evaluation execution 子步中，原生 `step_energy`、
`episode_energy` final 和 `sum(step_energy)` 均为 **0.0 mL**；其中没有一个非零子步。
同时，执行 trace 重算的总量为 4.716--39.481 mL。这不是正常的零能耗驾驶：MetaDrive
在计算原生值时仍看到 kinematic policy 在该 phase 之前置入的零物理速度/位移，而 waypoint
在后续阶段才写入；因此上游累计边界不覆盖项目实际记录的 waypoint displacement。原生值
可作为该 phase-ordering 的审计证据，但不能用于当前 kinematic execution 的能耗比较、
reward 或 normalization。

| 指标 | 最小 | P25 | P50 | P75 | 最大 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原生 step/episode energy（mL） | 0 | 0 | 0 | 0 | 0 |
| trace-recomputed step energy（mL，6501 steps） | 0.000058 | 0.042708 | 0.052699 | 0.059181 | 0.181605 |
| trace-recomputed episode total（mL，24 episodes） | 4.716 | 9.748 | 11.727 | 12.933 | 39.481 |
| `ml_per_km`（24 episodes） | 43.204 | 46.720 | 48.033 | 49.265 | 54.883 |
| `ml_per_km`（仅 arrive，15 episodes） | 46.446 | 47.597 | 49.054 | 49.766 | 54.883 |

子步实际位移的范围为 0.00177--2.37576 m（中位 1.09375 m），执行速度范围为
0.01773--23.75761 m/s；6 步低于 0.1 m/s，72 步低于 1 m/s。即使在这些低速/极小位移步骤，
重算值仍有限且非负；无零位移，故本矩阵没有 `ml_per_km` 的零分母。traffic warmup 的
20 个 stationary steps 也被独立记录，且不进入 evaluation energy/distance 分子或分母。

## 固定 seed 指标表

下表为 trace-recomputed proxy。`distance` 是 episode 实际执行距离；结论必须同时阅读
total 与 `mL/km`，且提前终止不能与完整到达回合混为节能改善。

| 场景 | Guidance | Termination | Distance (m) | Total (mL) | mL/km | Progress | Mean speed (m/s) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| cruise | baseline | arrive | 262.516 | 12.933 | 49.265 | 0.983 | 11.514 |
| cruise | negative | arrive | 267.738 | 12.888 | 48.137 | 0.983 | 10.710 |
| cruise | zero | arrive | 262.516 | 12.933 | 49.265 | 0.983 | 11.514 |
| cruise | positive | arrive | 269.086 | 14.768 | 54.883 | 0.984 | 13.942 |
| curve | baseline | arrive | 238.720 | 11.153 | 46.720 | 0.981 | 9.626 |
| curve | negative | out_of_road | 138.717 | 6.344 | 45.730 | 0.558 | 8.563 |
| curve | zero | arrive | 238.720 | 11.153 | 46.720 | 0.981 | 9.626 |
| curve | positive | arrive | 244.707 | 12.301 | 50.267 | 0.980 | 11.073 |
| intersection | baseline | time truncation | 182.277 | 7.950 | 43.617 | 0.835 | 6.076 |
| intersection | negative | out_of_road | 137.134 | 5.925 | 43.204 | 0.607 | 5.552 |
| intersection | zero | time truncation | 182.277 | 7.950 | 43.617 | 0.835 | 6.076 |
| intersection | positive | time truncation | 215.234 | 10.156 | 47.184 | 0.931 | 7.174 |
| merge_lane_change | baseline | arrive | 204.802 | 9.748 | 47.597 | 0.979 | 10.503 |
| merge_lane_change | negative | arrive | 212.995 | 9.893 | 46.446 | 0.980 | 9.551 |
| merge_lane_change | zero | arrive | 204.802 | 9.748 | 47.597 | 0.979 | 10.503 |
| merge_lane_change | positive | arrive | 208.776 | 10.810 | 51.778 | 0.981 | 12.502 |
| speed_limit_50_30_50 | baseline | arrive | 258.642 | 12.687 | 49.054 | 0.984 | 11.394 |
| speed_limit_50_30_50 | negative | arrive | 263.595 | 12.634 | 47.929 | 0.982 | 10.586 |
| speed_limit_50_30_50 | zero | arrive | 258.642 | 12.687 | 49.054 | 0.984 | 11.394 |
| speed_limit_50_30_50 | positive | arrive | 265.369 | 14.285 | 53.830 | 0.986 | 13.471 |
| traffic_follow | baseline | time truncation | 662.494 | 32.151 | 48.530 | 0.276 | 11.042 |
| traffic_follow | negative | out_of_road | 98.850 | 4.716 | 47.705 | 0.042 | 10.297 |
| traffic_follow | zero | time truncation | 662.494 | 32.151 | 48.530 | 0.276 | 11.042 |
| traffic_follow | positive | time truncation | 753.733 | 39.481 | 52.380 | 0.307 | 12.562 |

zero 与 baseline 在所有字段逐值一致，符合 neutral guidance 契约。negative 在 6/6 场景降低
mean speed；在 curve、intersection、traffic_follow 同时少走、低 progress 且 out-of-road。
positive 在 6/6 场景提高 mean speed 与 `mL/km`，并在 intersection/traffic_follow 提高
progress。因此 `guidance -> speed/progress -> trace proxy` 的方向可解释；negative 的更低
total energy 不构成节能结论，尤其 traffic-follow 的 4.716 mL 伴随 0.042 progress。

## #59 smoke normalization 建议

只用于 `plannerrft_energy_v1` 的第一版 sanity check：使用 trace-recomputed
`ml_per_km / 50.0` 作为无量纲 energy raw quantity 的 reference scale。50 mL/km 接近完成
回合的中位 49.054 mL/km，覆盖本矩阵成功回合约 0.93--1.10 的量级；它必须以配置的
`provisional` / `smoke-only` 参数保存，而不是最终研究常量。

该 quantity 必须与 terminal/progress gates 一同审计：只比较 total energy 会奖励停车、少走或
提前失败；只看 `mL/km` 也不能抵消 out-of-road/低 progress 的任务失败。原生 MetaDrive
`step_energy`/`episode_energy` 为零，明确不能接入该 score。

**产物**：`outputs/energy_matrix/e-018-native-energy-proxy-comparison-run-1/`；12 个 job
各含 resolved config、Hydra overrides、runtime metadata、tracked diff、job/episode summary
及完整 trace。早期 launcher 根 `outputs/energy_matrix/e-018-native-energy-proxy-comparison/`
只保留排障用的不完整输出，不用于结论。

**结论边界**：本记录只支持 MetaDrive 0.4.3 当前 kinematic waypoint execution 下的
trace-recomputed fuel proxy 及其 reward smoke scale。它不支持真实车辆能耗、动力学能耗、
FASTSim、最终 reward 权重或跨 seed 的统计结论。
