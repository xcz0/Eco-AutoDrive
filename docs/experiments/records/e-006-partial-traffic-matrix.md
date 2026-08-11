# E-006 seeds 0..2 长时交通部分矩阵

[返回实验索引](../README.md)

**日期 / 类型 / 目的**：2026-08-04，本机部分长时验证。验证官方 EMA 在 2–5 km 程序化长路线、`Trigger` IDM 车辆交通、密度 `0.05/0.10` 下的观测接口、长时执行和驾驶结果。原计划 20 回合，用户在 12 个完整回合后要求停止，因此本条不是 seeds `0..4` 正式基线。

**代码与环境**：运行 HEAD `5d6ff646f8c70198092c01369eba79763034010e`；运行时 tracked diff、untracked 状态和完整 patch 保存在每个作业的 `runtime_metadata.json` / `tracked_diff.patch`。Windows 10、Python `3.10.20`、PyTorch `2.13.0+cpu`、MetaDrive/assets `0.4.3`；`uv.lock` SHA-256 为 `4bb855d50a11468141d079c1f350b11bd86ea80904f021e186fc4c574507dd77`。
模型和上游版本使用共同资产。

**配置**：`S×40` 路线实际约 `2.44–2.48 km`，`SC×20` 约 `3.38–3.90 km`；每回合先固定 ego 预热 20 个 0.1 s 子步，再最多正式评测 3000 子步；查询半径 100 m，2 Hz 重规划、10 Hz 执行，地图 seed 与噪声 seed 配对。已完成 grid 为 seeds `0..2` × densities `0.05/0.10` × 两路线。

**运行命令**：

```powershell
uv run python -m eco_planner.evaluate --config-name evaluation/traffic --multirun `
  'seed=0,1,2,3,4' 'env.traffic_density=0.05,0.10' `
  'hydra.sweep.dir=outputs/traffic/2026-08-04/formal-long-traffic-seeds-0-4-v2'
```

运行在处理 seed 3 前后按用户要求终止；只有带 episode 和作业级 summary 的前 6 个作业纳入统计。

**逐组结果**：

| 路线 | 密度 | seeds | 终止 | 平均时间 (s) | 平均距离 (m) | 平均 route completion | 平均速度 (m/s) |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| `long_straight` | 0.05 | 0,1,2 | 3/3 `crash_vehicle` | 145.567 | 1369.299 | 0.556628 | 9.5260 |
| `long_straight` | 0.10 | 0,1,2 | 3/3 `crash_vehicle` | 170.633 | 1104.461 | 0.451855 | 6.9745 |
| `long_mixed` | 0.05 | 0,1,2 | 3/3 `crash_vehicle` | 167.167 | 1349.870 | 0.374790 | 8.1238 |
| `long_mixed` | 0.10 | 0,1,2 | 3/3 `out_of_road` | 25.500 | 220.336 | 0.062003 | 8.6787 |

12 回合到达率为 0。所有回合都实际观察到 100 m 内交通，含交通规划帧比例为 `0.9420–1.0`；碰撞回合终止前最近车辆约 `5.62–7.18 m`，但该状态是 MetaDrive 在碰撞后采样的对象中心距离，不能当成碰撞几何间隙。高密度混合路线的 3 个回合均在 `16.1–30.9 s` 出界，表现与低密度组明显不同，但 `n=3` 且没有同长路线无交通对照，不能作因果归因。

阶段性 trace 审计还观察到，部分长时回合中的背景车辆在持续运行后出现异常行为，随后触发 ego 碰撞或出界；低密度组在进入该阶段前可持续行驶更长时间。该观察只支持把这组结果用于定位背景交通稳定性和失败阶段，不能把最终终止直接归因为 planner 的基本驾驶能力，也不能作为稳定能耗基线。

**接口与产物验证**：12 份 episode summary、trace 和 GIF 均存在且非空；6 份作业级 summary、resolved config、Hydra overrides、运行元数据和 diff 齐全。所有数值数组有限，预热均为 20 帧，邻车观测固定 `[32,21,11]`，时间轴一致。全体最大位置/heading 执行误差约为 `6.10e-5 m` / `2.59e-6 rad`，低于 `1e-3 m` / `1e-4 rad` 门槛。部分统计使用固定 seed 0、10,000 次 bootstrap，详见 `partial_matrix_report.json`；由于每组只有 3 回合，区间只作描述。

**验证命令**：56 个非 GPU/simulator/slow 测试、11 个显式 simulator 测试、`ruff check .` 和 `ruff format --check .` 通过。当前安装为 CPU-only，因此没有运行 GPU marker。

**产物**：`outputs/traffic/2026-08-04/formal-long-traffic-seeds-0-4-v2/`。job `6` 是被终止的不完整作业，没有 episode summary，已由部分汇总器排除。

**结论边界**：支持通用交通观测、真实 2 s 历史、长路线运动学接口和失败产物记录在已完成 12 回合中成立；也表明未经交通适配训练的官方 EMA 在该部分矩阵中没有完成路线。由于长时背景交通异常会主导部分失败，该矩阵不能单独支持对 planner 基本驾驶能力或能耗性能的判断。不能推出 seeds `3..4`、完整 20 回合、真实 nuPlan 交通、低层控制、CUDA 或服务器性能结论。
