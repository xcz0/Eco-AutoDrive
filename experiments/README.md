# 实验记录

本文件只登记实际运行过的实验：使用了什么代码、数据、参数，得到什么结果，以及结论能支持到
哪里。大型产物、raw tensor、GIF 和原始日志保存在被 Git 忽略的 `outputs/` 或外部存储，不嵌入
本文件。

## 记录规则

每次用于研究结论的运行至少记录：

- 实验 ID、日期、目的和状态（诊断/本机验证/正式基线）；
- Git commit 和运行时未提交 diff；若当时未采集，明确写“未记录”；
- 上游源码 commit、依赖环境、设备；
- 数据集或程序化地图、场景 seed、噪声 seed；
- checkpoint 路径和参数量；
- resolved config、全部 Hydra overrides 和运行命令；
- 主要结果、失败状态、可支持的结论和不能支持的结论；
- `summary.json`、`trace.npz`、视频和外部归档位置。

实验目录应由 Hydra 独立创建，不覆盖旧结果。计划但未运行的工作由项目 GitHub Issues 跟踪，
不在这里伪装成实验记录。

## 共同资产

除条目另有说明，现有 Diffusion Planner 运行使用：

| 资产 | 固定值 |
| --- | --- |
| 上游源码 | `a3a621f0b724c5fa6447f7a2fbaf9e0387bd35df` |
| 模型 revision | `ae5baf1c57229c53f6309332df960ae27d35333f` |
| EMA | 276 tensors；6,042,628 parameters |
| 轨迹 | 80 点、10 Hz、8 s；每次执行前 5 点 |
| baseline sampler | 10 步、二阶 multistep DPM-Solver++ |

## 实验索引

| ID | 日期 | 类型 | 内容 | 结论状态 |
| --- | --- | --- | --- | --- |
| E-000 | 原记录未注明 | 本机验证 | 阶段 0 上游数值对齐 | 有效，但只覆盖固定合成输入 |
| E-001 | 2026-08-03 | 错误接口对照 | 修正前 `S`/`SC` 20 s 上限闭环 | 结果保留，原因解释已修正 |
| E-002 | 2026-08-04 | 诊断 | 固定输入只改变 lane 限速 | 证明限速输入具有首要因果影响 |
| E-003 | 2026-08-04 | 本机验证 | 修正限速、修正执行器后的 `S` 2 s 闭环 | 有效短程接口证据，不是正式基线 |
| E-004 | 2026-08-04 | 本机长时验证 | `S`/`SC`、噪声 seeds `0..4`、20 s 上限矩阵 | 10 个回合无驾驶或接口失败 |
| E-005 | 2026-08-04 | 错误接口对照 | 首次长时交通矩阵的曲线 lane 长度审计失败 | 标量校验已修正，不是性能结果 |
| E-006 | 2026-08-04 | 本机部分长时验证 | 2–5 km、两密度、paired seeds `0..2` 的 12 回合交通闭环 | 接口有效；矩阵按用户要求停止，不完整 |
| E-007 | 2026-08-11 | 阶段 1 正式验收 | DPM-10、标准高斯 DDIM-5 与 `0.5` 噪声 DDIM-5 的 30 回合配对闭环 | 5-step DDIM 阶段门槛通过 |
| E-008 | 2026-08-11 | 并行可行性诊断 | Windows CPU 上的 Joblib 作业级并行、短回合剖析和串并行一致性 | 进程隔离可行；短矩阵无净加速，长交通矩阵尚未验收 |

## E-000 阶段 0 上游数值对齐

**目的**：确认本项目的 checkpoint-compatible 模型和官方 baseline sampler 在固定输入上与本地
上游快照数值一致。

**代码与环境**：原记录未保存运行 commit；Windows CPU；`ref/Diffusion-Planner` 使用共同资产中
的上游 commit。无 nuPlan 数据或地图。

**输入与参数**：固定合成 observation、完全相同的初始噪声、共同资产中的 checkpoint 和 10 步
DPM-Solver++。

**结果**：本实现与上游最终输出最大绝对误差 `3.1e-5`，既定对齐容差为
`atol=5e-5, rtol=0`；严格 EMA 加载、有限输出和同实现重复推理逐值一致通过。

**验证命令**：

```powershell
uv run pytest tests/smoke/test_stage0_baseline.py -m slow -s
```

**结论边界**：支持固定合成输入上的模型移植一致性；不支持官方 nuPlan 闭环指标复现，也没有完成
真实 nuPlan observation 的逐层对照。

## E-001 修正前无交通完整回合

**目的**：首次验证官方 EMA 在 MetaDrive `S`/`SC` 无交通运动学闭环中的端到端运行。

**代码与环境**：2026-08-03，Windows CPU；运行 commit 和未提交 diff 当时未写入产物，不能从
当前仓库状态补猜。该功能最初由 `5d416ef` 引入，但这不等于已证明运行时恰为该 commit。

**配置**：

- `model.device=cpu`；其余来自当时的 `configs/evaluation/no_traffic.yaml`；
- `S` seed 0 和 `SC` seed 0；噪声 seed 0；
- 无交通，20 s 上限，2 Hz 重规划、10 Hz 执行；
- 当时配置尚无 `programmatic_lane_speed_limit_kmh`。

**运行命令**：

```powershell
uv run python -m eco_planner.evaluate model.device=cpu
```

**结果**：

| 场景 | 周期/仿真时间 | 距离 | route completion | 终止 |
| --- | ---: | ---: | ---: | --- |
| `S`, seed 0 | 4 / 1.7 s | `27.769 m` | `0.2610` | `out_of_road` |
| `SC`, seed 0 | 4 / 1.7 s | `27.769 m` | `0.0953` | `out_of_road` |

两场景速度范围均约 `13.68–26.96 m/s`，无碰撞。模型输入把 PGMap 的
`1000 km/h` 哨兵编码成有效 `277.78 m/s`。

**本地产物**：`outputs/no_traffic/cpu-full-final/`，包含 resolved config、summary、两个场景的
trace 和 GIF。目录被 Git 忽略，跨机器不可依赖该路径存在。

**结论边界**：只证明旧接口会导致快速失败。原先“结果直接证明 nuPlan 到 MetaDrive 域偏移”的
解释无效；后续诊断确认首要根因是未设置限速哨兵被编码为有效限速。该产物不得覆盖，保留作错误接口对照。

## E-002 固定输入限速因果诊断

**目的**：在 checkpoint、地图、raw observation 的其他字段和初始噪声不变时，只改变 lane 限速
字段，判断原始失效是否由限速输入主导。

**代码与环境**：2026-08-04，Windows CPU；运行 commit、完整命令和独立产物目录在原记录中未
保存。该条目来自修正实施前的诊断记录，因此属于诊断证据，不是可独立发布的正式实验。

**对照与结果**：

| lane 限速输入 | 0.1 s 等效速度 | 0.5 s 局部 `(x, y)` | 8 s 局部 `(x, y)` |
| --- | ---: | ---: | ---: |
| `277.8 m/s + valid` | `26.7 m/s` | `(8.31, -1.316)` | `(144.09, -19.01)` |
| `0 + invalid` | `9.7 m/s` | `(5.01, -0.051)` | `(87.00, -0.96)` |
| `13.9 m/s + valid` | `10.9 m/s` | `(5.34, -0.042)` | `(87.67, -1.02)` |

用 `13.9 m/s` 和 E-001 的四轮噪声重放直道短闭环，四轮后最大 lane 横向偏差约 `0.053 m`，
没有 `out_of_road`。

**结论边界**：支持“错误限速是 E-001 的首要因果因素”。`13.9 m/s` 只是诊断近似，不能成为
Python 静默默认值；正式实验必须由配置指定真实的 km/h 条件。

## E-003 修正后直道 2 s 短闭环

**目的**：验证显式 PGMap 限速和修正后的 waypoint 生命周期能共同满足短程无交通接口契约。

**代码与环境**：2026-08-04，Windows CPU。运行发生在最终提交前，运行时未提交 diff 未单独
保存；对应实现随后提交为 `f578833060f0a715c4864064e4739f4f25773bf8`。这是本机接口证据；
当前基础代码构建阶段不要求追加 Docker/服务器验收。

**Hydra overrides**：

```yaml
- model.device=cpu
- env.horizon=20
- video.enabled=false
- scenarios=[{name:straight,map:S,seed:0}]
```

resolved config 的关键条件：`programmatic_lane_speed_limit_kmh=50.0`、场景 seed 0、噪声 seed 0、
无交通、20 个 0.1 s 子步。checkpoint 使用共同资产。

**运行命令**：

```powershell
uv run python -m eco_planner.evaluate model.device=cpu env.horizon=20 `
  video.enabled=false 'scenarios=[{name:straight,map:S,seed:0}]'
```

**结果**：

- 18 条 PGMap 哨兵 lane 被设置为 `50 km/h`，没有已有真实限速需要保留；
- 模型有效限速全部为 `13.888889 m/s`；
- 4 个规划周期、20 个子步、2.0 s，无碰撞、未出界，以 `max_step` 截断；
- 距离 `21.2079 m`，route completion `0.2160`；
- 速度最小/平均/最大为 `10.4492 / 10.6035 / 10.8656 m/s`；
- 位置执行误差最大/平均/最终为
  `0.000459833 / 0.000211897 / 0.000052932 m`；
- heading 执行误差最大为 `4.50203e-7 rad`。

**本地产物**：

- 无视频运行：`outputs/no_traffic/2026-08-04/11-23-18/`；
- 相同数值并生成 GIF：`outputs/no_traffic/2026-08-04/11-28-40/`。

两者包含 resolved config、summary 和完整 trace；后一目录另含 `closed_loop.gif`。

**结论边界**：支持限速接口和运动学执行器在单一直道、单一噪声 seed、2 s Windows CPU 条件下
已修正。不能支持 `SC` 驾驶表现、20 s 稳定性、多 seed 稳健性、能耗改进或服务器训练性能。

## E-004 无交通 20 s 上限多 seed 闭环矩阵

**日期 / 类型 / 目的**：2026-08-04，本机长时验证。验证修正后的官方 EMA 在 `S`/`SC`、
噪声 seeds `0..4` 下的闭环稳定性、限速语义、执行误差和产物完整性。`20 s` 是回合上限；
提前 `arrive_dest` 是正常成功。

**代码与运行时工作区**：

- Git HEAD：`54715966c461e3dce5026e14d1df115556c57cef`；运行前没有 tracked diff；
- `git status --short`：`?? .dockerignore`、`?? docker/`、`?? scripts/`、
  `?? third_party/nuplan-devkit/`、`?? uv.lock`；这些既有未跟踪文件未被清理或修改；
- Diffusion Planner 上游 commit：`a3a621f0b724c5fa6447f7a2fbaf9e0387bd35df`；
- `uv.lock` SHA-256：`4bb855d50a11468141d079c1f350b11bd86ea80904f021e186fc4c574507dd77`。

**环境与资产**：Windows、Python `3.10.20`、uv `0.11.29`、PyTorch `2.13.0+cpu`，无 CUDA。
MetaDrive 源码与 assets 版本均为 `0.4.3`；本地 editable 源码没有独立 Git 元数据，上游 commit
明确记为“未记录”。排除 `__pycache__`、`.pyc` 和 egg-info 后，本地 MetaDrive 1599 个文件的
确定性清单 SHA-256 为
`f1bb84643d90578565a41100b1efaa3efb6066895477cac7a80d4cb8b31440eb`。
checkpoint、args 和参数量使用“共同资产”表中的固定值。

**配置与命令**：固定地图 seed 0、无交通、`env.horizon=200`、2 Hz 重规划、10 Hz 运动学执行；
每个 Hydra 作业内 `S` 与 `SC` 重置同一噪声 seed，5 个作业由 BasicSweeper 串行执行。

```powershell
uv sync --all-groups
uv run python -m eco_planner.evaluate --multirun model.device=cpu `
  'model.seed=0,1,2,3,4' env.horizon=200 video.enabled=true `
  'hydra.sweep.dir=outputs/no_traffic/2026-08-04/16-10-00-long-20s-seeds-0-4'
```

**逐回合结果**：

| 地图 | 噪声 seed | 终止 | 时间 (s) | 距离 (m) | route completion | 平均速度 (m/s) | reward |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `S` | 0 | `arrive_dest` | 10.0 | 112.291 | 0.966709 | 11.22899 | 15.00030 |
| `S` | 1 | `arrive_dest` | 10.0 | 112.289 | 0.966691 | 11.22878 | 15.00041 |
| `S` | 2 | `arrive_dest` | 10.0 | 112.369 | 0.967350 | 11.23678 | 15.00382 |
| `S` | 3 | `arrive_dest` | 10.0 | 112.222 | 0.966141 | 11.22211 | 14.99727 |
| `S` | 4 | `arrive_dest` | 10.0 | 112.216 | 0.966092 | 11.22151 | 14.99707 |
| `SC` | 0 | `max_step` | 20.0 | 180.021 | 0.559883 | 9.00102 | 8.10092 |
| `SC` | 1 | `max_step` | 20.0 | 179.987 | 0.559832 | 8.99933 | 8.09939 |
| `SC` | 2 | `max_step` | 20.0 | 180.073 | 0.560047 | 9.00360 | 8.10324 |
| `SC` | 3 | `max_step` | 20.0 | 179.995 | 0.559903 | 8.99972 | 8.09975 |
| `SC` | 4 | `max_step` | 20.0 | 180.026 | 0.559946 | 9.00127 | 8.10114 |

10 个回合均无碰撞、无出界或异常终止。`S` 的 18 条未设置限速 lane 均替换为 `50 km/h`；
`SC` 的 18 条未设置 lane 替换为 `50 km/h`，并保留 12 条已有 `20 km/h` lane。

**分地图统计**：每项格式为“均值 / 中位数 / 均值的 95% percentile bootstrap 区间”；固定
统计 seed 0，每项独立执行 10,000 次有放回重采样。

| 地图 | 时间 (s) | 距离 (m) | route completion | 平均速度 (m/s) | reward |
| --- | --- | --- | --- | --- | --- |
| `S` | 10.000 / 10.000 / [10.000, 10.000] | 112.277 / 112.289 / [112.233, 112.324] | 0.966597 / 0.966691 / [0.966231, 0.966980] | 11.22763 / 11.22878 / [11.22321, 11.23229] | 14.99977 / 15.00030 / [14.99780, 15.00183] |
| `SC` | 20.000 / 20.000 / [20.000, 20.000] | 180.021 / 180.021 / [179.997, 180.048] | 0.559922 / 0.559903 / [0.559865, 0.559990] | 9.00099 / 9.00102 / [8.99982, 9.00236] | 8.10089 / 8.10092 / [8.09984, 8.10212] |

**接口与产物验证**：10 份 episode `summary.json`、`trace.npz` 和 `closed_loop.gif` 均存在且
非空；5 份 resolved config 与 Hydra overrides 均存在。逐数组检查未发现非有限值，规划周期、
执行子步和 `0.1 s` 时间轴全部一致。全矩阵最大位置误差为 `4.60045e-4 m`，最大 heading 误差
为 `1.78748e-6 rad`，分别低于 `1e-3 m` 和 `1e-4 rad` 门槛。固定统计和元数据另存于
`matrix_report.json`。

验证前 `53` 个非 GPU/simulator/slow 测试和 `9` 个显式 simulator 测试通过；`ruff check src tests`
与 `ruff format --check src tests` 通过。`ruff check .` 会扫描既有未跟踪的
`third_party/nuplan-devkit/` 上游快照并报告其格式问题，因此不能作为本次项目代码的通过证据，
且未修改该快照。

**本地产物**：`outputs/no_traffic/2026-08-04/16-10-00-long-20s-seeds-0-4/`。目录含 5 个 Hydra
作业，每个作业包含两个 episode 的 raw trace、summary 和 GIF；该目录被 Git 忽略。

**结论边界**：支持固定 `S`/`SC` 地图 seed 0、噪声 seeds `0..4`、Windows CPU 条件下，`S`
均能正常到达且 `SC` 均能稳定运行完整 20 s，无驾驶失败或接口失败。不能推出有交通、低层
steering/throttle、2–5 km 长路线、能耗改进、噪声 seeds `5..15`、CUDA 或服务器性能结论。

## E-005 首次长时交通矩阵路线审计失败

**日期 / 类型 / 目的**：2026-08-04，错误接口对照。计划运行两条 2–5 km 路线、交通密度
`0.05/0.10`、paired seeds `0..4`，验证通用交通观测和长时闭环。

**结果与根因**：纯直道 episode 可以运行，但每个作业进入混合路线时，评测边界把 MetaDrive
曲线 lane 的有限 `numpy.float32 length` 错误拒绝为非法类型。当前边界已接受 Python 和 NumPy
真实数值标量，并由 NumPy 标量回归及 `SC×20` 短闭环验证。矩阵在确认系统性失败后停止；已生成的直道结果属于诊断
产物，不并入 E-006 统计。

**产物**：`outputs/traffic/2026-08-04/formal-long-traffic-seeds-0-4/`。目录保留且不覆盖。

**结论边界**：只证明旧路线审计不兼容 MetaDrive 曲线数值类型，不能用于评价 planner 驾驶性能。

## E-006 seeds 0..2 长时交通部分矩阵

**日期 / 类型 / 目的**：2026-08-04，本机部分长时验证。验证官方 EMA 在 2–5 km 程序化长路线、
`Trigger` IDM 车辆交通、密度 `0.05/0.10` 下的观测接口、长时执行和驾驶结果。原计划 20 回合，
用户在 12 个完整回合后要求停止，因此本条不是 seeds `0..4` 正式基线。

**代码与环境**：运行 HEAD `5d6ff646f8c70198092c01369eba79763034010e`；运行时 tracked diff、
untracked 状态和完整 patch 保存在每个作业的 `runtime_metadata.json` / `tracked_diff.patch`。
Windows 10、Python `3.10.20`、PyTorch `2.13.0+cpu`、MetaDrive/assets `0.4.3`；`uv.lock`
SHA-256 为 `4bb855d50a11468141d079c1f350b11bd86ea80904f021e186fc4c574507dd77`。
模型和上游版本使用共同资产。

**配置**：`S×40` 路线实际约 `2.44–2.48 km`，`SC×20` 约 `3.38–3.90 km`；每回合先固定 ego
预热 20 个 0.1 s 子步，再最多正式评测 3000 子步；查询半径 100 m，2 Hz 重规划、10 Hz 执行，
地图 seed 与噪声 seed 配对。已完成 grid 为 seeds `0..2` × densities `0.05/0.10` × 两路线。

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

12 回合到达率为 0。所有回合都实际观察到 100 m 内交通，含交通规划帧比例为
`0.9420–1.0`；碰撞回合终止前最近车辆约 `5.62–7.18 m`，但该状态是 MetaDrive 在碰撞后采样的
对象中心距离，不能当成碰撞几何间隙。高密度混合路线的 3 个回合均在 `16.1–30.9 s` 出界，
表现与低密度组明显不同，但 `n=3` 且没有同长路线无交通对照，不能作因果归因。

阶段性 trace 审计还观察到，部分长时回合中的背景车辆在持续运行后出现异常行为，随后触发 ego
碰撞或出界；低密度组在进入该阶段前可持续行驶更长时间。该观察只支持把这组结果用于定位背景
交通稳定性和失败阶段，不能把最终终止直接归因为 planner 的基本驾驶能力，也不能作为稳定能耗
基线。

**接口与产物验证**：12 份 episode summary、trace 和 GIF 均存在且非空；6 份作业级 summary、
resolved config、Hydra overrides、运行元数据和 diff 齐全。所有数值数组有限，预热均为 20 帧，
邻车观测固定 `[32,21,11]`，时间轴一致。全体最大位置/heading 执行误差约为
`6.10e-5 m` / `2.59e-6 rad`，低于 `1e-3 m` / `1e-4 rad` 门槛。部分统计使用固定 seed 0、
10,000 次 bootstrap，详见 `partial_matrix_report.json`；由于每组只有 3 回合，区间只作描述。

**验证命令**：56 个非 GPU/simulator/slow 测试、11 个显式 simulator 测试、`ruff check .` 和
`ruff format --check .` 通过。当前安装为 CPU-only，因此没有运行 GPU marker。

**产物**：`outputs/traffic/2026-08-04/formal-long-traffic-seeds-0-4-v2/`。job `6` 是被终止的
不完整作业，没有 episode summary，已由部分汇总器排除。

**结论边界**：支持通用交通观测、真实 2 s 历史、长路线运动学接口和失败产物记录在已完成
12 回合中成立；也表明未经交通适配训练的官方 EMA 在该部分矩阵中没有完成路线。由于长时背景
交通异常会主导部分失败，该矩阵不能单独支持对 planner 基本驾驶能力或能耗性能的判断。不能推出
seeds `3..4`、完整 20 回合、真实 nuPlan 交通、低层控制、CUDA 或服务器性能结论。

## E-007 可切换 5-step DDIM 阶段验收

**日期 / 类型 / 目的**：2026-08-11，阶段 1 正式验收。验证显式 Hydra sampler 选择不会改变
官方 DPM-10 baseline，并验证本项目决定的标准高斯 5-step DDIM 在固定小型稳定场景上的数学、
随机性、产物和闭环边界。

**代码与环境**：运行 HEAD `1598ea2d683d2e64e3c415b2642363feb8e16d8a`；实现仍为未提交
tracked/untracked diff，逐作业保存在 `runtime_metadata.json` 和 `tracked_diff.patch`。Windows 10、
Python `3.10.20`、PyTorch `2.6.0+cpu`、Lightning `2.6.5`、MetaDrive/assets `0.4.3`；CPU
`32-true`。checkpoint、EMA tensor 数和参数量使用“共同资产”表中的固定值。

**Sampler 与配置**：三组都使用无交通 `S`/`SC`、地图 seed 0、噪声 seeds `0..4`、正式
评测上限 200 个 `0.1 s` 子步和视频输出。每组由 5 个 Hydra 作业组成，每作业重置同一个噪声
seed 分别运行两个场景。

- `dpm10`：官方 10-step DPM-Solver++，`0.5 * N(0,I)`；
- `ddim5`：标准高斯，`ddim_stochasticity=0`，在
  `t=[1.0,0.8,0.6,0.4,0.2]` 预测并转移到 `[0.8,0.6,0.4,0.2,0.0]`；
- `ddim5_project_noise`：相同 DDIM 时间表，初始尺度改为 `0.5`，只作隔离诊断。

DDIM 时间表是 ADR 0011 记录的本项目复现决定；PlannerRFT 公开材料没有给出作者 timestep
子序列。

**运行命令**：三次运行只替换 sampler 与输出目录；以下为标准高斯组，另外两组分别使用
`sampler=dpm10` 和 `sampler=ddim5_project_noise`。

```powershell
uv run python scripts/evaluate.py --multirun sampler=ddim5 `
  runtime.accelerator=cpu runtime.precision=32-true `
  'runtime.seed=0,1,2,3,4' video.enabled=true `
  'hydra.sweep.dir=outputs/no_traffic/2026-08-11/stage1-paired/ddim5'
```

**结果**：每项为 5 个噪声 seed 的均值；所有 `straight` 回合均 `arrive_dest`，所有
`gentle_curve` 回合均以 `max_step` 完成 20 s，无碰撞、出界或运行错误。

| Sampler | 场景 | 时间 (s) | 距离 (m) | route completion | 平均速度 (m/s) | reward |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| DPM-10 | `straight` | 10.000 | 112.277 | 0.966597 | 11.22763 | 14.99977 |
| DPM-10 | `gentle_curve` | 20.000 | 180.021 | 0.559922 | 9.00099 | 8.10089 |
| DDIM-5 标准高斯 | `straight` | 9.920 | 111.866 | 0.963176 | 11.27727 | 14.98039 |
| DDIM-5 标准高斯 | `gentle_curve` | 20.000 | 181.882 | 0.567147 | 9.09404 | 8.18464 |
| DDIM-5 `0.5` 隔离变体 | `straight` | 9.980 | 112.037 | 0.964586 | 11.22621 | 14.98853 |
| DDIM-5 `0.5` 隔离变体 | `gentle_curve` | 20.000 | 181.159 | 0.564943 | 9.05790 | 8.15211 |

三组每个 seed/场景保存的未缩放 `initial_noise` 逐值相同。30 个 trace 均通过完整 schema、有限性
和时间轴校验；最大位置误差 `4.600353932190726e-4 m`，最大 heading 误差
`2.1423861840119685e-6 rad`，低于 `1e-3 m` / `1e-4 rad` 门槛。15 份作业级和 30 份回合级
summary、resolved config、Hydra overrides、runtime metadata、tracked diff、trace 与 GIF 均存在。

**验证命令**：111 个非 GPU/simulator/slow 测试、DPM 与 DDIM 的 2 个 slow checkpoint 测试、
11 个显式 simulator 测试、`ruff check .` 和 `ruff format --check .` 通过。安装的是 CPU-only
PyTorch，因此未运行 GPU marker。

**产物**：`outputs/no_traffic/2026-08-11/stage1-paired/`；严格审计和逐回合/配对统计见
`matrix_report.json`。目录被 Git 忽略，不覆盖 E-004 或其他既有实验。

**结论边界**：支持可切换 sampler、DPM baseline 保持、标准高斯 DDIM-5 确定性重放，以及其在
固定 `S`/`SC`、无交通、CPU FP32、MetaDrive 运动学执行条件下没有明显驾驶退化。`0.5` 变体只
用于隔离初始尺度。不能推出 PlannerRFT 作者 timestep parity、nuPlan 指标、交通场景、CUDA/
mixed precision、guidance、PPO、能耗改进或低层控制可执行性。

## E-008 Windows CPU 作业级并行可行性诊断

**日期 / 类型 / 目的**：2026-08-11，方案实施前诊断。确认 MetaDrive 独立进程与 Hydra Joblib
作业并行能否共同运行，核对短回合串并行产物，并测量 Windows worker 启动成本。该运行不表示
Issue #5 或 ADR 0012 已实现。

**代码与环境**：运行 HEAD `a770b5bcf3488ff01cc6fe12f2699da20a973ea6`。首次剖析时 tracked
diff 只有 `pyproject.toml`；后续基准还包含当时未提交的 `docs/agents/issue-tracker.md`，每个作业的
实际 diff 保存在各自 `tracked_diff.patch`。Windows 10、16 个逻辑处理器、Python `3.10.20`、
PyTorch `2.6.0+cpu`、Lightning `2.6.5`、MetaDrive/assets `0.4.3`；临时解析
`hydra-joblib-launcher 1.2.0` 和 `joblib 1.5.3`。模型、checkpoint 与 DPM-10 使用共同资产。

**配置与命令**：全部使用 CPU `32-true`、关闭视频。Joblib 使用两个 `loky` worker；串行对照
使用 Hydra BasicLauncher。以下是四作业完整无交通对照的核心命令，串行命令移除 launcher
overrides 并替换输出目录：

```powershell
uv run --with hydra-joblib-launcher python scripts/evaluate.py --multirun `
  hydra/launcher=joblib hydra.launcher.n_jobs=2 hydra.launcher.backend=loky `
  runtime.accelerator=cpu runtime.precision=32-true 'runtime.seed=0,1,2,3' `
  video.enabled=false `
  'hydra.sweep.dir=C:/Users/xcz/AppData/Local/Temp/eco-autodrive-joblib-full-benchmark/parallel2'

uv run python scripts/evaluate.py --multirun `
  runtime.accelerator=cpu runtime.precision=32-true 'runtime.seed=0,1,2,3' `
  video.enabled=false `
  'hydra.sweep.dir=C:/Users/xcz/AppData/Local/Temp/eco-autodrive-joblib-full-benchmark/serial2'
```

另运行两个 2 s、固定地图 seed 0 的短作业，以及一个带 `cProfile` 的 2 s 直道回合。短作业
显式覆盖 `evaluation.evaluated_horizon_steps=20`、`env.horizon=20` 和单一 `straight` 场景。

**结果**：

- 2 s 直道进程总耗时 `9.444 s`，其中 `run_evaluation` 为 `1.797 s`、回合为 `1.303 s`；四次
  推理累计 `0.641 s`，地图适配 `0.070 s`，环境 trajectory step `0.061 s`。短进程主要成本是
  Python、Torch、MetaDrive 等模块导入。
- 两个 2 s 作业的 2-worker 总耗时为 `24.988 s`，串行为 `9.379 s`，串行/并行比为 `0.375`。
  两个 seed 的 job summary 与 `trace.npz` 全部数组逐值一致。
- 四个默认无交通作业（每作业依次运行 `straight` 与 `gentle_curve`，最多 200 个正式子步）的
  2-worker 总耗时为 `44.076 s`，串行为 `37.918 s`，串行/并行比为 `0.860`。去除一次性启动
  后，并行两波作业约 `20.2 s`，串行四作业约 `30.5 s`，说明进程执行有吞吐收益，但本矩阵仍
  被额外 worker 启动成本抵消。

**产物**：`%TEMP%/eco-autodrive-plan-profile/`、
`%TEMP%/eco-autodrive-joblib-benchmark/` 和
`%TEMP%/eco-autodrive-joblib-full-benchmark/`。这些是本机临时诊断目录，可能被系统清理；各 Hydra
作业在生成时包含 resolved config、runtime metadata、tracked diff、summary 和 episode trace。

**结论边界**：支持 Windows 上“每进程一个 MetaDrive 与一个 Fabric runtime”的作业隔离可以
运行，并支持已核对的两个短作业在 CPU FP32 下串并行逐值一致。不支持“并行已经提升当前默认
矩阵吞吐”的结论，也没有验证 traffic、长回合、DDIM、视频、CUDA 或多 GPU。只有 Issue #5 的
6-job、300-step traffic matrix 达到至少 20% 总墙钟改善并完成回归后，才能把并行入口描述为可用。

## 服务器训练与正式实验登记模板

```markdown
## E-NNN 标题

**日期 / 类型 / 目的**：
**代码**：Git commit；`git status --short`；上游 commit
**环境**：OS；Python；CUDA/GPU；关键依赖；容器/镜像（如使用）
**数据**：数据版本或地图配置；场景 seed；交通条件
**模型**：checkpoint 路径、参数量；sampler
**配置**：resolved config 路径；Hydra overrides；随机种子
**命令**：
**结果**：主要指标、失败状态和统计区间
**产物**：summary、trace、视频、外部归档
**结论边界**：支持什么；不支持什么
```
