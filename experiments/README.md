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
- checkpoint 路径、SHA-256 和参数量；
- resolved config、全部 Hydra overrides 和运行命令；
- 主要结果、失败状态、可支持的结论和不能支持的结论；
- `summary.json`、`trace.npz`、视频和外部归档位置。

实验目录应由 Hydra 独立创建，不覆盖旧结果。计划但未运行的矩阵写在 `../docs/STATUS.md`，不在这里伪装
成实验记录。

## 共同资产

除条目另有说明，现有 Diffusion Planner 运行使用：

| 资产 | 固定值 |
| --- | --- |
| 上游源码 | `a3a621f0b724c5fa6447f7a2fbaf9e0387bd35df` |
| 模型 revision | `ae5baf1c57229c53f6309332df960ae27d35333f` |
| `args.json` SHA-256 | `7e62b89a50953f133d55484777e54490f7f24e58feec1efcf696bcc7b91bdf10` |
| `model.pth` SHA-256 | `7a441df91ebe1c912d8262010c40486da24f425f757e2b4228072e251ab67d45` |
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
解释无效；首要根因已由 C-001 修正。该产物不得覆盖，保留作错误接口对照。

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

## 服务器训练与正式实验登记模板

```markdown
## E-NNN 标题

**日期 / 类型 / 目的**：
**代码**：Git commit；`git status --short`；上游 commit
**环境**：OS；Python；CUDA/GPU；关键依赖；容器/镜像（如使用）
**数据**：数据版本或地图配置；场景 seed；交通条件
**模型**：checkpoint 路径、SHA-256、参数量；sampler
**配置**：resolved config 路径；Hydra overrides；随机种子
**命令**：
**结果**：主要指标、失败状态和统计区间
**产物**：summary、trace、视频、外部归档
**结论边界**：支持什么；不支持什么
```
