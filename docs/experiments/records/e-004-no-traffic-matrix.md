# E-004 无交通 20 s 上限多 seed 闭环矩阵

[返回实验索引](../README.md)

**日期 / 类型 / 目的**：2026-08-04，本机长时验证。验证修正后的官方 EMA 在 `S`/`SC`、噪声 seeds `0..4` 下的闭环稳定性、限速语义、执行误差和产物完整性。`20 s` 是回合上限；提前 `arrive_dest` 是正常成功。

**代码与运行时工作区**：

- Git HEAD：`54715966c461e3dce5026e14d1df115556c57cef`；运行前没有 tracked diff；
- `git status --short`：`?? .dockerignore`、`?? docker/`、`?? scripts/`、`?? third_party/nuplan-devkit/`、`?? uv.lock`；这些既有未跟踪文件未被清理或修改；
- Diffusion Planner 上游 commit：`a3a621f0b724c5fa6447f7a2fbaf9e0387bd35df`；
- `uv.lock` SHA-256：`4bb855d50a11468141d079c1f350b11bd86ea80904f021e186fc4c574507dd77`。

**环境与资产**：Windows、Python `3.10.20`、uv `0.11.29`、PyTorch `2.13.0+cpu`，无 CUDA。MetaDrive 源码与 assets 版本均为 `0.4.3`；本地 editable 源码没有独立 Git 元数据，上游 commit 明确记为“未记录”。排除 `__pycache__`、`.pyc` 和 egg-info 后，本地 MetaDrive 1599 个文件的确定性清单 SHA-256 为 `f1bb84643d90578565a41100b1efaa3efb6066895477cac7a80d4cb8b31440eb`。checkpoint、args 和参数量使用“共同资产”表中的固定值。

**配置与命令**：固定地图 seed 0、无交通、`env.horizon=200`、2 Hz 重规划、10 Hz 运动学执行；每个 Hydra 作业内 `S` 与 `SC` 重置同一噪声 seed，5 个作业由 BasicSweeper 串行执行。

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

10 个回合均无碰撞、无出界或异常终止。`S` 的 18 条未设置限速 lane 均替换为 `50 km/h`；`SC` 的 18 条未设置 lane 替换为 `50 km/h`，并保留 12 条已有 `20 km/h` lane。

**分地图统计**：每项格式为“均值 / 中位数 / 均值的 95% percentile bootstrap 区间”；固定统计 seed 0，每项独立执行 10,000 次有放回重采样。

| 地图 | 时间 (s) | 距离 (m) | route completion | 平均速度 (m/s) | reward |
| --- | --- | --- | --- | --- | --- |
| `S` | 10.000 / 10.000 / [10.000, 10.000] | 112.277 / 112.289 / [112.233, 112.324] | 0.966597 / 0.966691 / [0.966231, 0.966980] | 11.22763 / 11.22878 / [11.22321, 11.23229] | 14.99977 / 15.00030 / [14.99780, 15.00183] |
| `SC` | 20.000 / 20.000 / [20.000, 20.000] | 180.021 / 180.021 / [179.997, 180.048] | 0.559922 / 0.559903 / [0.559865, 0.559990] | 9.00099 / 9.00102 / [8.99982, 9.00236] | 8.10089 / 8.10092 / [8.09984, 8.10212] |

**接口与产物验证**：10 份 episode `summary.json`、`trace.npz` 和 `closed_loop.gif` 均存在且非空；5 份 resolved config 与 Hydra overrides 均存在。逐数组检查未发现非有限值，规划周期、执行子步和 `0.1 s` 时间轴全部一致。全矩阵最大位置误差为 `4.60045e-4 m`，最大 heading 误差为 `1.78748e-6 rad`，分别低于 `1e-3 m` 和 `1e-4 rad` 门槛。固定统计和元数据另存于 `matrix_report.json`。

验证前 `53` 个非 GPU/simulator/slow 测试和 `9` 个显式 simulator 测试通过；`ruff check src tests` 与 `ruff format --check src tests` 通过。`ruff check .` 会扫描既有未跟踪的 `third_party/nuplan-devkit/` 上游快照并报告其格式问题，因此不能作为本次项目代码的通过证据，且未修改该快照。

**本地产物**：`outputs/no_traffic/2026-08-04/16-10-00-long-20s-seeds-0-4/`。目录含 5 个 Hydra 作业，每个作业包含两个 episode 的 raw trace、summary 和 GIF；该目录被 Git 忽略。

**结论边界**：支持固定 `S`/`SC` 地图 seed 0、噪声 seeds `0..4`、Windows CPU 条件下，`S` 均能正常到达且 `SC` 均能稳定运行完整 20 s，无驾驶失败或接口失败。不能推出有交通、低层 steering/throttle、2–5 km 长路线、能耗改进、噪声 seeds `5..15`、CUDA 或服务器性能结论。
