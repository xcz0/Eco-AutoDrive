# E-001 修正前无交通完整回合

[返回实验索引](../README.md)

**目的**：首次验证官方 EMA 在 MetaDrive `S`/`SC` 无交通运动学闭环中的端到端运行。

**代码与环境**：2026-08-03，Windows CPU；运行 commit 和未提交 diff 当时未写入产物，不能从当前仓库状态补猜。该功能最初由 `5d416ef` 引入，但这不等于已证明运行时恰为该 commit。

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

两场景速度范围均约 `13.68–26.96 m/s`，无碰撞。模型输入把 PGMap 的 `1000 km/h` 哨兵编码成有效 `277.78 m/s`。

**本地产物**：`outputs/no_traffic/cpu-full-final/`，包含 resolved config、summary、两个场景的 trace 和 GIF。目录被 Git 忽略，跨机器不可依赖该路径存在。

**结论边界**：只证明旧接口会导致快速失败。原先“结果直接证明 nuPlan 到 MetaDrive 域偏移”的解释无效；后续诊断确认首要根因是未设置限速哨兵被编码为有效限速。该产物不得覆盖，保留作错误接口对照。
