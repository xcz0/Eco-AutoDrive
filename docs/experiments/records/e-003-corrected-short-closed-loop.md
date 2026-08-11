# E-003 修正后直道 2 s 短闭环

[返回实验索引](../README.md)

**目的**：验证显式 PGMap 限速和修正后的 waypoint 生命周期能共同满足短程无交通接口契约。

**代码与环境**：2026-08-04，Windows CPU。运行发生在最终提交前，运行时未提交 diff 未单独保存；对应实现随后提交为 `f578833060f0a715c4864064e4739f4f25773bf8`。这是本机接口证据；当前基础代码构建阶段不要求追加 Docker/服务器验收。

**Hydra overrides**：

```yaml
- model.device=cpu
- env.horizon=20
- video.enabled=false
- scenarios=[{name:straight,map:S,seed:0}]
```

resolved config 的关键条件：`programmatic_lane_speed_limit_kmh=50.0`、场景 seed 0、噪声 seed 0、无交通、20 个 0.1 s 子步。checkpoint 使用共同资产。

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
- 位置执行误差最大/平均/最终为 `0.000459833 / 0.000211897 / 0.000052932 m`；
- heading 执行误差最大为 `4.50203e-7 rad`。

**本地产物**：

- 无视频运行：`outputs/no_traffic/2026-08-04/11-23-18/`；
- 相同数值并生成 GIF：`outputs/no_traffic/2026-08-04/11-28-40/`。

两者包含 resolved config、summary 和完整 trace；后一目录另含 `closed_loop.gif`。

**结论边界**：支持限速接口和运动学执行器在单一直道、单一噪声 seed、2 s Windows CPU 条件下已修正。不能支持 `SC` 驾驶表现、20 s 稳定性、多 seed 稳健性、能耗改进或服务器训练性能。
