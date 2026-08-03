# Eco-AutoDrive
通过强化学习优化扩散自动驾驶模型在长程运行中的能耗

## 官方权重无交通闭环

`evaluate.py` 使用固定的官方 EMA checkpoint，在 MetaDrive 直道 `S` 和缓弯 `SC` 上以
2 Hz 重规划、10 Hz 运动学轨迹执行运行无交通闭环。正式 CUDA 评测命令为：

```bash
uv run python -m eco_planner.evaluate
```

每个场景在 `outputs/no_traffic/` 下生成 `summary.json`、`trace.npz` 和带规划/执行轨迹叠加的
`closed_loop.gif`。CPU 仅用于单周期链路检查，例如：

```powershell
uv run python -m eco_planner.evaluate model.device=cpu env.horizon=5 `
  'scenarios=[{name:straight,map:S,seed:0}]'
```

该入口会拒绝任何背景车辆或静态交通物体，不包含邻车历史、低密度交通、FASTSim 或 DPPO。
详见 [阶段 1 无交通闭环](docs/阶段1_无交通闭环.md)。
