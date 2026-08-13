## E-011 planner-facing MetaDrive 环境性能重构

**日期 / 类型 / 目的**：2026-08-13；本机性能验收；在保持 planner 输入、轨迹执行、奖励、
终止和交通历史语义的条件下，降低长程有交通环境周期耗时。

**代码**：基线与修改均基于 `408f4b19c7747fd94a355230b73bebaefb6f2ddc`；测量时存在本次
未提交修改，涉及 `src/eco_planner/envs/`、性能脚本、测试、依赖声明和本文档。上游本地
MetaDrive 源码 commit 与主仓库相同；`third_party/metadrive/` 未修改。

**环境**：Windows 10 build 26200；Python 3.10.20；CPU headless；MetaDrive/assets 0.4.3；
NumPy 2.2.6；Shapely 2.1.2。

**数据**：程序化长混合地图 `SCSCSCSCSCSCSCSCSCSCSCSCSCSCSCSCSCSCSCSC`；map seed 0；
交通密度 0.05；`traffic_mode=trigger`；无随机交通；性能回路不执行模型推理，因此没有噪声 seed
或 checkpoint。

**配置**：物理步长 `0.02 s`、`decision_repeat=5`、轨迹 horizon 80、每 action 执行 5 点、
地图查询半径 `100 m`、20 步交通历史预热、10 个计时预热周期、100 个测量周期、重复 5 次。
每周期包含 observation adapter build、五个 0.1 s 仿真子步及 history append；不含模型推理、
视频和产物写入。resolved config 与 Hydra overrides 不适用；参数由性能脚本固定。

**命令**：

```powershell
.venv\Scripts\python.exe scripts\benchmark_envs.py
.venv\Scripts\python.exe scripts\benchmark_envs.py `
  --traffic-baseline-ms 41.697374 --no-traffic-baseline-ms 8.451062
```

**结果**：

| 模式 | 修改前 5 次周期耗时 (ms) | 修改后 5 次周期耗时 (ms) | 中位变化 |
| --- | --- | --- | --- |
| traffic | 42.774, 41.559, 41.725, 41.697, 41.447 | 20.486, 21.341, 20.902, 20.737, 20.626 | 41.697 → 20.737，降低 50.3% |
| no-traffic | 8.451, 8.368, 8.552, 8.461, 8.379 | 4.413, 4.042, 4.189, 4.031, 4.051 | 8.451 → 4.051，降低 52.1% |

修改中间态只消除默认 observation、地图和 Python 组装冗余时，traffic 中位耗时为
`38.332 ms`，未达到 20% 门槛。cProfile 显示 100 周期的 2500 次 Bullet `doPhysics` 调用累计
`2.916 s / 4.223 s`，因此最终将每个 0.1 s 对外子步的五个 `0.02 s` Bullet substep 合并到
一次原生调用；这是主要加速来源。

**产物**：本次只产生控制台基准和测试输出，没有 `summary.json`、`trace.npz` 或视频；原始日志
未单独归档。

**结论边界**：支持同一 Windows CPU/headless 环境中 planner-facing env 周期达到至少 20% 的
相对加速，并由 legacy 物理路径回归测试覆盖 planner observation、artifact participant 选择、
ego 子步状态和 route completion。背景交通浮点位置存在不超过 `1e-5` 的 planner tensor 尾差；
不能据此推断 GPU 推理、视频渲染、MetaDrive episode recording 或其他机器的绝对性能。
