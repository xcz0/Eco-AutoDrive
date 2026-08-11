# E-007 可切换 5-step DDIM 阶段验收

[返回实验索引](../README.md)

**日期 / 类型 / 目的**：2026-08-11，阶段 1 正式验收。验证显式 Hydra sampler 选择不会改变官方 DPM-10 baseline，并验证本项目决定的标准高斯 5-step DDIM 在固定小型稳定场景上的数学、随机性、产物和闭环边界。

**代码与环境**：运行 HEAD `1598ea2d683d2e64e3c415b2642363feb8e16d8a`；实现仍为未提交 tracked/untracked diff，逐作业保存在 `runtime_metadata.json` 和 `tracked_diff.patch`。Windows 10、Python `3.10.20`、PyTorch `2.6.0+cpu`、Lightning `2.6.5`、MetaDrive/assets `0.4.3`；CPU `32-true`。checkpoint、EMA tensor 数和参数量使用“共同资产”表中的固定值。

**Sampler 与配置**：三组都使用无交通 `S`/`SC`、地图 seed 0、噪声 seeds `0..4`、正式评测上限 200 个 `0.1 s` 子步和视频输出。每组由 5 个 Hydra 作业组成，每作业重置同一个噪声 seed 分别运行两个场景。

- `dpm10`：官方 10-step DPM-Solver++，`0.5 * N(0,I)`；
- `ddim5`：标准高斯，`ddim_stochasticity=0`，在 `t=[1.0,0.8,0.6,0.4,0.2]` 预测并转移到 `[0.8,0.6,0.4,0.2,0.0]`；
- `ddim5_project_noise`：相同 DDIM 时间表，初始尺度改为 `0.5`，只作隔离诊断。

DDIM 时间表是 ADR 0011 记录的本项目复现决定；PlannerRFT 公开材料没有给出作者 timestep 子序列。

**运行命令**：三次运行只替换 sampler 与输出目录；以下为标准高斯组，另外两组分别使用 `sampler=dpm10` 和 `sampler=ddim5_project_noise`。

```powershell
uv run python scripts/evaluate.py --multirun sampler=ddim5 `
  runtime.accelerator=cpu runtime.precision=32-true `
  'runtime.seed=0,1,2,3,4' video.enabled=true `
  'hydra.sweep.dir=outputs/no_traffic/2026-08-11/stage1-paired/ddim5'
```

**结果**：每项为 5 个噪声 seed 的均值；所有 `straight` 回合均 `arrive_dest`，所有 `gentle_curve` 回合均以 `max_step` 完成 20 s，无碰撞、出界或运行错误。

| Sampler | 场景 | 时间 (s) | 距离 (m) | route completion | 平均速度 (m/s) | reward |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| DPM-10 | `straight` | 10.000 | 112.277 | 0.966597 | 11.22763 | 14.99977 |
| DPM-10 | `gentle_curve` | 20.000 | 180.021 | 0.559922 | 9.00099 | 8.10089 |
| DDIM-5 标准高斯 | `straight` | 9.920 | 111.866 | 0.963176 | 11.27727 | 14.98039 |
| DDIM-5 标准高斯 | `gentle_curve` | 20.000 | 181.882 | 0.567147 | 9.09404 | 8.18464 |
| DDIM-5 `0.5` 隔离变体 | `straight` | 9.980 | 112.037 | 0.964586 | 11.22621 | 14.98853 |
| DDIM-5 `0.5` 隔离变体 | `gentle_curve` | 20.000 | 181.159 | 0.564943 | 9.05790 | 8.15211 |

三组每个 seed/场景保存的未缩放 `initial_noise` 逐值相同。30 个 trace 均通过完整 schema、有限性和时间轴校验；最大位置误差 `4.600353932190726e-4 m`，最大 heading 误差 `2.1423861840119685e-6 rad`，低于 `1e-3 m` / `1e-4 rad` 门槛。15 份作业级和 30 份回合级 summary、resolved config、Hydra overrides、runtime metadata、tracked diff、trace 与 GIF 均存在。

**验证命令**：111 个非 GPU/simulator/slow 测试、DPM 与 DDIM 的 2 个 slow checkpoint 测试、11 个显式 simulator 测试、`ruff check .` 和 `ruff format --check .` 通过。安装的是 CPU-only PyTorch，因此未运行 GPU marker。

**产物**：`outputs/no_traffic/2026-08-11/stage1-paired/`；严格审计和逐回合/配对统计见 `matrix_report.json`。目录被 Git 忽略，不覆盖 E-004 或其他既有实验。

**结论边界**：支持可切换 sampler、DPM baseline 保持、标准高斯 DDIM-5 确定性重放，以及其在固定 `S`/`SC`、无交通、CPU FP32、MetaDrive 运动学执行条件下没有明显驾驶退化。`0.5` 变体只用于隔离初始尺度。不能推出 PlannerRFT 作者 timestep parity、nuPlan 指标、交通场景、CUDA/mixed precision、guidance、PPO、能耗改进或低层控制可执行性。
