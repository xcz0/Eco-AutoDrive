# E-014 Stage 6 小规模 closed-loop PPO smoke training

[返回实验索引](../README.md)

**日期 / 类型 / 目的**：2026-08-14；阶段 6 正式验收；验证 Stage 4 rollout、Stage 5
TorchRL updater、MetaDrive 10 Hz 闭环和 Training Artifact v1 能形成可重放的学习链路。

**代码**：运行起点 `b632bed96c49322d943c821bd2a55a56ab0ddc1b`；Stage-6 实现为未提交
tracked/untracked diff，各运行保存当时的 `tracked_diff.patch`。上游和 checkpoint 使用共同资产。

**环境**：Windows；Python 3.10.20；PyTorch 2.12.1+cu126；Lightning 2.6.5；MetaDrive
0.4.3；NVIDIA GeForce RTX 3050 Laptop GPU 4 GB；CUDA BF16 mixed precision；单进程单设备。

**配置**：标准高斯 DDIM-5、orthogonal policy guidance、无交通 `S`/`SC`、地图 seed 0。
每次运行含两个串行逻辑环境、每环境每轮 16 个 0.1 s transitions、四轮更新，共 128
transitions。训练 seeds `0/1` 各运行 replay `0/1`。reward 为 ADR 0019 固定的
`metadrive_builtin_v1`，不是论文 parity reward。

**命令**：四次调用 `scripts/train_stage6.py`，分别覆盖 `runtime.seed=0,1` 和
`training.replay_id=0,1`，输出到：

```text
outputs/training/stage6/2026-08-14/acceptance/seed-{0,1}-replay-{0,1}
```

随后运行：

```powershell
.venv\Scripts\python.exe scripts\summarize_stage6_training.py `
  outputs\training\stage6\2026-08-14\acceptance
```

**结果**：汇总器逐值比较同 seed 的 summary、全部 episode NPZ 和最终 checkpoint hash，四次
运行、512 transitions 全部通过。

| seed | 四轮 reward | 四轮平均速度范围 (m/s) | 梯度范数范围 | 最终 policy hash |
| --- | --- | ---: | ---: | --- |
| 0 | 1.522764, 1.533483, 1.522102, 1.542181 | 10.5702--10.7096 | 0.4392--1.5986 | `0ea994921042...` |
| 1 | 1.554177, 1.556889, 1.533923, 1.527997 | 10.6111--10.8117 | 0.5207--1.3582 | `861ec6b280bb...` |

两组 seed 的 policy hash 均相对初始化改变，且同 seed replay 的最终 hash 完全相同。冻结 planner
训练前后 hash 均为 `e1188b5ec562...`。固定 probe 的 Beta 参数变化超过 `1e-6`；训练后 guidance
均值在 `S`/`SC` 观测间的最大差异分别为约 `2.09e-6`（seed 0）和 `1.22e-5`（seed 1）。
所有轮次进度为正，停驶比例、碰撞、出界和 terminal overwrite 均为零；最大执行误差保持低于
既有 `1e-3 m` / `1e-4 rad` 门槛。reward 序列可重放但不要求单调。

**产物**：四个运行目录分别包含 resolved config、runtime metadata、tracked diff、初始/逐轮/最终
policy checkpoint、8 个完整 episode NPZ 和严格 summary；根目录包含 `stage6_report.json`。

**验证**：针对性 rollout/config 单元测试 15 项通过，Stage 5 PPO smoke 1 项通过，真实
MetaDrive rollout/simulator 测试 2 项通过；排除 `gpu`、`simulator` 和 `slow` marker 的快速
测试共 238 项通过。`ruff check .` 和 `ruff format --check .` 均通过。Windows 退出阶段仍会打印
TorchRL 延迟导入触发的 `0xc0000139` 诊断栈，但上述 pytest 进程退出码均为 0，断言均已执行并通过。

**结论边界**：支持在本机单 GPU、CUDA BF16、无交通短回合和 MetaDrive 原始 score 下，完整
closed-loop PPO 数据流、非零学习信号、观测相关分布变化、冻结边界和同设备确定性重放。不能支持
PlannerRFT reward/数值 parity、reward 改善、交通泛化、能耗改善、learned policy 优于对照、
低层动力学执行或论文规模训练。
