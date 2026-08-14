# E-013 阶段 2 CUDA BF16 快速验收

[返回实验索引](../README.md)

**日期 / 类型 / 目的**：2026-08-14；阶段 2 快速验收；在当前 Artifact v3 和当前 HEAD 上，
验证 reference-centered orthogonal guidance 的 3-seed CUDA BF16 配对矩阵。

**代码**：`a1b1d38411e53be6d91d0254ec2f77761649bd25`；运行时存在本次未提交的
stage-2 summarizer 与测试修改，完整 tracked diff 保存于每个 job 的
`runtime_metadata.json` 与 `tracked_diff.patch`。上游 Diffusion Planner commit 为
`a3a621f0b724c5fa6447f7a2fbaf9e0387bd35df`；checkpoint 为
`checkpoints/DP-Origin/model.pth`，276 EMA tensors、6,042,628 parameters。

**环境**：Windows 10 build 26200；Python 3.10.20；PyTorch `2.12.1+cu126`；Lightning
2.6.5；MetaDrive/assets 0.4.3；单张 NVIDIA GeForce RTX 3050 Laptop GPU，`cuda:0`；
`runtime.accelerator=cuda`、`runtime.precision=bf16-mixed`。

**配置**：无交通 `S` / `SC`、地图 seed 0、diffusion noise seeds `0..2`、20 s 上限、
视频开启、标准高斯 DDIM-5（`ddim_stochasticity=0`）。先重跑三组 unguided DDIM-5
Artifact v3 anchor；随后运行 unguided、neutral、lateral `+1/-1`、longitudinal `+1/-1`
六组，每组 3 个 Hydra jobs、每个 job 两个场景，共 36 个 guidance-matrix 回合和 6 个 anchor
回合。旧 E-007 产物不迁移，也不用于 v3 reader。

**命令**：

```powershell
.venv\Scripts\python.exe scripts\evaluate.py --multirun sampler=ddim5 guidance=none `
  runtime.accelerator=cuda runtime.precision=bf16-mixed runtime.seed=0,1,2 `
  'hydra.sweep.dir=outputs/no_traffic/2026-08-14/stage2-cuda-bf16-anchor' `
  'hydra.sweep.subdir=${hydra.job.num}'

.venv\Scripts\python.exe scripts\summarize_stage2_matrix.py `
  outputs/no_traffic/2026-08-14/stage2-cuda-bf16-matrix `
  outputs/no_traffic/2026-08-14/stage2-cuda-bf16-anchor `
  --expected-seeds 0 1 2 --expected-accelerator cuda --expected-precision bf16-mixed
```

五个 active-guidance profiles 使用相同命令并显式传入对应的 lateral/longitudinal scale 和
各自独立的 `hydra.sweep.dir`。

**结果**：严格汇总器验证 36 个 matrix 回合、全部 GIF/summary/trace、CUDA/BF16 runtime
metadata、6 个 unguided 与当前 DDIM anchor 的 initial noise/prediction/execution 逐值匹配，
以及 6 个 neutral 与 unguided 的逐值匹配。neutral prediction 逐值等于其 reference。三组
lateral positive 的首周期横向偏移均为正，lateral negative 均为负；三组 longitudinal positive
的沿参考速度变化均为正，longitudinal negative 均为负。所有执行误差低于
`1e-3 m` / `1e-4 rad` 阈值。所有回合被保留：longitudinal negative 的两个直道回合以
`out_of_road` 终止，未被过滤或回退。

**产物**：
`outputs/no_traffic/2026-08-14/stage2-cuda-bf16-anchor/` 与
`outputs/no_traffic/2026-08-14/stage2-cuda-bf16-matrix/`；后者包含
`matrix_report.json`。

**结论边界**：支持阶段 2 的修订快速门槛：单张 RTX 3050、CUDA BF16、固定无交通 `S` / `SC`
与 seeds `0..2` 下，frozen reference guidance 的随机流、neutral 退化、方向趋势和运动学执行
接口均通过。它不等同于原 CPU FP32、5-seed、60 回合口径，也不支持交通、CUDA FP32、mixed-
precision 跨设备逐值一致性、Exploration Policy、reward、rollout、GAE、PPO 或论文数值 parity
的结论。
