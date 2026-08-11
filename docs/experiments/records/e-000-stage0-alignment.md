# E-000 阶段 0 上游数值对齐

[返回实验索引](../README.md)

**目的**：确认本项目的 checkpoint-compatible 模型和官方 baseline sampler 在固定输入上与本地上游快照数值一致。

**代码与环境**：原记录未保存运行 commit；Windows CPU；`ref/Diffusion-Planner` 使用共同资产中的上游 commit。无 nuPlan 数据或地图。

**输入与参数**：固定合成 observation、完全相同的初始噪声、共同资产中的 checkpoint 和 10 步 DPM-Solver++。

**结果**：本实现与上游最终输出最大绝对误差 `3.1e-5`，既定对齐容差为`atol=5e-5, rtol=0`；严格 EMA 加载、有限输出和同实现重复推理逐值一致通过。

**验证命令**：

```powershell
uv run pytest tests/smoke/test_stage0_baseline.py -m slow -s
```

**结论边界**：支持固定合成输入上的模型移植一致性；不支持官方 nuPlan 闭环指标复现，也没有完成真实 nuPlan observation 的逐层对照。
