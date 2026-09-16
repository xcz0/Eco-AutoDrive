# 测试

测试按研究工作流组织，以回归、集成和功能验证为主，不按源码中的每个模块建立对应测试文件。

| 目录 | 主要覆盖 |
| --- | --- |
| `configuration/` | CLI、job composition、实验配置与配对约束 |
| `planning/` | observation、sampling、reference guidance |
| `simulation/` | 坐标与能耗数值回归、环境失败边界、真实 MetaDrive 闭环 |
| `training/` | reward、rollout、PPO、policy 与 trainer、checkpoint/MLflow、fixed-batch 与实验诊断 |
| `evaluation/` | 评测执行、policy checkpoint、trace 与报告产物回归 |
| `analysis/` | 从持久化产物重算统计与生成报告 |
| `benchmarking/` | rollout profiling 的计时边界 |

优先保留能够复现实际问题、检查完整功能结果或跨模块数据流的测试。即使是局部测试，只要保护 reward 公式、坐标/单位、随机流、episode boundary、GAE/bootstrap、配对统计或 undefined 状态，也应保留独立的预期值和断言。

仅检查内部转发、已完成的历史模块清退、第三方基础运算，或已被功能测试覆盖的重复测试可以删除。同一工作流的小文件优先合并；不同实验语义仍分别命名和断言。模拟数据与 mock 用于低成本验证软件契约，不作为科研结论的证据。

入口由 [justfile](../justfile) 定义：

```powershell
just test                         # 跨工作流 smoke 子集
just test-all-cpu                 # 完整 CPU 集合，排除 gpu/simulator/slow
just test-workflow training       # 某一目录的 CPU 测试
just test-target tests/training/test_credit_assignment.py
just test-sim                     # simulator，排除 gpu/slow
just test-gpu                     # gpu，排除 slow；包含同时标记 simulator 的测试
```

目录不决定运行资源，`smoke`、`gpu`、`simulator`、`slow` marker 才决定筛选。`simulation/` 中的纯数值和失败边界测试也会进入 CPU 集合；真实 checkpoint 的慢测试需显式选择相应 marker。
