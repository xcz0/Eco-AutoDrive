# 实验记录

本目录只登记实际运行过的实验：使用了什么代码、数据、参数，得到什么结果，以及结论能支持到哪里。大型产物、raw tensor、GIF 和原始日志保存在被 Git 忽略的 `outputs/` 或外部存储，不嵌入 Git 跟踪的文档。

本入口文档维护登记规则、共同资产、实验索引和记录模板；每次实验的完整记录按 ID 单独保存在 `records/`。

## 记录规则

每次用于研究结论的运行至少记录：

- 实验 ID、日期、目的和状态（诊断/本机验证/正式基线）；
- Git commit 和运行时未提交 diff；若当时未采集，明确写“未记录”；
- 上游源码 commit、依赖环境、设备；
- 数据集或程序化地图、场景 seed、噪声 seed；
- checkpoint 路径和参数量；
- resolved config、全部 Hydra overrides 和运行命令；
- 主要结果、失败状态、可支持的结论和不能支持的结论；
- `summary.json`、`trace.npz`、视频和外部归档位置。

实验目录应由 Hydra 独立创建，不覆盖旧结果。计划但未运行的工作由项目 GitHub Issues 跟踪，不在本目录伪装成实验记录。

## 共同资产

除条目另有说明，现有 Diffusion Planner 运行使用：

| 资产 | 固定值 |
| --- | --- |
| 上游源码 | `a3a621f0b724c5fa6447f7a2fbaf9e0387bd35df` |
| 模型 revision | `ae5baf1c57229c53f6309332df960ae27d35333f` |
| EMA | 276 tensors；6,042,628 parameters |
| 轨迹 | 80 点、10 Hz、8 s；每次执行前 5 点 |
| baseline sampler | 10 步、二阶 multistep DPM-Solver++ |

## 实验索引

| ID | 日期 | 类型 | 内容 | 结论状态 |
| --- | --- | --- | --- | --- |
| [E-000](records/e-000-stage0-alignment.md) | 原记录未注明 | 本机验证 | 阶段 0 上游数值对齐 | 有效，但只覆盖固定合成输入 |
| [E-001](records/e-001-pre-fix-no-traffic.md) | 2026-08-03 | 错误接口对照 | 修正前 `S`/`SC` 20 s 上限闭环 | 结果保留，原因解释已修正 |
| [E-002](records/e-002-speed-limit-diagnosis.md) | 2026-08-04 | 诊断 | 固定输入只改变 lane 限速 | 证明限速输入具有首要因果影响 |
| [E-003](records/e-003-corrected-short-closed-loop.md) | 2026-08-04 | 本机验证 | 修正限速、修正执行器后的 `S` 2 s 闭环 | 有效短程接口证据，不是正式基线 |
| [E-004](records/e-004-no-traffic-matrix.md) | 2026-08-04 | 本机长时验证 | `S`/`SC`、噪声 seeds `0..4`、20 s 上限矩阵 | 10 个回合无驾驶或接口失败 |
| [E-005](records/e-005-traffic-route-audit-failure.md) | 2026-08-04 | 错误接口对照 | 首次长时交通矩阵的曲线 lane 长度审计失败 | 标量校验已修正，不是性能结果 |
| [E-006](records/e-006-partial-traffic-matrix.md) | 2026-08-04 | 本机部分长时验证 | 2–5 km、两密度、paired seeds `0..2` 的 12 回合交通闭环 | 接口有效；矩阵按用户要求停止，不完整 |
| [E-007](records/e-007-five-step-ddim-acceptance.md) | 2026-08-11 | 阶段 1 正式验收 | DPM-10、标准高斯 DDIM-5 与 `0.5` 噪声 DDIM-5 的 30 回合配对闭环 | 5-step DDIM 阶段门槛通过 |
| [E-008](records/e-008-joblib-parallel-diagnosis.md) | 2026-08-11 | 并行可行性诊断 | Windows CPU 上的 Joblib 作业级并行、短回合剖析和串并行一致性 | 进程隔离可行；短矩阵无净加速，长交通矩阵尚未验收 |
| [E-009](records/e-009-stage2-guidance-smoke.md) | 2026-08-11 | 阶段 2 快速验收 | reference-centered orthogonal guidance 的数学、随机性、checkpoint 与 2 s 闭环 smoke | 快速门槛通过；60 回合正式矩阵延期，Issue #6 保持开启 |
| [E-010](records/e-010-parallel-evaluation-acceptance.md) | 2026-08-12 | 并行正式验收 | Windows CPU 6-job/300-step 矩阵与单 GPU 双进程一致性 | CPU 加速与 CPU/CUDA 串并行逐值一致性门槛通过 |

## 服务器训练与正式实验登记模板

```markdown
## E-NNN 标题

**日期 / 类型 / 目的**：
**代码**：Git commit；`git status --short`；上游 commit
**环境**：OS；Python；CUDA/GPU；关键依赖；容器/镜像（如使用）
**数据**：数据版本或地图配置；场景 seed；交通条件
**模型**：checkpoint 路径、参数量；sampler
**配置**：resolved config 路径；Hydra overrides；随机种子
**命令**：
**结果**：主要指标、失败状态和统计区间
**产物**：summary、trace、视频、外部归档
**结论边界**：支持什么；不支持什么
```
