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
| [E-011](records/e-011-env-performance-refactor.md) | 2026-08-13 | 本机性能验收 | planner-facing MetaDrive 环境周期重构与等价性验证 | 长程交通周期中位耗时降低 50.3%，门槛通过 |
| [E-012](records/e-012-evaluation-long-horizon-performance-refactor.md) | 2026-08-14 | 本机性能验收 | CUDA BF16 长程 traffic/full 端到端评测重构 | 进程总墙钟中位降低 4.85%，evaluation artifact contract 严格校验通过 |
| [E-013](records/e-013-stage2-cuda-bf16-fast-acceptance.md) | 2026-08-14 | 阶段 2 快速验收 | CUDA BF16、3-seed 的 reference-centered guidance 配对矩阵 | 修订门槛通过；结论限于单 GPU、BF16 与 3 seeds |
| [E-014](records/e-014-stage6-closed-loop-smoke-training.md) | 2026-08-14 | 阶段 6 正式验收 | CUDA BF16、双 seed 重放的 `2 x 16 x 4` closed-loop PPO smoke | 阶段门槛通过；仅支持 MetaDrive smoke 学习链路 |
| [E-015](records/e-015-benchmark-scaling-baseline.md) | 2026-08-21 | 本机性能基线 | Reference planner、traffic/no-traffic vector env、serial/vector rollout scaling | 本机建议有效；远程机与 planner 饱和点未完成 |
| [E-016](records/e-016-machine-benchmark-rx-a4000.md) | 2026-08-21 | 本机性能基线（远程训练机） | 在 RTX A4000 / Xeon w5-2455X 上复现 planner、vector env、rollout scaling | 本机建议有效（B=8–16；workers 2–4）；planner 饱和点、三式 evaluation 墙钟对照已完成 |
| [E-017](records/e-017-energy-guidance-baseline.md) | 2026-08-24 | 正式固定 seed 基线 | 6 场景、4 guidance 的 CUDA BF16 能耗矩阵与重复运行 | 48 回合逐值可重复；Issue #1 基线完成 |
| [E-018](records/e-018-energy-benchmark-bringup.md) | 2026-08-24 | 本机验证与诊断 | Issue #1 fuel-proxy、分段限速与 DDIM5 guidance bring-up | 历史诊断有效；完整矩阵由 E-017 补全 |
| [E-019](records/e-019-metadrive-native-energy-proxy-comparison.md) | 2026-08-24 | 正式固定 seed 指标审计 | MetaDrive 原生 energy 与执行 trace proxy 对照 | 原生值在当前 kinematic execution 下恒为零；#59 使用 trace proxy 的 provisional scale |
| [E-020](records/e-020-persistent-vector-rollout-rtx3050.md) | 2026-08-24 | 本机验证 | Issue #63 持久 vector rollout worker pool | 稳态 vector 吞吐超过同次 serial；首轮启动成本单独记录 |
| [E-021](records/e-021-strict-job-level-evaluation-benchmark.md) | 2026-08-24 | 本机正式基准 | Issue #66 的 RTX 3050 strict serial/job-level traffic matrix | 2 × 8 profile 的 job-level 比同配置 serial 快 35.3%；仅限此机器与矩阵 |
| [E-022](records/e-022-persistent-vector-evaluation-benchmark.md) | 2026-08-24 | 本机正式基准 | Issue #65 的 chunked/persistent/serial vector evaluation 对照 | persistent pool 比 chunked 快 15.2%，并保留场景完成语义；仅限此机器与矩阵 |
| [E-023](records/e-023-torchrl-parallel-env-correctness.md) | 2026-08-25 | 本机迁移验证 | 自定义 MetaDrive IPC 到 TorchRL ParallelEnv 的单次真实差分 | 正确性通过；未形成性能结论 |
| [E-024](records/e-024-plannerrft-energy-reward-smoke.md) | 2026-08-26 | 本机实现验收 | `plannerrft_energy_v1` 的 32-transition 单次真实 PPO update | 链路、审计与有限值通过；不构成 A/B、parity 或节能结论 |
| [E-025](records/e-025-issue59-stage-a-ppo-reward-ab-mechanical-gate.md) | 2026-08-27 | 本机机械门控 | Issue #59 阶段 A builtin/energy 匹配 PPO A/B（4 updates × 2 profiles） | 9 项机械检查全部通过，无 reward hacking；阶段 A 验收完成 |
| [E-026](records/e-026-issue74-corrected-rollout-profiler.md) | 2026-08-28 | 本机正式基准 | Issue #74 的 PPO scheduler 与 CUDA 异步计时口径修复 | B=2/B=4 真实 4-epoch profiler 完成；旧 planner/bootstrap 分项失效，执行路径优化未完成 |
| [E-026](records/e-026-issue59-stage-b-ppo-reward-ab-short-trend.md) | 2026-08-28 | 本机短趋势验证 | Issue #59 阶段 B builtin/energy 匹配 PPO A/B（20 updates × 3 seeds × 2 profiles） | 27 项机械检查全部通过，无 reward hacking；两 profile 间无显著差异；阶段 B 验收完成；退化归因 reset bug（见 E-028 修正） |
| [E-027](records/e-027-issue76-stage1-p0-failure-and-p1-conservative-baseline.md) | 2026-08-28 | 本机基线 | Issue #76 阶段 1 P0 失败基线保留与 P1 人工保守基线 | P1 20 updates 全程稳定；"update 强度过大"归因已被 reset bug 修正推翻（见 E-028 修正） |
| [E-027](records/e-027-issue74-rollout-hotpath-optimization.md) | 2026-08-28 | 本机优化验证 | Issue #74 的局部 DiT 编译、同步与 DDIM allocation 优化 | 编译因 Windows 无可用 Triton 未过门槛，保持 eager 默认；两项等价微优化落地，不主张全局吞吐改善 |
| [E-028](records/e-028-runtime-ownership-refactor.md) | 2026-09-01 | 本机实现验证与诊断复测 | runtime ownership、evaluation topology、rollout/worker 分层与 CUDA audit stream 候选 | 正确性验证通过；性能受机器状态影响，不作加速结论；最终不复用 audit stream |
| [E-028](records/e-028-issue76-ppo-stability-search.md) | 2026-09-01 | 远程训练机正式搜索 | Issue #76 reset 修复后的 P0/P1 复验与 Optuna Stage A/B/C 分层稳定超参数搜索 | P0/P1 修复后均稳定（E-026 式退化归因 reset bug）；config-0001（batch=128、epochs=1、lr=1.63e-5）为唯一 3 seeds × 100 updates 稳定候选；主要失败模式为 epochs=3 × 高 lr 的 Beta 边界塌缩 |

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
