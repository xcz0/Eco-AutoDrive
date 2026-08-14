# PlannerRFT PPO-only 契约与待决问题

## 目的

下表把会改变数据流、训练语义或结论口径的问题列为 gate。gate 关闭前可以做只读研究或隔离的
数学原型，但不能把对应阶段转成实现 Issue。决定若改变长期系统边界，应写 ADR；实现后再同步
system contract。

## Gate 清单

| ID | 问题 | 当前证据/约束 | 候选选择 | 关闭 gate 所需产物 |
| --- | --- | --- | --- | --- |
| G-01 | 一个 PPO MDP step 多长？ | **已关闭**：Stage 4 rollout 每次执行 `1 x 0.1 s` 并重规划；既有 evaluation 继续执行 `5 x 0.1 s` | 见 ADR 0017 | Issue #18；single-substep simulator、时间轴和 evaluation 隔离测试 |
| G-02 | reference trajectory 如何生成？ | **已关闭**：同一冻结 checkpoint、每周期一次 scene encoding；标准高斯 DDIM-5 reference 每周期刷新并与 guided pass 共享 initial noise 和 transition draws | 见 ADR 0013 | Issue #6；随机流重放、单 encoder、冻结范围和 reference trace 测试 |
| G-03 | orthogonal guidance 的离散定义和 neutral action 是什么？ | **已关闭**：heading 定义切/左法向，当前点加未来点作 10 Hz 差分；零速合法；centered gradient delta 使 `(0,0)` 精确 neutral | 见 ADR 0013；这是项目复现决定 | Issue #6；手算方向/量纲、零速/退化边界和 rigid-transform 测试 |
| G-04 | policy 的完整 observation 是什么？ | **已关闭**：一次冻结 scene encoding 及 padding mask、一次冻结 route/navigation encoding 及 validity mask、ego-local physical reference `[B,80,4]` | 见 ADR 0016 | Issue #16；shape/dtype/device/mask、单次编码和冻结反向测试 |
| G-05 | Beta action 如何映射和记概率？ | **已关闭**：保存严格位于 `(0,1)^2` 的 base action `u`；guidance 为 `2u-1`；joint transformed log-prob 减 `2 log 2`，entropy 加 `2 log 2` | 见 ADR 0016；这是项目复现决定 | Issue #16；sample/rsample/mean、support、log-prob、entropy 和 old/new 重算测试 |
| G-06 | 多候选 `K` 如何进入 PPO 概率？ | **已关闭**：阶段 3--5 固定单候选 `K=1`，不定义额外候选选择概率 | 见 ADR 0016 | Issue #16；单 action 概率接口；未来若改变 `K` 必须重开 gate |
| G-07 | PPO reward 如何在 MetaDrive 定义？ | 当前只有 MetaDrive 内置 reward/substep；系统未定义优化 reward | 论文 parity reward；加入能耗的项目扩展 reward | 每组件数据源、单位、频率、范围、terminal gate、失败语义和反停驶测试 |
| G-08 | terminal 与 truncation 如何 bootstrap？ | **已关闭**：`bootstrap_mask = not terminated`；纯 time-limit truncation 与 rollout tail 保存 final-state old value，GAE 递归仍在 done 边界停止 | 见 ADR 0017/0018 | Issue #18/#19；terminal/truncation/tail buffer 与 GAE 手算测试 |
| G-09 | 训练 runtime 如何扩展？ | 当前 ADR 0010 限定单进程单设备 evaluation | 独立同步向量环境；多进程收集；其他训练编排 | ADR；seed 派生、设备、异常传播、资源释放和产物聚合测试 |
| G-10 | parity 与项目扩展如何分开？ | 本项目目标包含道路预瞄与能耗，PlannerRFT 原目标不同 | 先 parity 后扩展；直接项目 reward | 两套命名独立的 config/实验矩阵和结论边界 |
| G-11 | DDIM 初始分布和五步时间表是什么？ | **已关闭**：默认论文文字 profile 使用标准高斯；本项目选择 `t=[1.0,0.8,0.6,0.4,0.2] -> [0.8,0.6,0.4,0.2,0.0]`；`0.5` 倍噪声仅作隔离变体 | 见 ADR 0011；时间表不得描述为作者事实 | Issue #4；严格 sampler 配置、数学/随机性测试和配对闭环证据 |
| G-12 | guidance 梯度在哪个空间、作用于谁？ | **已关闭**：physical ego energy 经 normalizer/冻结 DiT 对 normalized noisy joint sample 求导；transition 后单位系数更新；屏蔽 current 与 neighbor，只应用 ego future 梯度 | 见 ADR 0013；记录被屏蔽 neighbor gradient | Issue #6；sample gradient、参数冻结、mask、有限性和逐步 diagnostics 测试 |

## 必须保持的不变量

无论 gate 最终如何关闭，下列当前约束都不能被研究实现静默绕过：

- checkpoint、normalization、shape、dtype、device、有限性和单位在系统边界显式验证；
- baseline DPM 路径保持可复现，reference/planner/encoder 的冻结范围可审计；
- 地图 seed、diffusion noise seed 和新增的 policy action seed 分开记录；
- 原始预测与实际执行 trace 分开保存，失败回合不删除；
- 不通过平滑、裁剪、中心线投影、回退控制器、零轨迹或坏样本跳过掩盖失效；
- 运动学执行结论不外推为低层 steering/throttle/brake 可执行性；
- MetaDrive 代理能耗与 FASTSim 等精细模型不混用。

地图 seed、噪声 seed 与 policy action seed 的规范定义见 `CONTEXT.md`；三条随机流不得复用或
通过全局 RNG 隐式耦合。

## 配置分层建议

关闭各 gate 后，建议使用独立、必需字段齐全的 Hydra 配置组，避免把实验参数硬编码进 Python：

```text
configs/
  sampler/{dpm10,ddim5}.yaml
  guidance/plannerrft.yaml
  policy/exploration_beta.yaml
  reward/{plannerrft_parity,eco_extension}.yaml
  rollout/{smoke,parity}.yaml
  train/ppo_exploration.yaml
  experiment/plannerrft_ppo_*.yaml
```

这是候选目录，不表示这些配置已存在。当前 `configs/experiment/baseline.yaml` 引用了不存在的
`train/dppo`，`configs/reward/energy.yaml` 也未接入现有 evaluation runner；两者不能作为 RL 已实现
或 reward 已定义的证据。实现第一阶段时应在对应 Issue 中删除或修正这些遗留配置，而不是为其
添加假入口。

## Issue 拆分建议

gate 关闭后，可按以下依赖链建立独立 Issues；每个 Issue 只完成一个逻辑变更：

```text
baseline revalidation
  -> DDIM sampler
  -> fixed orthogonal guidance
  -> Exploration Policy distribution
  -> rollout contract
  -> reward adapter
  -> GAE/PPO math
  -> smoke training
  -> paired ablation
  -> scale-out runtime
```

Issue 之间通过验收产物依赖，不通过复制状态到 Markdown 跟踪进度。
