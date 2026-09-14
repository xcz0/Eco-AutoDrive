# 0038 — 按当前科研工作流整合实验架构

- 状态：接受
- 日期：2026-09-14
- 来源：Issue #97
- 部分取代：ADR 0034、0035、0036、0037

## 背景

历史 study 将固定批次采集、reward 校准、actor backward、人工回合执行与实验设计放在同层。
多个工作流通过历史 reference-dir 和冻结数值串联，旧搜索阶段及专属报告继续约束当前研究入口。
这使底层机制依赖研究命名，也使新 batch 无法独立执行已有诊断。

## 决定

实验层仅编排 comparison、reward、credit、guidance、training 五类科研职责。
共享 train/held-out 定义及配置组合集中到小型 experiments.protocol，不建立通用调度框架。

固定批次归 rl.rollout；reward 提取、变换及本 batch 校准归 rl.reward；策略恢复、credit/advantage
变体、backward-only 梯度和参数/post-update KL 测量归 rl.optimization；人工干预的 reset/step、
固定噪声、终止与部分证据归 evaluation。底层不导入 experiments。

comparison 配置声明 arms/reward profiles/contrasts，共用三臂与 calibrated 双臂执行和比较机制。
reward run 只诊断奖励数据，credit run 统一编排命名的 objective/advantage/credit 轴。
校准和 energy-band 每次由源 batch 按显式配置重算，保存实际尺度，删除历史 reference-dir 与冻结
expected 值守卫；其覆盖的数值一致性由独立测试验证。

training 保留 optimizer 笛卡尔积、预算、阈值及最低 learning rate/epochs/gradient norm 选择规则。
无通过项明确报告无候选，失败保留原始证据。训练诊断先持久化 checkpoint/Torch 测量，确定性与
随机策略评测使用显式输入及 checkpoint/随机条件核验，不依赖 positive-control 目录。

CLI 为 just exp 的薄入口，由现有 Python CLI 延迟分派。删除旧 study CLI/import、stage A/B/C、
top-N 晋升、pruning、stability 数据库分析与独立 reproducibility 目录协议，不保留兼容层。
依赖清理不在本次范围内。

analysis 读取持久化数组与 typed summaries，重算分布、matched 差值并逐 seed 报告缺项；
离线报告不执行 backward、评测或 gate 裁定，不改写源目录。保留 JSON、Markdown、SVG/PNG。

## 保留的约束与取代范围

取代 ADR 0034–0037 中将通用执行机制留在 experiments、旧研究入口和历史工作流保留、强制参考链
及冻结来源检查的决定。保留其中关于轻量离线分析、原始证据、source/output 分离、typed summaries、
配对/随机性、逐 seed bootstrap、未定义统计和结论边界的决定。

planner、PPO 更新算法、reward 公式与仿真语义不变。batch/action/log-prob/value/boundary 配对、
checkpoint 身份、train/held-out 不相交、无 optimizer 更新和失败证据继续由代码及测试保证。
人工 intervention/rollout 的 0.1 s 与普通 evaluation 的 0.5 s 边界保持不变。

历史实验记录及其原始命令、路径和结果不改写；新实现不声称重现历史结果。本次验收只运行相关
数值、配置、CLI、报告及 simulator 测试，不运行研究网格，不产生节能结论。当前实现契约见
[实验工具与离线分析](../agents/contracts/experiments.md)。
