# Eco-AutoDrive 架构与流程图

三张图面向宽屏演示，使用中文说明与代码中的关键英文名，以横向分列布局为主。闭环与训练图采用上下分行的回路，避免单行过长导致 slide 中字号过小。这里是当前实现的展示视图；执行契约仍以 [system contract](../agents/system-contract.md)及对应代码为准。

| 图 | LikeC4 源文件 | 主要代码依据 |
| --- | --- | --- |
| 项目架构 | [architecture.c4](architecture.c4) | [jobs](../../src/eco_planner/jobs.py)、[evaluation engine](../../src/eco_planner/evaluation/engine.py)、[trainer](../../src/eco_planner/rl/trainer.py) |
| 闭环仿真 | [closed-loop.c4](closed-loop.c4) | [environment slot](../../src/eco_planner/envs/metadrive/slot.py)、[execution](../../src/eco_planner/envs/metadrive/execution.py)、[evaluation agent](../../src/eco_planner/evaluation/inference/agent.py) |
| 训练数据流程 | [training-data.c4](training-data.c4) | [collector](../../src/eco_planner/rl/rollout/collector.py)、[rollout runtime](../../src/eco_planner/rl/rollout/runtime.py)、[PPO](../../src/eco_planner/rl/optimization/ppo.py)、[reward](../../src/eco_planner/rl/reward/objectives/plannerrft.py) |

导出图片可直接插入 slide，按比例缩放并为标题、讲解注释留出空间：

- [项目架构 PNG](images/index.png)
- [闭环仿真 PNG](images/closed_loop.png)
- [训练数据流程 PNG](images/training_data.png)

## 配色与图形图例

三张图共享 [specification.c4](specification.c4) 中的元素样式。颜色区分模块职责及其与训练更新的关系，下表中的颜色名称对应当前导出图的视觉效果；
LikeC4 token 用于精确对应源文件，切换主题时具体色值可能变化。

| 颜色 | LikeC4 类型 / token | 含义 | 各图中的例子 |
| --- | --- | --- | --- |
| 灰蓝色 | `frozen` / `slate` | 冻结的 Planner 计算及其特征，PPO 不更新 Planner 参数 | 架构图的冻结 Diffusion Planner；闭环图的冻结规划器推理；训练图的冻结 context 与 reference、Guided DDIM-5 |
| 橙色 | `trainable` / `amber` | 可训练策略及其训练、优化环节 | 架构图的策略训练；训练图的 Exploration Policy、PPO 更新 |
| 绿色 | `simulation` / `green` | 仿真环境、实际状态及在线交互执行 | 架构图的 MetaDrive 环境适配；闭环图的仿真真实状态、运动学轨迹执行；训练图的在线 rollout |
| 蓝色 | `module` / `blue` | 作业编排、运行支持、数据处理或算法计算 | 架构图的作业组合、闭环评测、共享运行时、实验与分析；闭环图的观测构建；训练图的 RewardEvaluator、GAE 与训练 batch |
| 青蓝色 | `data` / `sky` | 配置、内存数据集合或持久化产物，结合节点文字判断 | 架构图的配置与运行入口、运行产物；闭环图的联合预测、评测记录；训练图的 episode 数据、CPU 审计、策略与训练状态 |

橙色表示与训练更新有关的职责，不表示节点中的所有内容都是可训练参数：例如 PPO 更新是优化过程，架构图的策略训练还包含 rollout 和 reward。灰蓝色表示 Planner 参数保持冻结，不表示 guided sampling 不计算对输入的梯度。

| 图形或连线 | 阅读方式 |
| --- | --- |
| 矩形 | 一个逻辑模块、计算环节或仿真职责；不自动表示独立进程、设备或部署单元 |
| 圆柱体 | 配置、数据集合或产物；不限定为数据库或磁盘文件。例如联合预测、按 episode 收集是内存数据，NPZ 和 checkpoint 是保存产物。“配置与运行入口”将配置及其 CLI 入口合并展示 |
| 节点主标题与小字 | 主标题标识职责，小字补充代码模块、数据内容或执行语义 |
| 普通灰色实线箭头 | 主要调用、执行或数据传递，具体含义由箭头文字说明；不编码同步 / 异步或进程间通信方式 |
| 灰蓝色虚线箭头 | 仅训练图使用：在线 rollout 产生下一轮观测，反馈到冻结 context 与 reference 的准备环节 |
| 橙色虚线箭头 | 仅训练图使用：PPO 更新 Exploration Policy 参数，供后续采集使用 |

回路由箭头方向和文字共同表达，并非所有反馈都画成虚线；闭环仿真图的实际状态反馈仍使用普通实线。箭头上的数据也不表示反向传播路径，只有训练图橙色反馈线专门标识策略参数更新。

## 三张图的阅读方式

| 图 | 主体阅读顺序 | 配色与图形的重点 |
| --- | --- | --- |
| 项目架构 | 左侧配置与入口进入作业组合，向右分为评测和训练，再连接共享运行时、Planner、环境与产物分析 | 橙色突出训练职责，灰蓝色突出冻结 Planner，绿色突出环境；青蓝色圆柱分别表示配置入口与运行产物。连线表示主要依赖，不是完整执行时序 |
| 闭环仿真 | 上排由实际状态向右进入观测与推理，右侧向下产生预测，下排向左执行并反馈真实状态；预测和执行事实分别进入评测记录 | 两个绿色矩形对应状态与执行，灰蓝色矩形对应 Planner 推理；预测圆柱与评测记录圆柱分别表示内存结果和保存产物 |
| 训练数据流程 | 上排向右完成冻结特征准备、策略采样、guided planning 和 rollout；下排由 reward 向左进入 episode 数据、GAE 和 PPO；审计与 checkpoint 就近向下保存 | 橙色节点及橙色虚线突出策略更新；灰蓝色顶部虚线单独表达观测反馈；两个保存产物圆柱与中间的内存 episode 数据圆柱分开，便于区分训练输入和审计输出 |

## 讲解边界

讲解时需保留的边界：

- 项目架构图展示主要调用和数据依赖，实验模块还通过 job 入口编排运行。
- 闭环仿真图使用 evaluation 的执行周期；训练 rollout 的周期在训练图中单独注明。
- 训练数据来自在线 rollout。预训练权重是外部输入，图中没有引入离线专家数据训练链路。
- reference 是当前冻结模型预测；PPO 更新 Exploration Policy，planner 参数保持冻结。
- 环境使用运动学轨迹执行，fuel proxy 的解释限于该仿真条件。
  FASTSim 属于独立离线指标路径，不进入图中的在线 reward。
- PPO training TensorDict 与 CPU audit 是不同数据边界；NPZ 审计文件不是 PPO 更新的回读数据源。

## 预览与导出

在仓库根目录运行：

```powershell
likec4 validate --json --no-layout docs/diagrams
likec4 export png --theme light --flat --outdir docs/diagrams/images docs/diagrams
likec4 serve docs/diagrams
```

浏览器预览中三个 view ID 为 `index`、`closed_loop`、`training_data`。
PNG 导出需要该 CLI 所用 Playwright 能启动 Chromium；首次使用时可能需要安装对应浏览器。
本次导出使用已安装的 Chromium，未修改项目 Python 依赖或运行契约。
