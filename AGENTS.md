# Repository Guidelines

## 项目原则与当前状态

本项目用于个人科研，目标是通过 DPPO 微调 Diffusion Planner，在 MetaDrive 长程闭环中优化能耗，并使用 FASTSim 计量能耗。

逻辑和实验正确性优先于“跑起来”。所有输入、配置、依赖和张量契约都应显式定义：不要添加静默默认值、模糊兜底、自动降级或吞掉异常；缺少配置、权重、上游源码或运行时依赖时应立即失败，并给出可定位的错误。实现应保持简洁，优先使用成熟第三方库，避免为个人科研引入不必要的抽象层和兼容层。

仓库目前已完成阶段 0、MetaDrive 环境适配，以及官方 EMA 权重的**无交通短程闭环接口修正与评测**。阶段 0 已具备 checkpoint-compatible Diffusion Planner 主体、严格的官方 EMA 权重加载、观测/噪声张量契约、归一化和 10 步 DPM-Solver++ baseline sampler；初始权重位于 `checkpoints/DP-Origin/`，对应单元与集成测试已经存在。环境侧已实现 `KinematicTrajectoryPolicy`、`TrajectoryMetaDriveEnv` 和 `MetaDriveMapAdapter`：环境接收 `[80, 4]` 后轴局部轨迹，每次执行前 5 点，并能从程序化地图构造固定形状的 lane、route lane 和限速张量。无交通配置必须显式提供 `programmatic_lane_speed_limit_kmh`；环境 reset 后仅将 PGMap 的精确 `1000 km/h` 未设置哨兵替换为该值，保留已有真实限速，adapter 也会拒绝任何残留哨兵进入模型。`KinematicTrajectoryPolicy` 在 MetaDrive `after_step` 生命周期写入目标 waypoint，环境在物理阶段前清除残留速度，因此实际每个子步可严格匹配目标 world center 和 heading。`NoTrafficMetaDriveObservationAdapter` 会严格拒绝动态或静态交通参与者，并为已明确限定的空场景构造官方输入；`evaluate.py` 使用官方权重、固定噪声种子和滚动重规划运行该闭环，并落盘 resolved config、完整 raw observation、噪声、预测、目标/执行状态、逐点误差和限速审计。Windows 的直道/弯道 simulator 回归及 2 秒直道 checkpoint 评测已通过，但 Linux/Docker 基准环境仍是正式验收依据。

原始评测在约 1.7 s 出界的首要原因已确认：PGMap 的 `1000 km/h` 未设置哨兵曾被错误编码为有效的 `277.78 m/s` 限速输入，而非普通域偏移。修正后的 seed 0 直道短闭环以 18 条 `50 km/h` lane 输入运行 2 秒、不出界，位置最大执行误差为约 `4.6e-4 m`；这只验证了接口与执行器，不构成最终驾驶性能基线。剩余表现仍须通过 `S`/`SC`、多噪声 seed、20 秒闭环和上游逐元素数值对照评估，且 Linux/Docker/CUDA 是正式验收依据。优先查验：后轴/车辆中心坐标与朝向约定、地图特征和归一化分布、静止初始化与模型速度先验、采样噪声与随机种子，以及同一场景下上游 nuPlan/合成输入的预测对照；应保留每轮的 trace、视频、配置覆盖、checkpoint/hash 和上游 commit。不要通过平滑、裁剪、回退控制器或吞掉异常来掩盖失效轨迹。

项目尚未完成完整阶段 1：有交通场景所需的 ego/邻车 21 帧历史、静态物体和通用 `MetaDriveObservationAdapter` 尚未实现；当前 `NoTrafficMetaDriveObservationAdapter` 不能用于交通场景。低层 steering/throttle 轨迹跟踪、DPPO sampler、critic、rollout/PPO 更新和长程道路预瞄也仍待落地。`src/eco_planner/train.py` 继续保持占位入口；`evaluate.py` 仅是上述无交通官方权重闭环评测入口。不要把这些规划中的模块描述为已经可用，也不要为了让占位入口表面可运行而加入假实现。

## 事实来源与决策优先级

- `docs/DiffusionPlanner_MetaDrive_DPPO_总体方案.md`：总体架构、研究边界、阶段目标和验收门槛，仅作为整体流程参考，具体实施时根据情况调整细节。
- `docs/Diffusion-Planner_源码复用说明.md`：上游模型的张量契约、checkpoint 兼容边界和移植注意事项。
- `pyproject.toml` 与 `uv.lock`：Python 版本、直接依赖、工具配置和可复现环境的唯一来源。
- `README.md` 与 `docker/`：当前可执行的本地和容器工作流。
- 已实现代码和测试：当前行为事实；若与设计文档冲突，不要自行掩盖，应说明冲突并同步修正文档或实现。

新增设计决策或研究结论放在 `docs/`。实现偏离总体方案时，应在同一变更中记录原因、影响和验证方式。

## 目录与职责

- `src/eco_planner/models/`：Diffusion Planner、归一化、baseline/DPPO sampler、critic 和预瞄编码器。
- `src/eco_planner/envs/`：MetaDrive 环境封装、观测适配、坐标变换、路线预瞄和轨迹执行。
- `src/eco_planner/energy/`：FASTSim 行程构造、能耗计量和单位转换。
- `src/eco_planner/rl/`：rollout、buffer、GAE、PPO/DPPO 更新逻辑。
- `src/eco_planner/train.py`、`evaluate.py`：Hydra 驱动的训练和固定种子评测入口。
- `configs/`：按 `env/`、`model/`、`reward/`、`train/`、`experiment/` 组合实验配置；实验参数不得硬编码进 Python。
- `tests/`：快速单元测试及带 marker 的 GPU、仿真器和慢速测试。
- `docker/`：Ubuntu 22.04、CUDA 12.4 远程训练环境；Linux/CUDA 行为以该环境为准。
- `third_party/metadrive/`：运行时使用的本地可编辑 MetaDrive 源码，目录被 Git 忽略；固定 commit 和重建方法记录在 `third_party/README.md`。
- `ref/`：仅供阅读的 Diffusion Planner、DPPO 等上游快照，不是运行时依赖，不得从业务代码导入。
- `checkpoints/`：存放阶段性模型权重。
- `.env`：存放与设备相关的环境配置。
- `output/`：即时运行输入。

不要提交 `.venv/`、`.env`、`outputs/`、`checkpoints/`、`datasets/`、`wandb/`、`ref/` 或忽略的上游源码目录。新增脚本统一放在根目录 `scripts/`，不要散落临时入口。

## 环境与常用命令

项目固定使用 Python 3.10 和 uv。

Windows 本机用于编辑、格式化和不依赖仿真器/GPU 的快速验证：

```powershell
uv sync --all-groups
uv run pytest -m "not gpu and not simulator and not slow"
uv run ruff check .
uv run ruff format --check .
```

需要修改格式时运行 `uv run ruff format .`。依赖变更应先编辑 `pyproject.toml`，再在 Ubuntu/Docker 基准环境中更新并验证 `uv.lock`；提交二者，禁止只改其中一个。

远程 Ubuntu 服务器需预装 NVIDIA 驱动、Docker Engine 和 NVIDIA Container Toolkit：

```bash
export CUDA_VISIBLE_DEVICES=0
docker compose -f docker/compose.yaml build
docker compose -f docker/compose.yaml run --rm trainer
```

进入容器后，如需运行完整测试，先执行 `uv sync --frozen --all-groups`，再执行 `uv run pytest`。依赖锁定、MetaDrive/FASTSim 集成、CUDA、完整仿真和正式实验结果以 Ubuntu Docker 环境为准，不以 Windows 上偶然可运行作为验收依据。

## 实现约束

- 使用四空格缩进；公共 API 必须有类型标注；只为非显然逻辑写简短 docstring。遵循 Ruff 的 Python 3.10、100 字符行宽及 `E/F/I/UP` 规则。
- 模块/函数用 `snake_case`，类用 `PascalCase`，常量用 `UPPER_SNAKE_CASE`。文件路径使用 `pathlib.Path`，不要硬编码 Windows 或 Linux 绝对路径。
- 接口应窄且面向领域，例如 `MetaDriveObservationAdapter`、`DPPOTrajectorySampler`、`FastSimTraceMeter`；在出现真实复用需求前不要建立通用框架。
- Hydra 配置中的必需字段必须显式提供。读取配置时不要用随意的 `.get(..., default)` 隐藏缺项；对维度、单位、时间频率、坐标基准和取值范围做靠近边界的断言或校验。
- 固定随机种子并记录配置覆盖、checkpoint/hash、上游 commit 与数据来源；不要通过捕获宽泛异常、跳过坏样本或返回零值来制造可复现假象。

## 测试与验收

使用 pytest。每个 bug 修复必须带能先复现问题的回归测试。

使用已有 marker：`gpu`、`simulator`、`slow`。依赖硬件或仿真器的测试必须显式标记，不能用运行时自动跳过来掩盖错误；测试所需依赖或资源缺失时让测试明确失败或由调用命令主动排除对应 marker。

提交前至少运行与改动相关的 pytest、`uv run ruff check .` 和 `uv run ruff format --check .`。涉及模型、仿真器、FASTSim、CUDA 或锁文件时，还必须在 Docker 基准环境中运行对应集成测试。

## Git、安全与实验产物

保持改动范围聚焦，不覆盖工作区中的无关修改。提交信息使用简短、祈使式主题，并在一次提交中只完成一个逻辑变更。涉及轨迹、地图叠加或评测行为的 PR/实验记录应包含配置覆盖、随机种子、验证命令及必要的图表或视频。
