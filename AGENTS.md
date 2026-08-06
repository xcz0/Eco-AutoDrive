# Repository Guidelines

## 开始工作前

按任务需要依次阅读：

1. `docs/STATUS.md`：确认当前实现边界、已验证结论和紧邻的下一步；
2. `docs/LOGIC.md`：确认输入、单位、坐标、时间、采样、执行和能耗契约；
3. `docs/DECISIONS.md`：确认相关方案为何被选定，以及是否已经实施；
4. `docs/CORRECTIONS.md`：涉及历史故障的区域先阅读既有根因和回归证据；
5. `experiments/README.md`：涉及评测结论时确认代码、配置、种子和产物来源；
6. `README.md`、`pyproject.toml`、`uv.lock` 和相关实现/测试：确认当前可执行方式。

事实优先级为：已实现代码和测试 > `pyproject.toml`/`uv.lock`/配置 > `docs/STATUS.md` > 其他设计和实验文档。发现冲突时不要自行掩盖；应说明冲突，并在同一变更中修正过期文档或实现。

## 工作原则

本项目用于个人科研。逻辑和实验正确性优先于“跑起来”：

- 所有必需输入、配置、依赖、张量形状、单位、时间频率和坐标基准必须显式定义；
- 缺少配置、权重、上游源码或运行时依赖时立即失败，并给出可定位错误；
- 不添加静默默认值、模糊兜底、自动降级、宽泛异常捕获或坏样本跳过；
- 不用平滑、裁剪、中心线投影、回退控制器或零轨迹掩盖模型失效；
- 实现保持简洁，优先使用成熟第三方库；出现真实复用需求前不建立通用框架或兼容层；
- 不得把 `docs/STATUS.md` 中“未完成”的模块描述为可用，也不得为占位入口加入假实现。
- 禁止写哈希值与SHA256，禁止为基本不可能出现的case写防御；需要rubric的地方不要过度机械化。

新增系统不变量写入 `docs/LOGIC.md`；新方案取舍写入 `docs/DECISIONS.md`；当前进度写入 `docs/STATUS.md`；错误认知、根因和修复证据写入 `docs/CORRECTIONS.md`；实际运行结果写入`experiments/README.md`。不要把普通代码改动逐条写入这些文档。

## 目录职责

- `src/eco_planner/models/`：Diffusion Planner、归一化、baseline/DPPO sampler、critic 和预瞄编码；
- `src/eco_planner/envs/`：MetaDrive 环境封装、观测适配、坐标变换、路线预瞄和轨迹执行；
- `src/eco_planner/energy/`：FASTSim 行程构造、能耗计量和单位转换；
- `src/eco_planner/rl/`：rollout、buffer、GAE 和 PPO/DPPO 更新；
- `src/eco_planner/train.py`、`scripts/evaluate.py`：Hydra 驱动的训练和评测入口；
- `configs/`：环境、模型、奖励、训练和实验配置；实验参数不得硬编码进 Python；
- `tests/`：快速单元测试和带 marker 的 GPU、仿真器、慢速测试；
- `docker/`：预留给后续服务器训练阶段评估的 Ubuntu 22.04 / CUDA 12.4 环境；
- `scripts/`：构建、部署和诊断脚本；禁止在其他目录散落临时入口；
- `third_party/metadrive/`：运行时本地 editable 源码；固定版本和重建方法应有记录；
- `ref/`：只读上游快照，不是运行时依赖，业务代码不得导入；
- `checkpoints/`、`outputs/`：本地权重和即时产物，均不得提交。

不要提交 `.venv/`、`.env`、`outputs/`、`checkpoints/`、`datasets/`、`wandb/`、`ref/` 或被忽略的上游源码。保持改动聚焦，不覆盖工作区中的无关修改。

## 实现约束

- 公共 API 必须有类型标注；只为非显然逻辑写简短 docstring；
- 遵循 Ruff 的 Python 3.10、100 字符行宽及 `E/F/I/UP` 规则；
- 模块和函数使用 `snake_case`，类使用 `PascalCase`，常量使用 `UPPER_SNAKE_CASE`；
- 文件路径使用 `pathlib.Path`，不得硬编码 Windows 或 Linux 绝对路径；
- 接口保持窄且面向领域，例如 `MetaDriveObservationAdapter`、`DPPOTrajectorySampler`、`FastSimTraceMeter`；
- Hydra 必需字段必须显式提供；不得用随意的 `.get(..., default)` 隐藏缺项；
- 在系统边界验证维度、dtype、设备、单位、范围和有限性；
- 固定随机种子，并记录配置覆盖、checkpoint/hash、上游 commit 和数据来源。

依赖变更先编辑 `pyproject.toml`，再更新并验证 `uv.lock`；二者必须同时提交。当前基础代码构建阶段先完成本机锁文件和快速测试验证，进入服务器训练阶段后再补充 Linux/CUDA 环境复验。

## 验证要求

每个 bug 修复必须带能复现问题的回归测试。使用已有 marker：`gpu`、`simulator`、`slow`。依赖硬件或仿真器的测试必须显式标记；缺少依赖时由调用命令主动排除 marker，测试本身不得运行时静默跳过。

Windows 快速验证：

```powershell
uv sync --all-groups
uv run pytest -m "not gpu and not simulator and not slow"
uv run ruff check .
uv run ruff format --check .
```

需要修改格式时运行 `uv run ruff format .`。涉及 MetaDrive 的改动还要运行本机可用的显式simulator 测试。当前目标是补齐代码逻辑并用小规模配置跑通最小强化学习流程。



## Git 与实验记录

提交信息使用简短、祈使式主题；一次提交只完成一个逻辑变更。涉及轨迹、地图、能耗或评测行为的实验，必须在 `experiments/README.md` 记录：代码 commit、未提交 diff、上游版本、数据/地图、checkpoint/hash、resolved config、Hydra overrides、随机种子、验证命令、结果和产物路径。

大型产物和原始日志保留在 `outputs/` 或外部存储，不写入 Markdown，不提交 Git。若某项元数据当时未记录，应明确写“未记录”，不得根据当前环境补猜。
