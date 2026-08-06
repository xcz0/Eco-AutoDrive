# Repository Guidelines

## 开始工作前

按任务需要阅读：

1. `docs/STATUS.md`：当前完成度、已验证结论和下一步；
2. `docs/LOGIC.md`：已实现系统必须遵守的数据与执行契约；
3. `docs/DECISIONS.md`：已经确定的重要方案选择；
4. `docs/CORRECTIONS.md`：与当前任务相关的历史错误和修正；
5. `docs/RESEARCH.md`：尚未确定或尚未实现的研究设想；
6. `experiments/README.md`：实际评测配置、种子、结果和产物来源；
7. `README.md`、`pyproject.toml`、`uv.lock`、相关代码和测试：当前可执行方式。

事实优先级为：已实现代码和测试 > 配置与依赖文件 > `docs/STATUS.md` > 其他文档。发现冲突时应明确指出，并同步修正文档或实现。

## 工作原则

本项目用于个人科研，正确性优先于表面可运行：

- 必需输入、配置、依赖、张量形状、单位、时间频率和坐标基准必须显式定义；
- 缺少必需条件时立即失败，并给出可定位错误；
- 不添加静默默认、自动降级、宽泛异常捕获或坏样本跳过；
- 不用平滑、裁剪、中心线投影、回退控制器或零轨迹掩盖模型失效；
- 优先使用成熟第三方库，出现明确复用需求前不建立通用框架；
- `STATUS.md` 中未完成的模块不得描述为可用，也不得为占位入口加入假实现。

文档职责：系统不变量写入 `LOGIC.md`；已确定的取舍写入 `DECISIONS.md`；当前进度写入 `STATUS.md`；历史错误写入 `CORRECTIONS.md`；未定研究设想写入 `RESEARCH.md`；实际运行结果写入 `experiments/README.md`。

## 目录职责

- `src/eco_planner/models/`：Diffusion Planner、归一化及模型扩展；
- `src/eco_planner/envs/`：MetaDrive 封装、观测适配、坐标变换和轨迹执行；
- `src/eco_planner/energy/`：能耗计量、行程构造和单位转换；
- `src/eco_planner/rl/`：后续强化学习相关代码；
- `src/eco_planner/train.py`、`scripts/evaluate.py`：训练和评测入口；
- `configs/`：环境、模型、训练和实验配置，实验参数不得硬编码进 Python；
- `tests/`：单元测试及带 `gpu`、`simulator`、`slow` marker 的测试；
- `scripts/`：构建、部署和诊断脚本；
- `third_party/metadrive/`：运行时本地 editable 源码；
- `ref/`：只读上游快照，业务代码不得导入；
- `checkpoints/`、`outputs/`：本地权重和即时产物，不得提交。

不要提交 `.venv/`、`.env`、`outputs/`、`checkpoints/`、`datasets/`、`wandb/`、`ref/` 或被忽略的上游源码。

## 实现约束

- 公共 API 必须有类型标注；只为非显然逻辑写简短 docstring；
- 遵循 Python 3.10、Ruff 100 字符行宽及 `E/F/I/UP` 规则；
- 文件路径使用 `pathlib.Path`，不得硬编码平台绝对路径；
- Hydra 必需字段必须显式提供，不得用随意默认值隐藏缺项；
- 在系统边界验证维度、dtype、设备、单位、范围和有限性；
- 固定随机种子，并记录配置覆盖、上游版本、数据来源和运行环境；
- 依赖变更必须同时更新 `pyproject.toml` 和 `uv.lock`。

## 验证要求

每个 bug 修复必须带能复现问题的回归测试。依赖硬件或仿真器的测试必须显式标记；缺少依赖时由调用命令排除 marker，测试本身不得静默跳过。

Windows 快速验证：

```powershell
uv sync --all-groups
uv run pytest -m "not gpu and not simulator and not slow"
uv run ruff check .
uv run ruff format --check .
```

涉及 MetaDrive 的改动还应运行本机可用的显式 simulator 测试。

## Git 与实验记录

提交信息使用简短、祈使式主题；一次提交只完成一个逻辑变更。涉及轨迹、地图、能耗或评测行为的实验，应在 `experiments/README.md` 记录代码状态、未提交差异、上游版本、数据或地图、resolved config、Hydra overrides、随机种子、验证命令、结果和产物路径。

大型产物和原始日志保留在 `outputs/` 或外部存储，不写入 Markdown，不提交 Git。未记录的元数据应明确标记为“未记录”，不得补猜。
