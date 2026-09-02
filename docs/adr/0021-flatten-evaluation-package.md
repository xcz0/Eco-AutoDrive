# Flatten the evaluation package while preserving dependency direction

**Status:** Superseded by ADR 0029
**Date:** 2026-08-25

ADR 0020 通过 `runtime/`、`artifacts/` 和 `analysis/` 子包建立了在线执行、持久化边界与
离线分析之间的单向依赖。当前每个子包只有少量实现文件，目录、空 `__init__.py` 和跨文件
转发已经超过其实际职责数量，增加了查找与修改成本。

因此，将 evaluation 压平为按职责命名的模块：`runtime.py`、`execution.py`、
`artifacts.py`、`trace.py`、`summaries.py`、`analysis.py`、`rendering.py`、`config.py`
和 `runner.py`。不保留旧内部导入路径的兼容转发。

压平只改变物理组织，不取消 ADR 0020 建立的逻辑依赖方向：

- `artifacts.py` 是轻量持久化边界，包含严格模型、trace schema/validation 和 JSON/NPZ
  reader/writer；导入它不得加载 Torch、MetaDrive 或 Panda3D。
- `analysis.py` 只依赖 artifacts、NumPy 与 OmegaConf，不参与在线执行。
- `trace.py`、`runtime.py`、`execution.py` 和 `runner.py` 组成在线路径，可以依赖
  artifacts；artifact 和 analysis 不得反向依赖在线路径。
- rendering 保持独立且延迟导入，避免 Panda3D 进入离线 reader 或无视频写入路径。

配置、随机流、仿真执行、artifact schema 和分析输出不因本次组织调整而变化。当前字段与
行为继续由 `docs/agents/system-contract.md` 规定。
