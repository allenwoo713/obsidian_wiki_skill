# Changelog

本项目变更记录，遵循 [Keep a Changelog](https://keepachangelog.com/) 风格。

## [Unreleased]

### Changed — 通用化改造（让 skill 可被他人复用）

#### 路径变量化

- **SKILL.md / README.md**：所有硬编码的本机绝对路径（用户目录、数据目录等）替换为占位符 `<venv_python>` / `<skill_dir>` / `<project_root>` / `<mineru_python>`，调用方按本机实际路径替换。
- **`scripts/build_index.py`**：删除 embedding 模型候选路径中硬编码的本机路径项；保留 env var (`WIKI_EMBEDDER_LOCAL_PATH`) → `~/.workbuddy/...` expanduser → HF 在线下载 三级回退。
- **`scripts/parsers/mineru_local.py`**：`_DEFAULT_MINERU_PYTHON` 由硬编码本机路径改为 `None`；构造函数在 `mineru_python_exe` 未传且 `MINERU_PYTHON_EXE` 环境变量未设置时，抛带说明的 `FileNotFoundError`（提示设置 env 或传参），不再静默回退到任何默认路径。
- **`scripts/build_graph.py`**：HTML header 从硬编码项目名改为从 `purpose.md` 动态读取标题，读不到时用 `"Wiki"` 兜底。
- **`.env.example`**：`MINERU_PYTHON_EXE` 改为占位符模板（含 Windows / Linux 路径示例），不再写死具体路径。

#### 示例脱敏

- **产品领域**：从原作者的实际知识库领域替换为虚构的工业相机领域（Acme VisionCam / Vega Opticam / ClientX），保留工作流示范价值。
- **查询预处理示例**：改为通用引导（"对照知识库 `purpose.md` 中的产品实体清单"），不再硬编码具体产品名列表。
- **端到端示例 + 出处标注示例**：全部替换为 Acme 工业相机场景，规格数值虚构。
- **README.md**：「专为 WorkBuddy agent 设计」改为「专为 AI agent 设计（兼容 WorkBuddy / Claude Code / 其他 agent 框架）」。

#### Tests 处理

- **`tests/` 目录不再随 skill 仓库公开发布**（`.gitignore` 已添加 `tests/` 排除规则；`git rm --cached -r tests/` 已从 git index 移除，本地文件保留）。
- **删除调试用 e2e 脚本**：`tests/debug_cloud_raw.py` / `e2e_cloud_fixed.py` / `e2e_cloud_real.py`（含真实知识库路径与 MinerU Cloud 调用，仅供作者本地调试，已从文件系统与 git index 移除）。

#### 历史重写

- 使用 `git filter-repo --replace-text` 清除了所有历史 commit 中的本机路径、用户名、真实产品名/客户名/规格数值。
- 验证：重写后历史中上述敏感字符串出现次数为 0（CHANGELOG.md 与 commit message 中的描述性提及除外）。

### Added

- **CHANGELOG.md**：本次创建，记录通用化改造。

### Notes

- 本次改造**不改变 skill 的功能行为**，仅做路径参数化与示例脱敏。
- 已脱敏的单元测试本地仍可运行验证（`pytest -p no:cacheprovider`），但不在公开发布的仓库中包含。如他人需要测试用例，可联系作者。
