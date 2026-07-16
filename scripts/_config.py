"""集中配置加载模块。

任何脚本只需 `import _config`（越早越好，需在 `from build_index import ...` /
`from models import ...` 等触发模型加载的 import 之前），即可幂等地：

1. 自动定位 skill 根目录（`SKILL_DIR`），无需占位符手填。
2. 加载 `<skill_dir>/.env` 到 `os.environ`（若存在），使 MINERU_API_TOKEN /
   MINERU_PYTHON_EXE / WIKI_EMBEDDER_LOCAL_PATH 等配置对所有脚本统一生效。

根因背景（ISSUE-01）：此前只有 update_wiki.py 手写了 load_dotenv，
build_index.py / query.py / build_graph.py 均不加载 .env，导致写入 .env 的
配置（如 WIKI_EMBEDDER_LOCAL_PATH）对这些脚本不生效。本模块统一收口，
避免同样的遗漏再次发生。

设计原则：
- 幂等：多次 import 只加载一次（Python import 机制天然保证）。
- 不覆盖已存在的系统环境变量（`load_dotenv` 默认 `override=False`），
  保证命令行显式 `set VAR=xxx` 或 CI 环境变量优先级高于 .env 文件。
- 加载失败（python-dotenv 未安装、.env 不存在）静默跳过，不阻断脚本运行——
  所有读取这些配置的地方都应有自己的回退逻辑（探测/报错提示），
  _config.py 只负责“如果 .env 存在就把它读进环境变量”，不代替业务层做校验。
"""
from __future__ import annotations

import os
from pathlib import Path

# skill 根目录：scripts/_config.py 的上一级即 <skill_dir>（含 SKILL.md）。
# 自推导，任何脚本都不再需要手填 <skill_dir> 占位符。
SKILL_DIR = Path(__file__).resolve().parent.parent


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        # python-dotenv 未安装：跳过。核心功能仍可通过系统环境变量运行。
        return

    env_path = SKILL_DIR / ".env"
    if env_path.exists():
        load_dotenv(env_path)  # override=False（默认）：不覆盖已存在的系统环境变量


_load_env()


def venv_python() -> str | None:
    """返回 .env / 环境变量中配置的核心 venv python 路径（可选，供 wrapper 脚本使用）。"""
    return os.environ.get("WIKI_VENV_PYTHON")
