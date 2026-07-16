"""MinerU local PDF parser: invokes MinerU pipeline in a dedicated venv subprocess."""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional

from parsers.base import DocumentParser, ParseResult
from parsers.mineru_common import mineru_markdown_to_parse_result


class MineruLocalPdfParser(DocumentParser):
    """Parse sensitive PDFs by running MinerU's native pipeline in an isolated venv.

    This bypasses the MinerU CLI server which may swallow logs and exceptions.
    """

    # ISSUE-05：不硬编码单一默认路径（避免锁死到某个人的本机），改为约定位置
    # 自动探测候选列表——命中就用，命中不到再抛错。用户仍可用
    # MINERU_PYTHON_EXE 环境变量或构造参数显式覆盖，优先级最高。
    _CANDIDATE_PATHS = [
        # WorkBuddy 用户的常见约定安装位置（跨平台各写一份）
        "~/.workbuddy/binaries/python/envs/mineru/Scripts/python.exe",  # Windows
        "~/.workbuddy/binaries/python/envs/mineru/bin/python",           # Linux/macOS
    ]

    def __init__(
        self,
        backend: str = "pipeline",
        language: str = "en",
        mineru_python_exe: Optional[str] = None,
    ):
        self.backend = backend
        self.language = language
        if mineru_python_exe is not None:
            self.mineru_python_exe = mineru_python_exe
        elif os.environ.get("MINERU_PYTHON_EXE"):
            self.mineru_python_exe = os.environ["MINERU_PYTHON_EXE"]
        else:
            detected = self._detect_mineru_python()
            if detected is not None:
                self.mineru_python_exe = detected
            else:
                raise FileNotFoundError(
                    "MinerU Local Python 解释器未配置，且约定位置未探测到。\n"
                    "MinerU Local 是可选组件，仅在处理敏感文档时需要（非敏感文档走 MinerU Cloud，不受影响）。\n"
                    "解决方式（任选其一）：\n"
                    "  1. 设置环境变量 MINERU_PYTHON_EXE，指向已装 MinerU 的 venv python 可执行文件；\n"
                    "  2. 或在 <skill_dir>/.env 中填写 MINERU_PYTHON_EXE=<你的路径>；\n"
                    "  3. 或按约定路径安装 venv："
                    f" {self._CANDIDATE_PATHS[0]}（Windows）"
                    f" / {self._CANDIDATE_PATHS[1]}（Linux/macOS），安装后无需任何配置。\n"
                    "完整安装步骤见 README.md > 文档解析后端 > MinerU Local。"
                )

    @classmethod
    def _detect_mineru_python(cls) -> Optional[str]:
        """按约定位置探测 MinerU venv python，找到第一个存在的就返回，否则返回 None。"""
        for candidate in cls._CANDIDATE_PATHS:
            expanded = os.path.expanduser(candidate)
            if os.path.isfile(expanded):
                return expanded
        return None

    @staticmethod
    def _result_subdir(ext: str) -> str:
        """MinerU do_parse 输出子目录：PDF→auto，Office（docx/pptx/xlsx）→office。"""
        return "office" if ext.lower() in (".docx", ".pptx", ".xlsx") else "auto"

    def parse(self, path: Path) -> ParseResult:
        pdf_path = Path(path)
        mineru_python_exe = Path(self.mineru_python_exe)
        if not mineru_python_exe.exists():
            raise FileNotFoundError(f"MinerU Python executable not found: {mineru_python_exe}")

        runner_script = Path(__file__).with_name("_mineru_local_runner.py")

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            cmd = [
                str(mineru_python_exe),
                str(runner_script),
                str(pdf_path),
                str(output_dir),
                self.backend,
                self.language,
            ]
            env = os.environ.copy()
            env["MINERU_MODEL_SOURCE"] = "local"
            env["MINERU_DEVICE_MODE"] = "cpu"
            env["CUDA_VISIBLE_DEVICES"] = ""

            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"MinerU local parser failed with code {result.returncode}: {result.stderr}"
                )

            stem = pdf_path.stem
            subdir = self._result_subdir(pdf_path.suffix)
            result_dir = output_dir / stem / subdir
            md_path = result_dir / f"{stem}.md"
            images_dir = result_dir / "images"

            markdown = md_path.read_text(encoding="utf-8")

            image_bytes_map: Dict[str, bytes] = {}
            if images_dir.exists():
                for img_path in images_dir.iterdir():
                    if img_path.is_file():
                        image_bytes_map[img_path.name] = img_path.read_bytes()

            return mineru_markdown_to_parse_result(
                markdown=markdown,
                doc_slug=stem,
                images_dir=images_dir,
                image_bytes_map=image_bytes_map,
            )
