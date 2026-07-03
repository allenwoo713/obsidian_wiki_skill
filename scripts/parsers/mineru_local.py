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

    _DEFAULT_MINERU_PYTHON = "<home>/.workbuddy/binaries/python/envs/mineru/Scripts/python.exe"

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
            self.mineru_python_exe = self._DEFAULT_MINERU_PYTHON

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
            auto_dir = output_dir / stem / "auto"
            md_path = auto_dir / f"{stem}.md"
            images_dir = auto_dir / "images"

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
