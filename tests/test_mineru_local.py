"""MineruLocalPdfParser 测试：mineru 3.4.2 支持 pptx，守卫已移除，pptx/pdf 均走到 venv 检查。"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from parsers.mineru_local import MineruLocalPdfParser


def test_mineru_local_pptx_not_blocked(tmp_path: Path):
    """mineru 3.4.2 支持 pptx，守卫不应拦截；用不存在的 venv 路径，期望 FileNotFoundError 而非 ValueError。"""
    pptx_path = tmp_path / "test.pptx"
    pptx_path.write_bytes(b"fake pptx")
    parser = MineruLocalPdfParser(mineru_python_exe=str(tmp_path / "nonexistent" / "python.exe"))

    with pytest.raises(FileNotFoundError):
        parser.parse(pptx_path)


def test_mineru_local_accepts_pdf(tmp_path: Path):
    """PDF 传入时不被拦截（用不存在的 venv 路径，守卫放行后立即 FileNotFoundError）。"""
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(b"fake pdf")
    parser = MineruLocalPdfParser(mineru_python_exe=str(tmp_path / "nonexistent" / "python.exe"))

    with pytest.raises(FileNotFoundError):
        parser.parse(pdf_path)


def test_result_subdir_pdf():
    """PDF 输出到 auto/ 子目录。"""
    assert MineruLocalPdfParser._result_subdir(".pdf") == "auto"


def test_result_subdir_office():
    """Office 文档（docx/pptx/xlsx）输出到 office/ 子目录。"""
    assert MineruLocalPdfParser._result_subdir(".pptx") == "office"
    assert MineruLocalPdfParser._result_subdir(".docx") == "office"
    assert MineruLocalPdfParser._result_subdir(".xlsx") == "office"
