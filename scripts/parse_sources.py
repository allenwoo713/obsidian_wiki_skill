"""源文档解析：docx / pdf / md / txt → ParsedDoc。"""
from __future__ import annotations
import hashlib
from pathlib import Path
from typing import List
import re

from models import ParsedDoc


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _extract_title(text: str, fallback: str) -> str:
    m = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    return m.group(1).strip() if m else fallback


def parse_markdown(path: Path) -> ParsedDoc:
    text = path.read_text(encoding="utf-8", errors="replace")
    title = _extract_title(text, path.stem)
    return ParsedDoc(
        path=path, title=title, text=text, tables=[],
        sha256=compute_sha256(path), doc_type="md",
    )


def parse_txt(path: Path) -> ParsedDoc:
    text = path.read_text(encoding="utf-8", errors="replace")
    title = _extract_title(text, path.stem)
    return ParsedDoc(
        path=path, title=title, text=text, tables=[],
        sha256=compute_sha256(path), doc_type="txt",
    )


def parse_docx(path: Path, assets_dir: Path = None) -> ParsedDoc:
    if assets_dir is not None:
        from extract_assets import extract
        return extract(path, assets_dir)
    from docx import Document
    doc = Document(str(path))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    text = "\n".join(paragraphs)
    tables: List[List[List[str]]] = []
    for t in doc.tables:
        rows = [[cell.text.strip() for cell in row.cells] for row in t.rows]
        tables.append(rows)
    if tables:
        text += "\n\n[表格]\n"
        for i, t in enumerate(tables):
            text += f"\n表 {i+1}:\n"
            for row in t:
                text += " | ".join(row) + "\n"
    title = _extract_title(text, path.stem)
    return ParsedDoc(
        path=path, title=title, text=text, tables=tables,
        sha256=compute_sha256(path), doc_type="docx",
    )


def parse_pdf(path: Path, assets_dir: Path = None) -> ParsedDoc:
    """PDF 解析：优先 pdfplumber，回退 PyPDF2。传 assets_dir 委托 extract_assets.extract()。"""
    if assets_dir is not None:
        try:
            from extract_assets import extract
            return extract(path, assets_dir)
        except ImportError as e:
            # pymupdf 未安装，回退纯文本（无图片提取）
            pass
        except Exception:
            # 其他提取器异常，也回退
            pass
    text = ""
    try:
        import pdfplumber
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                t = page.extract_text() or ""
                text += t + "\n"
    except ImportError:
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(str(path))
            for page in reader.pages:
                text += (page.extract_text() or "") + "\n"
        except ImportError:
            text = ""
    title = _extract_title(text, path.stem)
    return ParsedDoc(
        path=path, title=title, text=text, tables=[],
        sha256=compute_sha256(path), doc_type="pdf",
    )


_PARSERS = {
    ".docx": parse_docx,
    ".pdf": parse_pdf,
    ".md": parse_markdown,
    ".markdown": parse_markdown,
    ".txt": parse_txt,
}


def parse_file(path: Path, assets_dir: Path = None) -> ParsedDoc:
    suffix = path.suffix.lower()
    parser = _PARSERS.get(suffix)
    if parser is None:
        raise ValueError(f"unsupported file type: {suffix} ({path})")
    if assets_dir is not None and parser in (parse_docx, parse_pdf):
        return parser(path, assets_dir)
    return parser(path)
