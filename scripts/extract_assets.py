"""文档解析调度器：按扩展名选 Parser，落盘图片，返回 ParsedDoc。"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Optional

from models import ParsedDoc
from parsers.base import ParseResult
from parse_sources import compute_sha256, _extract_title


class UnsupportedFormat(Exception):
    """不支持的文档格式。"""
    pass


def _is_truthy_env(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.lower() in ("1", "true", "yes", "y")


def extract(
    path: Path,
    assets_dir: Path,
    is_sensitive: Optional[bool] = None,
) -> ParsedDoc:
    """解析文档，落盘图片到 assets_dir，返回 ParsedDoc。

    Args:
        path: 源文档路径。
        assets_dir: 图片输出目录。
        is_sensitive: 仅对 PDF 有效。True 表示敏感，使用 MinerU Local；
            False/None 表示非敏感，使用 MinerU Cloud。None 时可通过环境变量
            MINERU_PDF_SENSITIVE=1 强制走本地。
    """
    ext = path.suffix.lower()

    if ext == ".docx":
        from parsers.docx_parser import DocxParser
        parser = DocxParser()
    elif ext in (
        ".pdf",
        ".pptx",
        ".doc",
        ".ppt",
        ".xls",
        ".xlsx",
        ".html",
        ".htm",
    ):
        if ext in (".pdf", ".pptx") and (
            is_sensitive is True
            or (is_sensitive is None and _is_truthy_env(os.environ.get("MINERU_PDF_SENSITIVE")))
        ):
            from parsers.mineru_local import MineruLocalPdfParser
            parser = MineruLocalPdfParser()
        else:
            from parsers.mineru_cloud import MineruCloudParser
            parser = MineruCloudParser()
    else:
        raise UnsupportedFormat(
            f"{ext} 不在支持列表: [.docx, .pdf, .doc, .ppt, .pptx, .xls, .xlsx, .html, .htm]"
        )

    result: ParseResult = parser.parse(path)

    # 落盘图片，按 sha256 去重
    assets_dir.mkdir(parents=True, exist_ok=True)
    sha_to_filename: Dict[str, str] = {}
    for ref, img_bytes in zip(result.images, result._image_bytes):
        if ref.sha256 in sha_to_filename:
            ref.filename = sha_to_filename[ref.sha256]
            ref.rel_path = f"assets/{ref.filename}"
        else:
            out_path = assets_dir / ref.filename
            out_path.write_bytes(img_bytes)
            sha_to_filename[ref.sha256] = ref.filename

    return ParsedDoc(
        path=path,
        title=_extract_title(result.text, path.stem),
        text=result.text,
        tables=result.tables,
        sha256=compute_sha256(path),
        doc_type=ext.lstrip("."),
        images=result.images,
    )
