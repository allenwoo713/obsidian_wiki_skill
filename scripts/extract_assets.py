"""图片提取调度器：按扩展名选 Parser，落盘图片，返回 ParsedDoc。"""
from __future__ import annotations
from pathlib import Path
from typing import Dict

from models import ParsedDoc
from parsers.base import ParseResult
from parse_sources import compute_sha256, _extract_title


class UnsupportedFormat(Exception):
    pass


def extract(path: Path, assets_dir: Path) -> ParsedDoc:
    """解析文档，落盘图片到 assets_dir，返回 ParsedDoc（含 images）。"""
    ext = path.suffix.lower()

    # 延迟导入，避免解析器不存在时崩溃
    if ext == ".docx":
        from parsers.docx_parser import DocxParser
        parser = DocxParser()
    elif ext == ".pdf":
        try:
            from parsers.pdf_parser import PdfParser
        except ImportError:
            raise UnsupportedFormat(
                f"{ext} 解析器未安装（parsers.pdf_parser），"
                f"请完成 PdfParser 实现后重试"
            )
        parser = PdfParser()
    elif ext == ".pptx":
        try:
            from parsers.pptx_parser import PptxParser
        except ImportError:
            raise UnsupportedFormat(
                f"{ext} 解析器未安装（parsers.pptx_parser），"
                f"请完成 PptxParser 实现后重试"
            )
        parser = PptxParser()
    else:
        raise UnsupportedFormat(f"{ext} 不在支持列表: [.docx, .pdf, .pptx]")

    result: ParseResult = parser.parse(path)

    # 落盘图片，按 sha256 去重
    assets_dir.mkdir(parents=True, exist_ok=True)
    sha_to_filename: Dict[str, str] = {}
    for ref, img_bytes in zip(result.images, result._image_bytes):
        if ref.sha256 in sha_to_filename:
            # 复用已落盘文件
            ref.filename = sha_to_filename[ref.sha256]
            ref.rel_path = f"assets/{ref.filename}"
        else:
            out_path = assets_dir / ref.filename
            out_path.write_bytes(img_bytes)
            sha_to_filename[ref.sha256] = ref.filename

    # 构造 ParsedDoc
    return ParsedDoc(
        path=path,
        title=_extract_title(result.text, path.stem),
        text=result.text,
        tables=result.tables,
        sha256=compute_sha256(path),
        doc_type=ext.lstrip("."),
        images=result.images,
    )
