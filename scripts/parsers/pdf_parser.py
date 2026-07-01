"""PdfParser：PyMuPDF 提取文本、图片、图注。"""
from __future__ import annotations
import hashlib
from pathlib import Path
from typing import List

from models import ImageRef
from parsers.base import DocumentParser, ParseResult
from parsers.utils import slugify, image_filename, attach_captions

class PdfParser(DocumentParser):
    def parse(self, path: Path) -> ParseResult:
        import fitz
        doc_slug = slugify(path.stem)
        doc = fitz.open(str(path))
        text_parts: List[str] = []
        images: List[ImageRef] = []
        image_bytes_list: List[bytes] = []
        img_seq = 0

        for page_idx, page in enumerate(doc):
            d = page.get_text("dict")
            blocks = d.get("blocks", [])
            for block in blocks:
                if block.get("type") == 0:
                    text = self._extract_text(block)
                    if text.strip():
                        text_parts.append(text)
                elif block.get("type") == 1:
                    img_seq += 1
                    result = self._extract_image(doc, block, doc_slug, img_seq, page_idx)
                    if result:
                        ref, img_bytes = result
                        images.append(ref)
                        image_bytes_list.append(img_bytes)
                        text_parts.append(f"{{{{IMG|{ref.rel_path}|图注: 待补}}}}")

        doc.close()

        # L0: pdfplumber 表格提取，追加到 text_parts（在 attach_captions 之前，
        # 避免表格内容干扰图注匹配）
        tables = self._extract_tables_pdfplumber(path)
        if tables:
            text_parts.append("\n[表格]\n")
            for i, t in enumerate(tables):
                text_parts.append(f"\n表 {i+1}:\n")
                for row in t:
                    text_parts.append(" | ".join(row))

        text, images = attach_captions("\n".join(text_parts), images)
        return ParseResult(
            text=text, images=images, tables=tables,
            _image_bytes=image_bytes_list,
        )

    def _extract_text(self, block: dict) -> str:
        lines = []
        for line in block.get("lines", []):
            spans = []
            for span in line.get("spans", []):
                spans.append(span.get("text", ""))
            lines.append("".join(spans))
        return "\n".join(lines)

    def _extract_image(self, doc, block: dict, doc_slug: str, img_seq: int, page_idx: int):
        """从页内 images 列表逐张提取（按 xref 顺序，与 get_text blocks 对齐）。"""
        page = doc[page_idx]
        for img_info in page.get_images(full=True):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
                img_bytes = base_image["image"]
                ext = base_image["ext"]
                sha = hashlib.sha256(img_bytes).hexdigest()
                fname = image_filename(doc_slug, img_seq, ext)
                ref = ImageRef(
                    filename=fname,
                    rel_path=f"assets/{fname}",
                    caption="",
                    source_media_name=f"xref={xref}",
                    sha256=sha,
                    page_or_section=f"page {page_idx + 1}",
                )
                return ref, img_bytes
            except Exception:
                continue
        return None

    def _extract_tables_pdfplumber(self, path) -> list:
        """用 pdfplumber 二次遍历提取表格。返回 List[List[List[str]]]。"""
        import pdfplumber
        all_tables = []
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                for t in page.extract_tables() or []:
                    cleaned = [[(c or "").strip() for c in row] for row in t]
                    if any(any(cell for cell in row) for row in cleaned):
                        all_tables.append(cleaned)
        return all_tables
