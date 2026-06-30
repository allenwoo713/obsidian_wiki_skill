"""PdfParser：PyMuPDF 提取文本、图片、图注。"""
from __future__ import annotations
import hashlib
import re
from pathlib import Path
from typing import List

from models import ImageRef
from parsers.base import DocumentParser, ParseResult
from parsers.utils import slugify, image_filename

_CAPTION_RE = re.compile(r"^\s*(图|Figure|Fig\.?)\s*\d+", re.IGNORECASE)


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
        text, images = self._attach_captions("\n".join(text_parts), images)
        return ParseResult(text=text, images=images, tables=[], _image_bytes=image_bytes_list)

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

    def _attach_captions(self, text: str, images: List[ImageRef]):
        """占位符后第一个匹配 _CAPTION_RE 的段落作为图注。"""
        lines = text.split("\n")
        img_idx = 0
        for line_no, line in enumerate(lines):
            if "{{IMG|" not in line or "图注: 待补" not in line:
                continue
            if img_idx >= len(images):
                break
            caption = ""
            for j in range(line_no + 1, min(line_no + 5, len(lines))):
                candidate = lines[j].strip()
                if candidate and _CAPTION_RE.match(candidate):
                    caption = candidate
                    break
            images[img_idx].caption = caption
            lines[line_no] = line.replace("图注: 待补", f"图注: {caption or '[无图注]'}")
            img_idx += 1
        return "\n".join(lines), images
