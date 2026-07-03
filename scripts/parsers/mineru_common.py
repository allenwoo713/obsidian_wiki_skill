"""MinerU markdown -> ParseResult adapter.

Converts MinerU output (image refs + HTML tables) to unified ParseResult.
Supports merging multiple markdown segments (from split PDF parts).
"""
from __future__ import annotations
import hashlib
import re
from pathlib import Path
from typing import Dict, List

from models import ImageRef
from parsers.base import ParseResult
from parsers.utils import slugify, image_filename, attach_captions

_MINERU_IMG_RE = re.compile(r"!\[[^\]]*\]\(images/([^/)]+\.\w+)\)")


def mineru_markdown_to_parse_result(
    markdown: str,
    doc_slug: str,
    images_dir: Path,
    image_bytes_map: Dict[str, bytes],
) -> ParseResult:
    images: List[ImageRef] = []
    image_bytes_list: List[bytes] = []
    img_seq = 0

    def _replace_img(m: re.Match) -> str:
        nonlocal img_seq
        filename = m.group(1)
        img_bytes = image_bytes_map.get(filename, b"")
        sha = hashlib.sha256(img_bytes).hexdigest() if img_bytes else "unknown"
        img_seq += 1
        ext = filename.rsplit(".", 1)[-1] if "." in filename else "jpg"
        fname = image_filename(doc_slug, img_seq, ext)
        ref = ImageRef(
            filename=fname,
            rel_path=f"assets/{fname}",
            caption="",
            source_media_name=filename,
            sha256=sha,
            page_or_section="",
        )
        images.append(ref)
        image_bytes_list.append(img_bytes)
        return f"{{{{IMG|{ref.rel_path}|图注: 待补}}}}"

    text = _MINERU_IMG_RE.sub(_replace_img, markdown)
    tables = _extract_html_tables(text)
    text = _strip_html_tables(text)
    text, images = attach_captions(text, images)

    return ParseResult(
        text=text, images=images, tables=tables,
        _image_bytes=image_bytes_list,
    )


def _extract_html_tables(text: str) -> List[List[List[str]]]:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return []
    tables: List[List[List[str]]] = []
    soup = BeautifulSoup(text, "html.parser")
    for table_tag in soup.find_all("table"):
        rows: List[List[str]] = []
        for tr in table_tag.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
            if cells:
                rows.append(cells)
        if rows:
            tables.append(rows)
    return tables


def _strip_html_tables(text: str) -> str:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return text
    soup = BeautifulSoup(text, "html.parser")
    table_idx = 0
    for table_tag in soup.find_all("table"):
        table_idx += 1
        table_tag.replace_with(f"\n[table {table_idx}]\n")
    return str(soup)
