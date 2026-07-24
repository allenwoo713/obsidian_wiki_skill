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
from parsers.utils import slugify, image_filename, attach_captions, replace_image_placeholders

_MINERU_IMG_RE = re.compile(r"!\[([^\]]*)\]\(images/([^/)]+\.\w+)\)")
# MinerU 在 HTML 表格内以 <img src="images/<hash>.jpg" alt="..."> 输出图，
# 旧版正则只匹配 Markdown ![...](images/...)，会漏掉这些表格图（字节不写、图注丢）。
_HTML_IMG_RE = re.compile(r'<img[^>]+src="images/([^"]+\.\w+)"[^>]*>')


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
        alt = m.group(1).strip()
        filename = m.group(2)
        img_bytes = image_bytes_map.get(filename, b"")
        sha = hashlib.sha256(img_bytes).hexdigest() if img_bytes else "unknown"
        img_seq += 1
        ext = filename.rsplit(".", 1)[-1] if "." in filename else "jpg"
        fname = image_filename(doc_slug, img_seq, ext)
        ref = ImageRef(
            filename=fname,
            rel_path=f"assets/{fname}",
            caption=alt,  # 保留 MinerU 图注（alt 文本），不再丢弃
            source_media_name=filename,
            sha256=sha,
            page_or_section="",
        )
        images.append(ref)
        image_bytes_list.append(img_bytes)
        return f"{{{{IMG|{ref.rel_path}|图注: 待补}}}}"

    def _replace_html_img(m: re.Match) -> str:
        # MinerU 在 HTML 表格内输出的图：写字节、保留原始 hash 文件名、
        # 图注取自 alt 属性，引用改写为 Obsidian ![[...]] 以进入检索与图谱。
        filename = m.group(1)
        alt_m = re.search(r'alt="([^"]*)"', m.group(0))
        alt = alt_m.group(1).strip() if alt_m else ""
        img_bytes = image_bytes_map.get(filename, b"")
        sha = hashlib.sha256(img_bytes).hexdigest() if img_bytes else "unknown"
        ref = ImageRef(
            filename=filename,
            rel_path=f"assets/{filename}",
            caption=alt,
            source_media_name=filename,
            sha256=sha,
            page_or_section="",
        )
        images.append(ref)
        image_bytes_list.append(img_bytes)
        cap_line = f"  \n{alt}" if alt else ""
        return f"![[{filename}]]{cap_line}"

    text = _MINERU_IMG_RE.sub(_replace_img, markdown)
    # 处理 HTML 表格内的 <img> 图（须在 _extract_html_tables 之前，避免表格结构干扰）
    text = _HTML_IMG_RE.sub(_replace_html_img, text)
    tables = _extract_html_tables(text)
    # 保留 HTML 表格在 text 中（Obsidian 可渲染 HTML），
    # 不再用 [table N] 占位符替换，避免下游输出丢失表格内容。
    # tables 字段仍提取结构化数据供 FTS/RAG 索引器使用。
    text, images = attach_captions(text, images)
    text = replace_image_placeholders(text)

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
