"""DocxParser：XML 遍历提取文本、图片、图注。"""
from __future__ import annotations
import hashlib
import re
import zipfile
from pathlib import Path
from typing import List, Optional
from xml.etree import ElementTree as ET

from models import ImageRef
from parsers.base import DocumentParser, ParseResult
from parsers.utils import slugify, image_filename


_NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "pic": "http://schemas.openxmlformats.org/drawingml/2006/picture",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
}

_CAPTION_RE = re.compile(r"^\s*(图|Figure|Fig\.?)\s*\d+", re.IGNORECASE)


class DocxParser(DocumentParser):
    def parse(self, path: Path) -> ParseResult:
        doc_slug = slugify(path.stem)
        with zipfile.ZipFile(str(path)) as z:
            doc_xml = z.read("word/document.xml")
            rels = self._read_rels(z)
            media_bytes = {n: z.read(n) for n in z.namelist() if n.startswith("word/media/")}

        root = ET.fromstring(doc_xml)
        body = root.find("w:body", _NS)
        if body is None:
            return ParseResult(text="", images=[], tables=[], _image_bytes=[])

        text_parts: List[str] = []
        images: List[ImageRef] = []
        image_bytes_list: List[bytes] = []
        img_seq = 0

        for elem in body:
            tag = self._local_tag(elem.tag)
            if tag == "p":
                paragraph_text, pic_elem = self._parse_paragraph(elem)
                if pic_elem is not None:
                    img_seq += 1
                    ref_and_bytes = self._make_image_ref(pic_elem, rels, media_bytes, doc_slug, img_seq)
                    if ref_and_bytes:
                        ref, img_bytes = ref_and_bytes
                        images.append(ref)
                        image_bytes_list.append(img_bytes)
                        text_parts.append(f"{{{{IMG|{ref.rel_path}|图注: 待补}}}}")
                if paragraph_text.strip():
                    text_parts.append(paragraph_text)
            elif tag == "tbl":
                table_md = self._parse_table(elem)
                if table_md:
                    text_parts.append("")
                    text_parts.append(table_md)
                    text_parts.append("")

        text, images = self._attach_captions("\n".join(text_parts), images)
        text = self._replace_image_placeholders(text)
        return ParseResult(text=text, images=images, tables=[], _image_bytes=image_bytes_list)

    def _read_rels(self, z: zipfile.ZipFile) -> dict:
        try:
            rels_xml = z.read("word/_rels/document.xml.rels")
        except KeyError:
            return {}
        root = ET.fromstring(rels_xml)
        rels = {}
        for rel in root:
            rid = rel.attrib.get("Id", "")
            target = rel.attrib.get("Target", "")
            if target.startswith("media/"):
                rels[rid] = "word/" + target
        return rels

    def _local_tag(self, full_tag: str) -> str:
        return full_tag.split("}")[-1]

    def _parse_paragraph(self, p_elem):
        text_parts = []
        pic_elem = None
        heading_level = self._get_heading_level(p_elem)
        for child in p_elem.iter():
            tag = self._local_tag(child.tag)
            if tag == "t":
                text_parts.append(child.text or "")
            elif tag == "pic":
                pic_elem = child
        text = "".join(text_parts)
        if heading_level and text.strip():
            text = "#" * heading_level + " " + text
        return text, pic_elem

    def _get_heading_level(self, p_elem) -> int:
        """从 w:pStyle 提取 heading 层级，非 heading 返回 0。"""
        pPr = p_elem.find("w:pPr", _NS)
        if pPr is None:
            return 0
        pStyle = pPr.find("w:pStyle", _NS)
        if pStyle is None:
            return 0
        _W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
        val = pStyle.attrib.get(f"{{{_W}}}val", "")
        m = re.match(r"[Hh]eading\s*(\d+)", val)
        if m:
            return int(m.group(1))
        if val.isdigit():
            return int(val)
        return 0

    def _make_image_ref(self, pic_elem, rels, media_bytes, doc_slug, img_seq):
        blip = None
        for child in pic_elem.iter():
            if self._local_tag(child.tag) == "blip":
                blip = child
                break
        if blip is None:
            return None
        embed_attr = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed"
        rid = blip.attrib.get(embed_attr, "")
        if not rid:
            return None
        media_path = rels.get(rid)
        if not media_path or media_path not in media_bytes:
            return None
        img_bytes = media_bytes[media_path]
        sha = hashlib.sha256(img_bytes).hexdigest()
        ext = Path(media_path).suffix.lstrip(".")
        fname = image_filename(doc_slug, img_seq, ext)
        source_media_name = Path(media_path).name
        ref = ImageRef(
            filename=fname,
            rel_path=f"assets/{fname}",
            caption="",
            source_media_name=source_media_name,
            sha256=sha,
            page_or_section="body",
        )
        return ref, img_bytes

    def _parse_table(self, tbl_elem) -> str:
        rows = []
        for tr in tbl_elem.findall("w:tr", _NS):
            cells = []
            for tc in tr.findall("w:tc", _NS):
                cell_text = []
                for t in tc.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t"):
                    cell_text.append(t.text or "")
                cells.append("".join(cell_text).strip())
            rows.append(cells)
        if not rows:
            return ""
        max_cols = max(len(r) for r in rows)
        lines = ["| " + " | ".join(r + [""] * (max_cols - len(r))) + " |" for r in rows if any(r)]
        if not lines:
            return ""
        header = lines[0]
        sep = "| " + " | ".join(["---"] * max_cols) + " |"
        return header + "\n" + sep + "\n" + "\n".join(lines[1:])

    def _attach_captions(self, text: str, images: List[ImageRef]):
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

    def _replace_image_placeholders(self, text: str) -> str:
        """把 {{IMG|assets/xxx.png|图注: caption}} 替换为 ![[xxx.png]] + caption 可读文本。

        对齐 MinerU 输出格式：图片用 Obsidian ![[filename]] 嵌入，图注作为下一行纯文本。
        """
        def _repl(m):
            rel_path = m.group(1)
            caption = m.group(2)
            filename = Path(rel_path).name
            if caption and caption != "[无图注]":
                return f"![[{filename}]]  \n{caption}"
            return f"![[{filename}]]"
        text = re.sub(r"\{\{IMG\|([^|]+)\|图注: ([^}]*)\}\}", _repl, text)
        # 去重：占位符释放的 caption 与紧跟的原文档 caption 段落重复时删除后者
        text = re.sub(r"(!\[\[[^\]]+\]\]  \n)([^\n]+)\n\2\n", r"\1\2\n", text)
        return text
