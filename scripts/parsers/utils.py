"""parsers 共享工具：slug、图片命名与图注绑定。"""
import re
from typing import List, Tuple

from models import ImageRef


def slugify(text: str) -> str:
    """转 URL-safe slug。保留中文，仅替换标点为 -。"""
    s = re.sub(r"[^\w\u4e00-\u9fff]", "-", text, flags=re.UNICODE)
    s = re.sub(r"-+", "-", s).strip("-").lower()
    return s


def image_filename(doc_slug: str, img_seq: int, ext: str) -> str:
    """生成图片文件名：{doc_slug}_img{NN:02d}.{ext}。"""
    return f"{doc_slug}_img{img_seq:02d}.{ext.lstrip('.')}"


_CAPTION_RE = re.compile(r"^\s*(图|Figure|Fig\.?)\s*\d+", re.IGNORECASE)


def attach_captions(text: str, images: List[ImageRef]) -> Tuple[str, List[ImageRef]]:
    """按行扫描 text，找到图片占位符后 5 行内匹配图注正则的段落作为图注。

    返回 (text, images)，其中 images 会被原地修改（按顺序写入 caption）。
    """
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
