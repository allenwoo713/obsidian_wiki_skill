"""parsers 共享工具：slug 与图片命名。"""
import re


def slugify(text: str) -> str:
    """转 URL-safe slug。保留中文，仅替换标点为 -。"""
    s = re.sub(r"[^\w\u4e00-\u9fff]", "-", text, flags=re.UNICODE)
    s = re.sub(r"-+", "-", s).strip("-").lower()
    return s


def image_filename(doc_slug: str, img_seq: int, ext: str) -> str:
    """生成图片文件名：{doc_slug}_img{NN:02d}.{ext}。"""
    return f"{doc_slug}_img{img_seq:02d}.{ext.lstrip('.')}"
