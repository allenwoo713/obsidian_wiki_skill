"""FirecrawlPdfParser：调用 Firecrawl /parse API + markdown 适配 ParseResult。"""
from __future__ import annotations
import base64
import hashlib
import json
import re
from pathlib import Path
from typing import List

from models import ImageRef
from parsers.base import DocumentParser, ParseResult
from parsers.utils import slugify, image_filename, attach_captions

_IMG_RE = re.compile(r"!\[([^\]]*)\]\(data:image/(png|jpeg|jpg);base64,([^)]+)\)")
_MD_TABLE_RE = re.compile(r"(?:^\|.+\|\n)(?:^\|[\s:|-]+\|\n)(?:^\|.+\|\n?)+", re.MULTILINE)


def _markdown_to_parse_result(path: Path, markdown: str, local_images: List[ImageRef] = None, local_image_bytes: List[bytes] = None) -> ParseResult:
    """将 Firecrawl 返回的 markdown 解析为 ParseResult。

    混合模式：text/tables 来自 Firecrawl；images 来自本地 PdfParser（含 caption）。
    Firecrawl 对 PDF 当前不提取嵌入图片，必须用本地图片补齐。
    """
    doc_slug = slugify(path.stem)

    # Firecrawl markdown 中若含 base64 图片（理论上 PDF 不会，但防御性处理），提取之
    images: List[ImageRef] = []
    image_bytes_list: List[bytes] = []
    img_seq = 0

    def _replace_img(m):
        nonlocal img_seq
        alt, ext, b64 = m.group(1), m.group(2), m.group(3)
        img_bytes = base64.b64decode(b64)
        sha = hashlib.sha256(img_bytes).hexdigest()
        img_seq += 1
        fname = image_filename(doc_slug, img_seq, ext)
        ref = ImageRef(
            filename=fname, rel_path=f"assets/{fname}", caption="",
            source_media_name="firecrawl", sha256=sha, page_or_section="",
        )
        images.append(ref)
        image_bytes_list.append(img_bytes)
        return f"{{{{IMG|{ref.rel_path}|图注: 待补}}}}"

    text = _IMG_RE.sub(_replace_img, markdown)

    # 解析 markdown 表格
    tables: List[List[List[str]]] = []

    def _extract_table(m):
        block = m.group(0)
        lines = [ln for ln in block.strip().split("\n") if ln.strip()]
        rows = []
        for ln in lines:
            if re.match(r"^\|[\s:|-]+\|$", ln):
                continue
            cells = [c.strip() for c in ln.strip("|").split("|")]
            rows.append(cells)
        if rows:
            tables.append(rows)
        return ""

    text = _MD_TABLE_RE.sub(_extract_table, text)

    # 表格段追加到 text 末尾
    if tables:
        text = text.rstrip() + "\n\n[表格]\n"
        for i, t in enumerate(tables):
            text += f"\n表 {i+1}:\n"
            for row in t:
                text += " | ".join(row) + "\n"

    # 混合模式：若提供了本地图片（含 caption），用本地图片补齐 Firecrawl 的图片缺失
    if local_images is not None:
        # Firecrawl markdown 中无图片引用 → images 列表应为空
        # 直接使用本地图片（已含 caption，已按页序排列）
        images = list(local_images)
        image_bytes_list = list(local_image_bytes) if local_image_bytes else []
        # 图片占位符追加到 text 末尾，便于 RAG 检索到图片存在
        for ref in images:
            caption_str = ref.caption or "[无图注]"
            text += f"\n{{{{IMG|{ref.rel_path}|图注: {caption_str}}}}}\n"
        # 不再调 attach_captions——本地图片已有 caption
    else:
        # 纯 Firecrawl 模式（markdown 中含 base64 图片时，走原有逻辑）
        text, images = attach_captions(text, images)

    return ParseResult(
        text=text, images=images, tables=tables,
        _image_bytes=image_bytes_list,
    )


def _requests_post(url, **kwargs):
    """requests.post 间接层，便于测试 mock。"""
    import requests
    return requests.post(url, **kwargs)


def _load_env_if_present():
    """若 skill 根目录存在 .env 且 python-dotenv 可用，加载之。失败静默。

    .env 为可选配置途径；未安装 python-dotenv 时回退到系统环境变量。
    """
    try:
        from dotenv import load_dotenv
        skill_root = Path(__file__).resolve().parents[2]  # scripts/parsers/ -> skill root
        load_dotenv(skill_root / ".env", override=False)
    except ImportError:
        pass


class FirecrawlPdfParser(DocumentParser):
    """调用 Firecrawl /parse API 解析 PDF，失败回退本地 PdfParser。"""

    API_URL = "https://api.firecrawl.dev/v2/parse"
    TIMEOUT_SEC = 60.0

    def parse(self, path: Path) -> ParseResult:
        _load_env_if_present()
        import os
        api_key = os.environ.get("FIRECRAWL_API_KEY", "")
        if not api_key:
            print("WARNING: FIRECRAWL_API_KEY 未设置，回退本地 PdfParser")
            from parsers.pdf_parser import PdfParser
            return PdfParser().parse(path)

        try:
            with open(path, "rb") as f:
                file_bytes = f.read()
        except OSError as e:
            print(f"WARNING: 读取 PDF 失败 ({e})，回退本地 PdfParser")
            from parsers.pdf_parser import PdfParser
            return PdfParser().parse(path)

        try:
            resp = _requests_post(
                self.API_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                files={"file": (path.name, file_bytes, "application/pdf")},
                data={
                    "options": json.dumps({
                        "formats": ["markdown", "images"],
                        "onlyMainContent": False,
                    })
                },
                timeout=self.TIMEOUT_SEC,
            )
            resp.raise_for_status()
            payload = resp.json()
        except Exception as e:
            print(f"WARNING: Firecrawl /parse 请求失败 ({type(e).__name__}: {e})，回退本地 PdfParser")
            from parsers.pdf_parser import PdfParser
            return PdfParser().parse(path)

        if not payload.get("success"):
            print("WARNING: Firecrawl 返回 success=false，回退本地 PdfParser")
            from parsers.pdf_parser import PdfParser
            return PdfParser().parse(path)

        data = payload.get("data", {})
        markdown = data.get("markdown", "")
        if not markdown:
            print("WARNING: Firecrawl 返回空 markdown，回退本地 PdfParser")
            from parsers.pdf_parser import PdfParser
            return PdfParser().parse(path)

        # 混合模式：Firecrawl 对 PDF 不提取嵌入图片，用本地 PdfParser 补齐 images（含 caption）
        # text 和 tables 来自 Firecrawl（表格更干净，markdown 结构化）
        # images 和 image_bytes 来自本地 PyMuPDF（含图注关联）
        try:
            from parsers.pdf_parser import PdfParser
            local_result = PdfParser().parse(path)
            local_images = local_result.images
            local_image_bytes = local_result._image_bytes
        except Exception as e:
            print(f"WARNING: 本地 PdfParser 图片提取失败 ({type(e).__name__}: {e})，images 将为空")
            local_images = None
            local_image_bytes = None

        return _markdown_to_parse_result(
            path, markdown,
            local_images=local_images,
            local_image_bytes=local_image_bytes,
        )
