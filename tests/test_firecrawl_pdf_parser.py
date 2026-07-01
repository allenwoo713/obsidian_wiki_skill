"""FirecrawlPdfParser 测试：markdown 解析 + 回退。"""
import sys
import base64
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


def _b64_png():
    """1x1 PNG 的 base64 编码。"""
    import struct, zlib
    def chunk(t, d):
        c = t + d
        return struct.pack(">I", len(d)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    idat = zlib.compress(b"\x00\x00\x00\x00")
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")
    return base64.b64encode(png).decode()


def test_parse_markdown_extracts_base64_image():
    """纯 Firecrawl 模式（无 local_images）：markdown 中 base64 图片被提取。"""
    from parsers.firecrawl_pdf_parser import _markdown_to_parse_result
    b64 = _b64_png()
    md = f"Body text.\n\n![Figure 1 Diagram](data:image/png;base64,{b64})\n\nMore text."
    result = _markdown_to_parse_result(Path("test.pdf"), md)
    assert len(result.images) == 1
    assert len(result.images[0].sha256) == 64
    assert result.images[0].filename.endswith(".png")
    assert "{{IMG|" in result.text


def test_parse_markdown_extracts_table():
    from parsers.firecrawl_pdf_parser import _markdown_to_parse_result
    md = (
        "Intro text.\n\n"
        "| Field | Offset | Length |\n"
        "|---|---|---|\n"
        "| Header | 0 | 4 |\n"
        "| Payload | 4 | N |\n\n"
        "Outro text."
    )
    result = _markdown_to_parse_result(Path("test.pdf"), md)
    assert len(result.tables) == 1
    assert result.tables[0][0] == ["Field", "Offset", "Length"]
    assert result.tables[0][1] == ["Header", "0", "4"]
    assert "[表格]" in result.text
    assert "表 1:" in result.text
    assert "Intro text" in result.text
    assert "Outro text" in result.text


def test_parse_markdown_caption_attached():
    from parsers.firecrawl_pdf_parser import _markdown_to_parse_result
    b64 = _b64_png()
    md = f"![Diagram](data:image/png;base64,{b64})\n\nFigure 1 FOV Coverage\n\nbody"
    result = _markdown_to_parse_result(Path("test.pdf"), md)
    assert any("Figure" in img.caption for img in result.images)


def test_parse_falls_back_on_network_error(tmp_path, monkeypatch):
    """网络错误时回退本地 PdfParser。"""
    import fitz
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "fallback body", fontsize=12)
    pdf_path = tmp_path / "fb.pdf"
    doc.save(str(pdf_path))
    doc.close()

    import parsers.firecrawl_pdf_parser as mod

    def _fake_post(*a, **kw):
        raise ConnectionError("simulated network error")

    monkeypatch.setattr(mod, "_requests_post", _fake_post)
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fake-key")

    from parsers.firecrawl_pdf_parser import FirecrawlPdfParser
    result = FirecrawlPdfParser().parse(pdf_path)
    assert "fallback body" in result.text


def test_parse_falls_back_on_success_false(tmp_path, monkeypatch):
    """API 返回 success=false 时回退本地。"""
    import fitz
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "body", fontsize=12)
    pdf_path = tmp_path / "fb2.pdf"
    doc.save(str(pdf_path))
    doc.close()

    import parsers.firecrawl_pdf_parser as mod

    class _FakeResp:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return {"success": False, "data": {}}

    monkeypatch.setattr(mod, "_requests_post", lambda *a, **kw: _FakeResp())
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fake-key")

    from parsers.firecrawl_pdf_parser import FirecrawlPdfParser
    result = FirecrawlPdfParser().parse(pdf_path)
    assert "body" in result.text


def test_parse_no_api_key_falls_back(tmp_path, monkeypatch):
    """FIRECRAWL_API_KEY 未设置时回退本地。"""
    import fitz
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "nokey body", fontsize=12)
    pdf_path = tmp_path / "nk.pdf"
    doc.save(str(pdf_path))
    doc.close()

    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)

    from parsers.firecrawl_pdf_parser import FirecrawlPdfParser
    result = FirecrawlPdfParser().parse(pdf_path)
    assert "nokey body" in result.text


def test_parse_mixed_mode_uses_local_images(tmp_path, monkeypatch):
    """混合模式：text/tables 来自 Firecrawl，images 来自本地 PdfParser（含 caption）。"""
    import fitz
    from models import ImageRef
    import parsers.firecrawl_pdf_parser as mod

    # 构造含图片的 PDF（本地 PdfParser 能提取）
    # 注意：PyMuPDF 默认字体不支持中文，用英文 caption 避免编码问题
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "firecrawl text", fontsize=12)
    import io
    from PIL import Image as PILImage
    img_buf = io.BytesIO()
    PILImage.new("RGB", (10, 10), (255, 0, 0)).save(img_buf, format="PNG")
    img_bytes = img_buf.getvalue()
    page.insert_image(fitz.Rect(100, 100, 110, 110), stream=img_bytes)
    page.insert_text((72, 130), "Figure 1 Test Diagram", fontsize=10)
    pdf_path = tmp_path / "mixed.pdf"
    doc.save(str(pdf_path))
    doc.close()

    # Mock Firecrawl 返回不含图片的 markdown
    class _FakeResp:
        status_code = 200
        def raise_for_status(self): pass
        def json(self):
            return {"success": True, "data": {"markdown": "# Title\n\nfirecrawl text\n\n| Col1 | Col2 |\n|---|---|\n| a | b |"}}

    monkeypatch.setattr(mod, "_requests_post", lambda *a, **kw: _FakeResp())
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fake-key")

    from parsers.firecrawl_pdf_parser import FirecrawlPdfParser
    result = FirecrawlPdfParser().parse(pdf_path)

    # text 和 tables 来自 Firecrawl
    assert "firecrawl text" in result.text
    assert len(result.tables) == 1
    assert result.tables[0][0] == ["Col1", "Col2"]
    assert "[表格]" in result.text

    # images 来自本地 PdfParser（含 caption）
    assert len(result.images) >= 1
    assert len(result._image_bytes) >= 1
    # 图片占位符应在 text 中（追加在末尾）
    assert "{{IMG|" in result.text
    # 至少有一个图片含图注（本地 PdfParser 提取的 "Figure 1 Test Diagram"）
    captions = [img.caption for img in result.images if img.caption]
    assert any("Figure" in c or "Test" in c for c in captions), f"captions: {captions}"


def test_parse_mixed_mode_local_fails_images_empty(tmp_path, monkeypatch):
    """混合模式：本地图片提取失败时 images 为空，text/tables 仍来自 Firecrawl。"""
    import fitz
    import parsers.firecrawl_pdf_parser as mod

    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "body", fontsize=12)
    pdf_path = tmp_path / "nofail.pdf"
    doc.save(str(pdf_path))
    doc.close()

    class _FakeResp:
        status_code = 200
        def raise_for_status(self): pass
        def json(self):
            return {"success": True, "data": {"markdown": "firecrawl body"}}

    monkeypatch.setattr(mod, "_requests_post", lambda *a, **kw: _FakeResp())
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fake-key")

    # Mock PdfParser.parse 抛异常
    from parsers.pdf_parser import PdfParser
    original_parse = PdfParser.parse
    def _failing_parse(self, path):
        raise RuntimeError("simulated local parse failure")
    monkeypatch.setattr(PdfParser, "parse", _failing_parse)

    from parsers.firecrawl_pdf_parser import FirecrawlPdfParser
    result = FirecrawlPdfParser().parse(pdf_path)

    # text 仍来自 Firecrawl
    assert "firecrawl body" in result.text
    # images 为空（本地提取失败）
    assert len(result.images) == 0
    assert len(result._image_bytes) == 0
