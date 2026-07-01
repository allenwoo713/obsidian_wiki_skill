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
